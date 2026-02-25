import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, pinv, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat
from tqdm import tqdm

class KalmanFilter:
    """
    Kalmanův filtr pro t-invaritantní systém s lineární dynamikou.
    (Původní implementace beze změn)
    """
    def __init__(self,model):
        self.device = model.Q.device
        self.model = model
        self.Ex0 = model.Ex0
        self.P0 = model.P0
        self.F = model.F
        self.H = model.H
        self.Q = model.Q
        self.R = model.R
        self.state_dim = self.F.shape[0]
        self.obs_dim = self.H.shape[0]

        self.x_predict_current = None
        self.P_predict_current = None
        self.reset(model.Ex0, model.P0)

    def reset(self, Ex0, P0):
        self.x_predict_current = Ex0.clone().detach().reshape(self.state_dim, 1)
        self.P_predict_current = P0.clone().detach()

    def predict_step(self, x_filtered, P_filtered):
        x_predict = self.F @ x_filtered
        P_predict = self.F @ P_filtered @ self.F.T + self.Q
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        y_t = y_t.reshape(self.obs_dim, 1)
        innovation = self.compute_innovation(y_t, x_predict)
        K = self.compute_kalman_gain(P_predict)
        x_filtered = x_predict + K @ innovation
        I = torch.eye(self.state_dim, device=self.device)
        P_filtered = (I - K @ self.H) @ P_predict @ (I - K @ self.H).T + K @ self.R @ K.T 
        return x_filtered, P_filtered, K, innovation

    def compute_kalman_gain(self, P_predict):
        return P_predict @ self.H.T @ torch.linalg.inv(self.H @ P_predict @ self.H.T + self.R)
    
    def compute_innovation(self, y_t, x_predict):
        return y_t - self.H @ x_predict
    
    def step(self, y_t):
        x_filtered, P_filtered, _, _ = self.update_step(self.x_predict_current, y_t, self.P_predict_current)
        x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered)
        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next
        return x_filtered, P_filtered

    def process_sequence(self, y_seq, Ex0=None, P0=None):
        seq_len = y_seq.shape[0]
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        kalman_gain_history = torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device)
        innovation_history = torch.zeros(seq_len, self.obs_dim, device=self.device)

        if Ex0 is None: Ex0 = self.Ex0
        if P0 is None: P0 = self.P0
        x_predict_k = Ex0.clone().detach().reshape(self.state_dim, 1)
        P_predict_k = P0.clone().detach()
        
        for k in range(seq_len):
            x_filtered, P_filtered, K, innovation = self.update_step(x_predict_k, y_seq[k], P_predict_k)
            x_predict_k_plus_1, P_predict_k_plus_1 = self.predict_step(x_filtered, P_filtered)
            x_predict_k = x_predict_k_plus_1
            P_predict_k = P_predict_k_plus_1
            x_filtered_history[k] = x_filtered.squeeze()
            P_filtered_history[k] = P_filtered
            kalman_gain_history[k] = K
            innovation_history[k] = innovation.squeeze()

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'Kalman_gain': kalman_gain_history,
            'innovation': innovation_history
        }

class AdaptiveKalmanFilter_default:
    def __init__(self, model, mdm_L=4, mdm_version=1, lambda_rls=0.995, init_sigma_rls_value=10.0, alpha_nom=None):
        """
        Adaptivní KF (Odhad plných matic Q a R).
        Tato třída provádí online identifikaci všech unikátních prvků matic Q a R.
        """
        self.kf = KalmanFilter(model)
        self.device = model.Q.device
        
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        self.lambda_rls = lambda_rls
        self.init_sigma_val = init_sigma_rls_value
        
        self.nw = self.kf.F.shape[0] 
        self.nv = self.kf.H.shape[0] 
        
        # 1. POČET PARAMETRŮ PRO PLNÉ MATICE
        # Pro symetrickou matici NxN odhadujeme pouze horní trojúhelník: N*(N+1)/2 prvků
        self.n_params_Q = (self.nw * (self.nw + 1)) // 2 
        self.n_params_R = (self.nv * (self.nv + 1)) // 2
        self.n_params = self.n_params_Q + self.n_params_R
        
        # 2. INICIALIZACE RLS Z POČÁTEČNÍCH MATIC
        Q0 = model.Q.cpu().numpy()
        R0 = model.R.cpu().numpy()
        
        # Vytáhneme pouze unikátní prvky (horní trojúhelník) do 1D vektoru
        q_init_vec = Q0[np.triu_indices(self.nw)]
        r_init_vec = R0[np.triu_indices(self.nv)]
        
        # alpha_est nyní obsahuje všechny prvky Q následované všemi prvky R
        self.alpha_est = np.concatenate([q_init_vec, r_init_vec]).astype(np.float64)
        
        # Kovarianční matice RLS (jelikož už nemáme alpha_nom, použijeme čistou identitu škálovanou konstantou)
        self.Sigma_RLS = np.eye(self.n_params) * self.init_sigma_val
        
        # Buffery
        self.z_buffer = []
        self.u_buffer = []
        self.Upsilon_2 = None

    def _get_upsilon(self):
        if self.Upsilon_2 is None:
            w2b = [baseMatrix_fun(self.nw, 1)] 
            v2b = [baseMatrix_fun(self.nv, 1)] 
            self.Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        return self.Upsilon_2

    def _enforce_psd(self, matrix):
        """Pomocná funkce: Vynutí, aby matice byla pozitivně semidefinitní a plně symetrická."""
        # 1. Vynucení čisté symetrie
        sym_matrix = (matrix + matrix.T) / 2.0
        # 2. Oříznutí záporných vlastních čísel
        eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)
        eigenvalues[eigenvalues < 1e-10] = 1e-10
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def step_adaptive(self, y_t, u_t=None):
        y_np = y_t.cpu().numpy().squeeze()
        if y_np.ndim == 0: y_np = np.expand_dims(y_np, axis=0)
        
        if u_t is not None:
            u_np = u_t.cpu().numpy().squeeze()
            if u_np.ndim == 0: u_np = np.expand_dims(u_np, axis=0)
        else:
            u_np = np.zeros(1)

        self.z_buffer.append(y_np)
        self.u_buffer.append(u_np)
        
        if len(self.z_buffer) > self.mdm_L:
            self.z_buffer.pop(0)
            self.u_buffer.pop(0)

        if len(self.z_buffer) == self.mdm_L:
            try:
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                F_np = self.kf.F.cpu().numpy()
                H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1])) 
                E_np = np.eye(self.nw)
                D_np = np.eye(self.nv)
                nz_np = np.array([self.nv])

                r_list, Awv_matrix = MDM_nullO_LTI(
                    self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
                    z_window, u_window, self.mdm_version
                )
                r_k = r_list[0]
                nr = r_k.shape[0]
                
                Ksi = Ksi_fun(nr) 
                y_rls = Ksi @ kron2_vec(r_k)
                
                Upsilon = self._get_upsilon()
                H_full = Ksi @ kron2_mat(Awv_matrix) @ Upsilon

                H_Q_part_full = H_full[:, :self.n_params_Q] 
                H_R_part_full = H_full[:, self.n_params_Q:]
                
                # --- ZMĚNA: Žádné transformační šablony, spojíme matice přímo ---
                H_rls = np.hstack([H_Q_part_full, H_R_part_full])
                y_rls_flat = y_rls.flatten()
                
                # --- Standardní RLS Update ---
                dim_obs_rls = H_rls.shape[0]
                Omega = np.eye(dim_obs_rls)

                S = H_rls @ self.Sigma_RLS @ H_rls.T + Omega
                K_gain = self.Sigma_RLS @ H_rls.T @ np.linalg.pinv(S)
                
                error = y_rls_flat - H_rls @ self.alpha_est
                self.alpha_est = self.alpha_est + K_gain @ error
                
                I_p = np.eye(self.n_params)
                self.Sigma_RLS = (I_p - K_gain @ H_rls) @ self.Sigma_RLS / self.lambda_rls              
                
                # --- REKONSTRUKCE CELÝCH MATIC Z VEKTORU ---
                q_est_vec = self.alpha_est[:self.n_params_Q]
                r_est_vec = self.alpha_est[self.n_params_Q:]
                
                # Rekonstrukce Q
                Q_new = np.zeros((self.nw, self.nw))
                Q_new[np.triu_indices(self.nw)] = q_est_vec
                Q_new = Q_new + Q_new.T - np.diag(np.diag(Q_new)) # Doplnění spodního trojúhelníku
                Q_new = self._enforce_psd(Q_new) # Zajištění poz. semidefinitnosti
                
                # Rekonstrukce R
                R_new = np.zeros((self.nv, self.nv))
                R_new[np.triu_indices(self.nv)] = r_est_vec
                R_new = R_new + R_new.T - np.diag(np.diag(R_new)) # Doplnění spodního trojúhelníku
                R_new = self._enforce_psd(R_new) # Zajištění poz. semidefinitnosti
                
                # Update KF
                self.kf.Q = torch.from_numpy(Q_new).float().to(self.device)
                self.kf.R = torch.from_numpy(R_new).float().to(self.device)
                
            except Exception as e:
                print(f"Chyba v RLS aktualizaci: {e}")
                pass

        x_filt, P_filt = self.kf.step(y_t)
        return x_filt, P_filt, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        seq_len = y_seq.shape[0]
        
        self.__init__(
            self.kf.model, 
            self.mdm_L, 
            self.mdm_version, 
            self.lambda_rls,
            init_sigma_rls_value=self.init_sigma_val,
            alpha_nom=None # Ošetřeno pro kompatibilitu volání
        )

        if Ex0 is None: Ex0 = self.kf.model.Ex0
        if P0 is None: P0 = self.kf.model.P0

        self.kf.reset(Ex0, P0)
        
        self.z_buffer = []
        self.u_buffer = []

        x_hist = torch.zeros(seq_len, self.kf.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.kf.state_dim, self.kf.state_dim, device=self.device)
        Q_hist = []
        R_hist = []

        for k in tqdm(range(seq_len), desc="Processing sequence adaptively"):
            y_t = y_seq[k]
            u_t = u_seq[k] if u_seq is not None else None
            
            x, P, Q_curr, R_curr = self.step_adaptive(y_t, u_t)
            
            x_hist[k] = x.squeeze()
            P_hist[k] = P
            Q_hist.append(Q_curr.clone().detach())
            R_hist.append(R_curr.clone().detach())
            
        results_dict = {
            'x_filtered': x_hist,
            'P_filtered': P_hist
        }
        
        return results_dict, Q_hist, R_hist