import torch
import numpy as np
# Předpokládám, že MDM_functions jsou dostupné, jak je v zadání
from MDM.MDM_functions import MDM_nullO_LTI, pinv, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

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

        # Interní stav pro online použití
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
        P_filtered = (I - K @ self.H) @ P_predict @ (I - K @ self.H).T + K @ self.R @ K.T # Joseph form
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

import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

class AdaptiveKalmanFilter_online:
    def __init__(self, model, mdm_L=4, mdm_version=1, forgetting_factor=0.995):
        """
        Adaptivní KF (Online verze) kompatibilní s eval skriptem.
        """
        self.kf = KalmanFilter(model) # Ujisti se, že voláš správnou třídu KF
        self.device = model.Q.device
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        
        # --- RLS Init ---
        self.lambda_rls = forgetting_factor
        self.nw = self.kf.F.shape[0]
        self.nv = self.kf.H.shape[0]
        self.n_params = (self.nw * (self.nw + 1)) // 2 + (self.nv * (self.nv + 1)) // 2
        
        # 1. Inicializace odhadu (alpha) z modelu (nikoliv z nuly!)
        # Tím zajistíme, že filtr začne se "solidními" maticemi Q a R
        Q0 = model.Q.cpu().numpy()
        R0 = model.R.cpu().numpy()
        
        i_q, j_q = np.triu_indices(self.nw)
        alpha_q = Q0[i_q, j_q]
        
        i_r, j_r = np.triu_indices(self.nv)
        alpha_r = R0[i_r, j_r]
        
        self.alpha_est = np.concatenate([alpha_q, alpha_r])
        
        # 2. Inicializace kovariance RLS
        self.Sigma_RLS = np.eye(self.n_params) * 100.0 # Počáteční nejistota parametrů

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

    def _reconstruct_qr_from_alpha2(self, alpha_2):
        q_len = (self.nw * (self.nw + 1)) // 2
        alpha_q = alpha_2[:q_len]
        alpha_r = alpha_2[q_len:]
        
        Q_est = np.zeros((self.nw, self.nw))
        i, j = np.triu_indices(self.nw)
        Q_est[i, j] = alpha_q
        Q_est[j, i] = alpha_q
        
        R_est = np.zeros((self.nv, self.nv))
        i, j = np.triu_indices(self.nv)
        R_est[i, j] = alpha_r
        R_est[j, i] = alpha_r
        
        return Q_est, R_est

    def _project_to_psd(self, M, epsilon=1e-8):
        """Zajistí pozitivní definitnost matice (nutné pro KF)."""
        # Symmetrizace
        M = (M + M.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M)
        # Pokud jsou vlastní čísla příliš malá nebo záporná, opravíme je
        if np.any(eigvals < epsilon):
            eigvals[eigvals < epsilon] = epsilon
            M = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return M

    def step_adaptive(self, y_t, u_t=None):
        # 1. Update Bufferu
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

        # 2. Adaptivní část (MDM + RLS)
        if len(self.z_buffer) == self.mdm_L:
            try:
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                # Parametry pro MDM
                F_np = self.kf.F.cpu().numpy()
                H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1])) 
                E_np = np.eye(self.nw)
                D_np = np.eye(self.nv)
                nz_np = np.array([self.nv])

                # Volání MDM
                r_list, Awv_matrix = MDM_nullO_LTI(
                    self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
                    z_window, u_window, self.mdm_version
                )
                r_k = r_list[0]
                
                # RLS Update
                nr = r_k.shape[0]
                Ksi = Ksi_fun(nr)
                y_rls = Ksi @ kron2_vec(r_k)
                
                Upsilon = self._get_upsilon()
                H_rls = Ksi @ kron2_mat(Awv_matrix) @ Upsilon

                # Regularizovaná inverze pro stabilitu
                S = H_rls @ self.Sigma_RLS @ H_rls.T + self.lambda_rls * np.eye(H_rls.shape[0])
                K_gain_rls = self.Sigma_RLS @ H_rls.T @ np.linalg.pinv(S) # pinv je bezpečnější než inv
                
                pred_error = y_rls - H_rls @ self.alpha_est
                self.alpha_est = self.alpha_est + K_gain_rls @ pred_error
                
                I_p = np.eye(self.n_params)
                self.Sigma_RLS = (1.0 / self.lambda_rls) * (I_p - K_gain_rls @ H_rls) @ self.Sigma_RLS
                
                # Rekonstrukce a kontrola PSD
                Q_new, R_new = self._reconstruct_qr_from_alpha2(self.alpha_est)
                Q_new = self._project_to_psd(Q_new, epsilon=1e-6)
                R_new = self._project_to_psd(R_new, epsilon=1e-3) # Větší epsilon pro R (GPS nesmí být 0)
                
                # Update KF
                self.kf.Q = torch.from_numpy(Q_new).float().to(self.device)
                self.kf.R = torch.from_numpy(R_new).float().to(self.device)
                
            except Exception as e:
                # Pokud MDM selže (singularity atd.), ignorujeme tento krok adaptace 
                # a jedeme dál s původním Q a R.
                # print(f"MDM Error: {e}") # Odkomentuj pro debug
                pass

        # 3. Filtrace
        x_filt, P_filt = self.kf.step(y_t)
        return x_filt, P_filt, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Simuluje online běh.
        Vrací: (dict_vysledku, list_Q, list_R) - přesně jak čeká eval script.
        
        Args:
            y_seq: Sekvence měření.
            u_seq: Sekvence řízení (volitelné).
            Ex0: Počáteční stav (pokud None, použije se default z modelu).
            P0: Počáteční kovariance (pokud None, použije se default z modelu).
        """
        seq_len = y_seq.shape[0]
        
        # 1. Kompletní reset RLS estimátoru a vnitřních proměnných
        # Voláme __init__, abychom vyčistili Sigma_RLS, alpha_est a vytvořili čistý KF.
        # Tím zajistíme, že si neneseme chyby z předchozí trajektorie.
        self.__init__(self.kf.model, self.mdm_L, self.mdm_version, self.lambda_rls)

        # 2. Určení počátečních podmínek
        if Ex0 is None:
            Ex0 = self.kf.model.Ex0
        if P0 is None:
            P0 = self.kf.model.P0

        # 3. Reset Kalmanova filtru na SPRÁVNÝ startovní bod
        # Toto musíme udělat až po __init__, jinak by nám to init přepsal.
        self.kf.reset(Ex0, P0)
        
        # Buffery jsou již prázdné díky __init__, ale pro jistotu:
        self.z_buffer = []
        self.u_buffer = []

        x_hist = torch.zeros(seq_len, self.kf.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.kf.state_dim, self.kf.state_dim, device=self.device)
        Q_hist = []
        R_hist = []

        for k in range(seq_len):
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