import torch
import numpy as np
from tqdm import tqdm
from MDM.MDM_functions import MDM_nullO_LTI, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

class KalmanFilter:
    """
    Kalmanův filtr (beze změn, ponecháno pro kompatibilitu).
    """
    def __init__(self, model):
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
        innovation = y_t - self.H @ x_predict
        # Kalman Gain
        S = self.H @ P_predict @ self.H.T + self.R
        K = P_predict @ self.H.T @ torch.linalg.inv(S)
        
        x_filtered = x_predict + K @ innovation
        # Joseph form update (numericky stabilnější)
        I = torch.eye(self.state_dim, device=self.device)
        ImKH = I - K @ self.H
        P_filtered = ImKH @ P_predict @ ImKH.T + K @ self.R @ K.T
        return x_filtered, P_filtered

    def step(self, y_t):
        x_filtered, P_filtered = self.update_step(self.x_predict_current, y_t, self.P_predict_current)
        self.x_predict_current, self.P_predict_current = self.predict_step(x_filtered, P_filtered)
        return x_filtered, P_filtered

class FastAdaptiveKalmanFilter:
    def __init__(self, model, mdm_L=10, mdm_version=1, forgetting_factor=0.98):
        """
        Optimalizovaný Adaptivní KF.
        Předpočítává konstantní matice MDM algoritmu pro LTI systémy.
        """
        self.kf = KalmanFilter(model)
        self.device = model.Q.device
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        self.lambda_rls = forgetting_factor
        
        # Dimenze
        self.nw = self.kf.F.shape[0]
        self.nv = self.kf.H.shape[0]
        self.n_params = (self.nw * (self.nw + 1)) // 2 + (self.nv * (self.nv + 1)) // 2
        
        # --- OPTIMALIZACE: PŘEDPOČÍTÁNÍ MATIC ---
        # Pro LTI systém jsou matice Awv a Upsilon konstantní.
        # Nemusíme je počítat v každém kroku.
        self._precompute_mdm_constants()
        
        # Inicializace RLS (startujeme na hodnotách z modelu)
        self.alpha_est = self._init_alpha_from_model(model)
        self.Sigma_RLS = np.eye(self.n_params) * 100.0

        # Buffery
        self.z_buffer = []
        self.u_buffer = []

    def _init_alpha_from_model(self, model):
        """Vytáhne počáteční parametry Q a R z modelu do vektoru alpha."""
        Q0 = model.Q.cpu().numpy()
        R0 = model.R.cpu().numpy()
        i_q, j_q = np.triu_indices(self.nw)
        alpha_q = Q0[i_q, j_q]
        i_r, j_r = np.triu_indices(self.nv)
        alpha_r = R0[i_r, j_r]
        return np.concatenate([alpha_q, alpha_r])

    def _precompute_mdm_constants(self):
        """
        Spočítá konstantní regresní matici H_rls_base.
        Tím ušetříme SVD rozklad a Kroneckerovy součiny v každém kroku.
        """
        print("Optimalizace: Předpočítávám MDM matice...")
        
        # Dummy data pro získání strukturálních matic
        dummy_z = np.zeros((self.mdm_L, self.nv))
        dummy_u = np.zeros((self.mdm_L, 1)) # Předpoklad u_dim=1
        
        F_np = self.kf.F.cpu().numpy()
        H_np = self.kf.H.cpu().numpy()
        G_np = np.zeros((self.nw, dummy_u.shape[1])) 
        E_np = np.eye(self.nw)
        D_np = np.eye(self.nv)
        nz_np = np.array([self.nv])
        
        # 1. Získáme Awv (regresor modelu)
        # Voláme MDM jednou naprázdno
        r_list, Awv_matrix = MDM_nullO_LTI(
            self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
            dummy_z, dummy_u, self.mdm_version
        )
        
        # 2. Získáme Upsilon (vazební matice momentů)
        w2b = [baseMatrix_fun(self.nw, 1)] 
        v2b = [baseMatrix_fun(self.nv, 1)] 
        Upsilon = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        
        # 3. Předpočítáme finální regresní matici pro RLS
        # H_rls = Ksi * (Awv (x) Awv) * Upsilon
        # Tato matice je pro LTI systém KONSTANTNÍ!
        nr = r_list[0].shape[0]
        Ksi = Ksi_fun(nr)
        
        # Uložíme si hotovou matici. V cyklu ji už jen použijeme.
        self.H_rls_cached = Ksi @ kron2_mat(Awv_matrix) @ Upsilon
        
        print("Hotovo.")

    def _reconstruct_qr(self, alpha):
        """Rychlá rekonstrukce Q a R."""
        q_len = (self.nw * (self.nw + 1)) // 2
        alpha_q = alpha[:q_len]
        alpha_r = alpha[q_len:]
        
        Q_est = np.zeros((self.nw, self.nw))
        i, j = np.triu_indices(self.nw)
        Q_est[i, j] = alpha_q; Q_est[j, i] = alpha_q
        
        R_est = np.zeros((self.nv, self.nv))
        i, j = np.triu_indices(self.nv)
        R_est[i, j] = alpha_r; R_est[j, i] = alpha_r
        return Q_est, R_est

    def _project_to_psd(self, M, min_val=1e-6):
        """Rychlá projekce na PSD."""
        M = (M + M.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(M)
        if np.any(eigvals < min_val):
            eigvals = np.maximum(eigvals, min_val)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T
        return M

    def step_adaptive(self, y_t, u_t=None):
        # 1. Update Bufferu (Standardní Python list operace jsou rychlé)
        y_np = y_t.cpu().numpy().squeeze()
        if y_np.ndim == 0: y_np = np.expand_dims(y_np, axis=0)
        
        if u_t is not None:
            u_np = u_t.cpu().numpy().squeeze()
            if u_np.ndim == 0: u_np = np.expand_dims(u_np, axis=0)
        else: u_np = np.zeros(1)

        self.z_buffer.append(y_np)
        self.u_buffer.append(u_np)
        
        if len(self.z_buffer) > self.mdm_L:
            self.z_buffer.pop(0); self.u_buffer.pop(0)

        # 2. Adaptace (Jen pokud je buffer plný)
        if len(self.z_buffer) == self.mdm_L:
            try:
                # Zde musíme volat MDM jen pro výpočet rezidua 'r'.
                # Awv už máme zapracované v H_rls_cached.
                # Optimalizace: Volat MDM funkci je stále trochu drahé kvůli SVD uvnitř.
                # Ale bez přepsání MDM_functions to lépe nejde.
                # Ušetříme ale masivně na výpočtu H_rls.
                
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                # Parametry (jen pro volání funkce)
                F_np = self.kf.F.cpu().numpy(); H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1])); E_np = np.eye(self.nw)
                D_np = np.eye(self.nv); nz_np = np.array([self.nv])

                # Získání rezidua r_k (Data dependent part)
                r_list, _ = MDM_nullO_LTI(self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, z_window, u_window, self.mdm_version)
                r_k = r_list[0]
                
                # RLS UPDATE (OPTIMALIZOVANÝ)
                nr = r_k.shape[0]
                Ksi = Ksi_fun(nr)
                y_rls = Ksi @ kron2_vec(r_k) # Pozorování
                
                H_rls = self.H_rls_cached # Použijeme předpočítanou matici!
                
                # RLS rovnice (Stable Inverse Update)
                # K = P * H.T * inv(H * P * H.T + lambda * I)
                S = H_rls @ self.Sigma_RLS @ H_rls.T + self.lambda_rls * np.eye(H_rls.shape[0])
                K_gain = self.Sigma_RLS @ H_rls.T @ np.linalg.pinv(S)
                
                error = y_rls - H_rls @ self.alpha_est
                self.alpha_est = self.alpha_est + K_gain @ error
                
                # Update P
                I_p = np.eye(self.n_params)
                self.Sigma_RLS = (1.0 / self.lambda_rls) * (I_p - K_gain @ H_rls) @ self.Sigma_RLS
                
                # Rekonstrukce a vložení
                Q_new, R_new = self._reconstruct_qr(self.alpha_est)
                Q_new = self._project_to_psd(Q_new, 1e-6)
                R_new = self._project_to_psd(R_new, 1e-3)
                
                self.kf.Q = torch.from_numpy(Q_new).float().to(self.device)
                self.kf.R = torch.from_numpy(R_new).float().to(self.device)

            except Exception:
                pass # Numerical instability fallback

        # 3. Filtrace
        x, P = self.kf.step(y_t)
        return x, P, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Simuluje online běh.
        OPRAVENO: Správné uložení počátečního stavu do historie.
        """
        seq_len = y_seq.shape[0]
        
        # 1. Kompletní reset RLS estimátoru
        self.__init__(self.kf.model, self.mdm_L, self.mdm_version)

        # 2. Určení počátečních podmínek
        if Ex0 is None: Ex0 = self.kf.model.Ex0
        if P0 is None: P0 = self.kf.model.P0

        # 3. Reset Kalmanova filtru
        self.kf.reset(Ex0, P0)
        
        self.z_buffer = []
        self.u_buffer = []

        # Inicializace historie
        x_hist = torch.zeros(seq_len, self.kf.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.kf.state_dim, self.kf.state_dim, device=self.device)
        Q_hist = []
        R_hist = []

        # --- OPRAVA: Uložení počátečního stavu do t=0 ---
        # Protože smyčka začíná od k=1 (čekáme na měření), musíme t=0 vyplnit ručně.
        # Uložíme tam apriori odhad (Ex0).
        x_hist[0] = Ex0.squeeze()
        P_hist[0] = P0

        # Smyčka běží od 1, protože KF update potřebuje nové měření y_t
        for k in tqdm(range(1, seq_len), desc="Processing sequence adaptively"):
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