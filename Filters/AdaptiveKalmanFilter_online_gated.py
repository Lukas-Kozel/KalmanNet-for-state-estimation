import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, pinv, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

from tqdm import tqdm


class KalmanFilterGated:
    def __init__(self, model):
        self.device = model.Q.device
        self.dtype = model.F.dtype
        self.model = model
        self.F, self.H, self.Q, self.R = model.F, model.H, model.Q, model.R
        self.state_dim = self.F.shape[0]
        self.obs_dim = self.H.shape[0]
        self.I = torch.eye(self.state_dim, device=self.device, dtype=self.dtype)
        self.reset(model.Ex0, model.P0)

    def reset(self, Ex0, P0):
        # Ex0 je nyní dynamicky předáváno na začátku každé sekvence
        self.x_predict_current = Ex0.clone().detach().reshape(self.state_dim, 1)
        self.P_predict_current = P0.clone().detach()

    def step(self, y_t, gate_threshold=9.21): # 9.21 = Chi-Sq 99% limit pro 2 DOF
        rejected = False
        if y_t is not None:
            y_t_vec = y_t.reshape(self.obs_dim, 1)
            inov = y_t_vec - self.H @ self.x_predict_current
            S = self.H @ self.P_predict_current @ self.H.T + self.R
            S_inv = torch.linalg.inv(S)
            
            # --- PARCIÁLNÍ GATING (Pouze na pozici X a Y) ---
            # X je index 0, Y je index 2
            inov_gps = torch.cat([inov[0:1], inov[2:3]], dim=0) # [2, 1]
            
            S_gps = torch.zeros((2, 2), device=self.device, dtype=self.dtype)
            S_gps[0, 0] = S[0, 0]; S_gps[0, 1] = S[0, 2]
            S_gps[1, 0] = S[2, 0]; S_gps[1, 1] = S[2, 2]
            
            S_gps_inv = torch.linalg.inv(S_gps)
            mah_sq_gps = (inov_gps.T @ S_gps_inv @ inov_gps).item()
            
            if mah_sq_gps > gate_threshold:
                rejected = True

        if y_t is None or rejected:
            # PREDICT ONLY (Měření zahozeno nebo chybí)
            x_filtered = self.x_predict_current
            P_filtered = self.P_predict_current
            K = torch.zeros((self.state_dim, self.obs_dim), device=self.device, dtype=self.dtype)
        else:
            # STANDARD UPDATE
            K = self.P_predict_current @ self.H.T @ S_inv
            x_filtered = self.x_predict_current + K @ inov
            I_KH = self.I - K @ self.H
            P_filtered = I_KH @ self.P_predict_current @ I_KH.T + K @ self.R @ K.T

        # PREDICT PRO DALŠÍ KROK
        x_predict_next = self.F @ x_filtered
        P_predict_next = self.F @ P_filtered @ self.F.T + self.Q
        
        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next

        return x_filtered, P_filtered, K, rejected

    def process_sequence(self, y_seq, Ex0, P0, gate_threshold=9.21):
        seq_len = y_seq.shape[0]
        self.reset(Ex0, P0)
        x_hist = torch.zeros(seq_len, self.state_dim, device=self.device, dtype=self.dtype)
        P_hist = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device, dtype=self.dtype)
        
        for k in tqdm(range(seq_len), desc="Standard KF Gated", leave=False):
            x_f, P_f, _, _ = self.step(y_seq[k], gate_threshold)
            x_hist[k] = x_f.squeeze()
            P_hist[k] = P_f
            
        return {'x_filtered': x_hist, 'P_filtered': P_hist}


class AdaptiveKalmanFilter_online_Gated:
    def __init__(self, model, mdm_L=6, mdm_version=1, lambda_rls=0.99, init_sigma_rls_value=0.1, alpha_nom=None):
        self.kf = KalmanFilterGated(model)
        self.device, self.dtype = model.Q.device, model.Q.dtype
        self.np_dtype = np.float32 if self.dtype == torch.float32 else np.float64
        self.mdm_L, self.mdm_version, self.lambda_rls = mdm_L, mdm_version, lambda_rls
        self.nw, self.nv = self.kf.F.shape[0], self.kf.H.shape[0]
        
        self.F_np = self.kf.F.cpu().numpy().astype(self.np_dtype)
        self.H_np = self.kf.H.cpu().numpy().astype(self.np_dtype)
        self.E_np = np.eye(self.nw, dtype=self.np_dtype)
        self.D_np = np.eye(self.nv, dtype=self.np_dtype)
        self.nz_np = np.array([self.nv])
        
        self.Ksi, self.H_rls_norm, self.weights, self.HTH, self.HT = None, None, None, None, None
        
        T = 1.0
        self.Q_template = np.array([[T**3/3, T**2/2, 0, 0], [T**2/2, T, 0, 0], [0, 0, T**3/3, T**2/2], [0, 0, T**2/2, T]], dtype=self.np_dtype)
        self.vec_Q_template = self.Q_template[np.triu_indices(self.nw)][:, np.newaxis]
        self.n_params_Q_full, self.n_params_Q_reduced = (self.nw * (self.nw + 1)) // 2, 1
        self.n_params_R_full, self.n_params_R_reduced = (self.nv * (self.nv + 1)) // 2, 2
        
        self.T_R_matrix = np.zeros((self.n_params_R_full, self.n_params_R_reduced), dtype=self.np_dtype)
        self.T_R_matrix[0, 0] = self.T_R_matrix[7, 0] = self.T_R_matrix[4, 1] = self.T_R_matrix[9, 1] = 1.0 
        
        Q0, R0 = model.Q.cpu().numpy(), model.R.cpu().numpy()
        self.alpha_est = np.array([Q0[0,0] / self.Q_template[0,0], R0[0,0], R0[1,1]], dtype=self.np_dtype)
        self.alpha_nom = np.array(alpha_nom, dtype=self.np_dtype)

        sigma_q = (self.alpha_nom[0] * init_sigma_rls_value)**2 + 1e-12
        sigma_r1 = (self.alpha_nom[1] * init_sigma_rls_value)**2 + 1e-12
        sigma_r2 = (self.alpha_nom[2] * init_sigma_rls_value)**2 + 1e-12
        self.Sigma_RLS = np.diag([sigma_q, sigma_r1, sigma_r2]).astype(self.np_dtype)
        
        self.z_buffer, self.u_buffer = [], []
        self.Upsilon_2 = None

    def _get_upsilon(self):
        if self.Upsilon_2 is None:
            w2b, v2b = [baseMatrix_fun(self.nw, 1)], [baseMatrix_fun(self.nv, 1)]
            self.Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        return self.Upsilon_2

    def step_adaptive(self, y_t, gate_threshold=9.21):
        x_filt, P_filt, K, rejected = self.kf.step(y_t, gate_threshold)
        
        if rejected:
            # Měření je OUTLIER. Vymažeme buffer, neadaptujeme se.
            self.z_buffer = []
            self.u_buffer = []
            return x_filt, P_filt, K, self.kf.Q, self.kf.R

        y_np = y_t.cpu().numpy().squeeze()
        if y_np.ndim == 0: y_np = np.expand_dims(y_np, axis=0)
        u_np = np.zeros(1, dtype=self.np_dtype)

        self.z_buffer.append(y_np)
        self.u_buffer.append(u_np)
        
        if len(self.z_buffer) > self.mdm_L:
            self.z_buffer.pop(0)
            self.u_buffer.pop(0)

        if len(self.z_buffer) == self.mdm_L:
            try:
                z_window = np.array(self.z_buffer, dtype=self.np_dtype)
                u_window = np.array(self.u_buffer, dtype=self.np_dtype)
                G_np = np.zeros((self.nw, 1), dtype=self.np_dtype)

                r_list, Awv_matrix = MDM_nullO_LTI(self.mdm_L, self.F_np, G_np, self.E_np, self.nz_np, self.H_np, self.D_np, z_window, u_window, self.mdm_version)
                r_k = r_list[0]
                
                if self.H_rls_norm is None:
                    self.Ksi = Ksi_fun(r_k.shape[0])
                    H_full = self.Ksi @ kron2_mat(Awv_matrix) @ self._get_upsilon()
                    H_rls = np.hstack([H_full[:, :self.n_params_Q_full] @ self.vec_Q_template, H_full[:, self.n_params_Q_full:] @ self.T_R_matrix])
                    self.weights = np.abs(H_rls @ self.alpha_nom).flatten() + 1e-12
                    self.H_rls_norm = H_rls / self.weights[:, np.newaxis]
                    self.HT = self.H_rls_norm.T
                    self.HTH = self.HT @ self.H_rls_norm

                y_rls_norm = (self.Ksi @ kron2_vec(r_k)).flatten() / self.weights
                
                P_next_unscaled = np.linalg.inv(np.linalg.inv(self.Sigma_RLS) + self.HTH)
                self.Sigma_RLS = P_next_unscaled / self.lambda_rls
                self.alpha_est += (P_next_unscaled @ self.HT) @ (y_rls_norm - self.H_rls_norm @ self.alpha_est)
                
                Q_new = max(self.alpha_est[0], 1e-10) * self.Q_template
                r1, r2 = max(self.alpha_est[1], 1e-10), max(self.alpha_est[2], 1e-10)
                R_new = np.diag([r1, r2, r1, r2])
                
                self.kf.Q = torch.from_numpy(Q_new).to(self.device).to(self.dtype)
                self.kf.R = torch.from_numpy(R_new).to(self.device).to(self.dtype)
                
            except Exception as e:
                pass

        return x_filt, P_filt, K, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, Ex0, P0, gate_threshold=9.21):
        seq_len = y_seq.shape[0]
        self.kf.reset(Ex0, P0)
        self.z_buffer, self.u_buffer = [], []
        x_hist = torch.zeros(seq_len, self.nw, device=self.device, dtype=self.dtype)
        P_hist = torch.zeros(seq_len, self.nw, self.nw, device=self.device, dtype=self.dtype)
        Q_hist, R_hist = [], []

        for k in tqdm(range(seq_len), desc="AKF MDM Gated", leave=False):
            x, P, K, Q_curr, R_curr = self.step_adaptive(y_seq[k], gate_threshold)
            x_hist[k], P_hist[k] = x.squeeze(), P
            Q_hist.append(Q_curr.clone().detach())
            R_hist.append(R_curr.clone().detach())
            
        return {'x_filtered': x_hist, 'P_filtered': P_hist}, Q_hist, R_hist