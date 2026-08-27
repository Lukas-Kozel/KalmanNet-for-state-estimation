import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, pinv, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

from tqdm import tqdm

class KalmanFilter:
    """
    Kalman filter for time-invariant system with linear dynamics.
    (Original implementation unchanged)
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

        # Internal state for online use
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
        x_filtered, P_filtered, K, _ = self.update_step(self.x_predict_current, y_t, self.P_predict_current)
        x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered)
        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next
        return x_filtered, P_filtered, K

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

class AdaptiveKalmanFilter_online:
    def __init__(self, model, mdm_L=4, mdm_version=1, lambda_rls=0.995, init_sigma_rls_value=10.0, alpha_nom=None):
        """
        Adaptive KF (Semi-weighted Recursive version - Sw-Re).
        This class performs online identification of Q and R matrices.
        """
        self.kf = KalmanFilter(model)
        self.device = model.Q.device
        self.dtype = model.Q.dtype
        self.np_dtype = np.float32 if self.dtype == torch.float32 else np.float64
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        self.lambda_rls = lambda_rls
        self.init_sigma_val = init_sigma_rls_value
        self.alpha_nom_val = alpha_nom
        self.nw = self.kf.F.shape[0] # Dimension of state noise (W)
        self.nv = self.kf.H.shape[0] # Dimension of measurement noise (V)
        
        self.T = T = 1.0
        T2 = (T**2) / 2.0
        T3 = (T**3) / 3.0
        
        self.Q_template = np.array([
            [T3, T2, 0,  0],
            [T2, T,  0,  0],
            [0,  0,  T3, T2],
            [0,  0,  T2, T]
        ])
        self.vec_Q_template = self.Q_template[np.triu_indices(self.nw)][:, np.newaxis]
        self.n_params_Q_full = (self.nw * (self.nw + 1)) // 2 
        self.n_params_R_full = (self.nv * (self.nv + 1)) // 2
        self.n_params_Q_reduced = 1
        self.n_params_R_reduced = 2
        
        self.T_R_matrix = np.zeros((self.n_params_R_full, self.n_params_R_reduced), dtype=self.np_dtype)
        self.T_R_matrix[0, 0] = 1.0 
        self.T_R_matrix[7, 0] = 1.0 
        self.T_R_matrix[4, 1] = 1.0 
        self.T_R_matrix[9, 1] = 1.0 
        
        # 3. RLS INITIALIZATION
        self.n_params = self.n_params_Q_reduced + self.n_params_R_reduced

        Q0 = model.Q.cpu().numpy()
        self.q_init = Q0[0,0] / self.Q_template[0,0]
        
        R0 = model.R.cpu().numpy()
        r1_init = R0[0,0]
        r2_init = R0[1,1]
        
        self.alpha_est = np.array([self.q_init, r1_init, r2_init], dtype=self.np_dtype)
        self.alpha_nom = np.array(self.alpha_nom_val, dtype=self.np_dtype)


        # init_sigma_val is RELATIVE uncertainty (e.g. 0.1 = 10%)
        # alpha_nom are nominal values for Q and R which determine scale for sigma_q and sigma_r
        # squared because these are values estimated by RLS, units are Q and R
        # not units of their parameters (e.g. q vs q^2)
        sigma_q  = (self.alpha_nom[0] * self.init_sigma_val)**2 + 1e-12
        sigma_r1 = (self.alpha_nom[1] * self.init_sigma_val)**2 + 1e-12
        sigma_r2 = (self.alpha_nom[2] * self.init_sigma_val)**2 + 1e-12
        self.Sigma_RLS = np.diag([sigma_q, sigma_r1, sigma_r2])
        # Buffers
        self.z_buffer = []
        self.u_buffer = []
        self.Upsilon_2 = None

    def _get_upsilon(self):
        """
        Helper function to get Upsilon matrix.
        This matrix is purely structural (depends only on dimensions) and serves
        for manipulation with Kronecker products within MDM theory.
        Computed only once.
        """
        if self.Upsilon_2 is None:
            w2b = [baseMatrix_fun(self.nw, 1)] 
            v2b = [baseMatrix_fun(self.nv, 1)] 
            self.Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        return self.Upsilon_2


    def step_adaptive(self, y_t, u_t=None):
        """
        One step of adaptive filter with UNWEIGHTED RLS.
        """
        y_np = y_t.cpu().numpy().squeeze()
        if y_np.ndim == 0: y_np = np.expand_dims(y_np, axis=0)
        
        if u_t is not None:
            u_np = u_t.cpu().numpy().squeeze()
            if u_np.ndim == 0: u_np = np.expand_dims(u_np, axis=0)
        else:
            u_np = np.zeros(1)

        self.z_buffer.append(y_np)
        self.u_buffer.append(u_np)
        
        # Maintaining fixed window length L (FIFO queue)
        if len(self.z_buffer) > self.mdm_L:
            self.z_buffer.pop(0)
            self.u_buffer.pop(0)

        # --- 2. Q and R Identification (RLS) ---
        # Runs only if buffer is full (enough data for MDM)
        if len(self.z_buffer) == self.mdm_L:
            try:
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                F_np = self.kf.F.cpu().numpy()
                H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1]), dtype=self.np_dtype) # matrix for input u 
                E_np = np.eye(self.nw, dtype=self.np_dtype)
                D_np = np.eye(self.nv, dtype=self.np_dtype)
                nz_np = np.array([self.nv])

                # MDM: Residual and regressor calculation
                # r_k: Residual vector carrying information about noise in the data
                # Awv_matrix: Auxiliary matrix describing system dynamics in the window
                r_list, Awv_matrix = MDM_nullO_LTI(
                    self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
                    z_window, u_window, self.mdm_version
                )
                r_k = r_list[0]
                
                nr = r_k.shape[0]
                Ksi = Ksi_fun(nr) # Unification matrix (selects unique elements from symmetric matrix)
                
                # Observation for RLS (y_rls):
                # Derived from the outer product of residual (r_k * r_k^T), which is "instantaneous covariance".
                # kron2_vec converts it to vector and Ksi selects unique elements.
                y_rls = Ksi @ kron2_vec(r_k)
                
                Upsilon = self._get_upsilon()
                H_full = Ksi @ kron2_mat(Awv_matrix) @ Upsilon

                # Split into Q part and R part
                H_Q_part_full = H_full[:, :self.n_params_Q_full] 
                H_R_part_full = H_full[:, self.n_params_Q_full:]
                
                # Transform Q (10 columns -> 1 column)
                h_q_new = H_Q_part_full @ self.vec_Q_template

                # Transform R (10 columns -> 2 columns)
                # Result shape [N, 2]. First column is sensitivity to r1, second to r2.
                H_R_new = H_R_part_full @ self.T_R_matrix

                H_rls = np.hstack([h_q_new, H_R_new])
                
                # Flatten y_rls vector (safety against matrix errors)
                y_rls_flat = y_rls.flatten()
                
                # Weight calculation based on nominal values
                y_expected_nom = np.abs(H_rls @ self.alpha_nom).flatten()
                weights = y_expected_nom + 1e-12
                
                # Equation normalization (division by weights)
                y_rls_norm = y_rls_flat / weights
                H_rls_norm = H_rls / weights[:, np.newaxis]
                
                # RLS update on normalized data
                dim_obs_rls = H_rls_norm.shape[0]
                Omega_norm = np.eye(dim_obs_rls, dtype=self.np_dtype)

                S = H_rls_norm @ self.Sigma_RLS @ H_rls_norm.T + Omega_norm
                
                K_gain = self.Sigma_RLS @ H_rls_norm.T @ np.linalg.pinv(S)
                
                error = y_rls_norm - H_rls_norm @ self.alpha_est
                self.alpha_est = self.alpha_est + K_gain @ error
                
                I_p = np.eye(self.n_params, dtype=self.np_dtype)
                self.Sigma_RLS = (I_p - K_gain @ H_rls_norm) @ self.Sigma_RLS / self.lambda_rls              
                
                q_est = self.alpha_est[0]
                Q_new = q_est * self.Q_template
                
                # 2. R (parameters r1, r2)
                # r1_est = max(self.alpha_est[1], 1e-10) # Handle negative values
                # r2_est = max(self.alpha_est[2], 1e-10)
                r1_est = self.alpha_est[1]
                r2_est = self.alpha_est[2]
                # Build diagonal matrix diag([r1, r2, r1, r2])
                R_new = np.diag([r1_est, r2_est, r1_est, r2_est])
                
                # Update KF
                self.kf.Q = torch.from_numpy(Q_new).to(self.device).to(self.dtype)
                self.kf.R = torch.from_numpy(R_new).to(self.device).to(self.dtype)
                
            except Exception as e:
                print(f"Error in RLS update: {e}")
                pass

        # standard Kalman filter step with current Q and R
        x_filt, P_filt, K = self.kf.step(y_t)
        return x_filt, P_filt, K, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Helper method for simulating run on a complete data sequence.
        """
        seq_len = y_seq.shape[0]
        
        # Complete reset
        self.__init__(
            self.kf.model, 
            self.mdm_L, 
            self.mdm_version, 
            self.lambda_rls,
            init_sigma_rls_value=self.init_sigma_val,
            alpha_nom=self.alpha_nom_val
        )

        if Ex0 is None: Ex0 = self.kf.model.Ex0
        if P0 is None: P0 = self.kf.model.P0

        # Reset internal KF
        self.kf.reset(Ex0, P0)
        
        self.z_buffer = []
        self.u_buffer = []

        x_hist = torch.zeros(seq_len, self.kf.state_dim
