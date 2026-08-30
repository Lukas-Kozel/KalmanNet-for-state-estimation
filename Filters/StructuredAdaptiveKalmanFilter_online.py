import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

class KalmanFilter:
    """
    Kalman filter for a t-invariant system with linear dynamics.
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

class StructuredAdaptiveKalmanFilter:
    def __init__(self, model, mdm_L=6, mdm_version=1, forgetting_factor=0.995, 
                 estimate_q_scale=True, diagonal_R=True):
        """
        Adaptive KF with STRUCTURED estimation of Q and R.
        
        This class implements expert advice: instead of searching for full Q and R matrices,
        we search only for their key parameters (scalar scaling for Q, diagonal for R).
        
        Args:
            model: Dynamic system object (contains F, H, Q_base, R_base).
            estimate_q_scale: If True, searches for a single scalar 'q' that scales model.Q.
            diagonal_R: If True, searches only for diagonal elements of R (others are 0).
        """
        self.kf = KalmanFilter(model) # Assumes the existence of the Filters.KalmanFilter class
        self.device = model.Q.device
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        self.lambda_rls = forgetting_factor
        
        # --- 1. Problem dimensions definition ---
        self.nw = self.kf.F.shape[0] # State noise dimension
        self.nv = self.kf.H.shape[0] # Measurement noise dimension
        
        # Full number of unique elements in symmetric matrices (upper triangle)
        # This is what the MDM function standardly "produces".
        self.n_params_full_Q = (self.nw * (self.nw + 1)) // 2
        self.n_params_full_R = (self.nv * (self.nv + 1)) // 2
        
        # --- 2. Reduced (search) space definition ---
        self.estimate_q_scale = estimate_q_scale
        self.diagonal_R = diagonal_R
        
        # Q: We search for either 1 scalar or a full matrix
        if self.estimate_q_scale:
            self.n_params_red_Q = 1 
            # Save the "base" matrix Q from the model (e.g., kinematic model).
            # We will search for coefficient 'q' such that: Q_new = q * Q_base
            self.Q_base_np = model.Q.cpu().numpy() 
        else:
            self.n_params_red_Q = self.n_params_full_Q
            
        # R: We search for either just the diagonal (nv elements) or a full matrix
        if self.diagonal_R:
            self.n_params_red_R = self.nv 
        else:
            self.n_params_red_R = self.n_params_full_R
            
        # Total number of estimated parameters (e.g., 1 + 4 = 5)
        self.n_params_red = self.n_params_red_Q + self.n_params_red_R

        # --- 3. Mapping Matrix M Construction ---
        # This is the key trick. Relation: alpha_FULL = M * alpha_RED
        self.M = self._construct_mapping_matrix()
        
        # --- 4. RLS Algorithm Initialization ---
        # We need to initialize the alpha_est vector (estimated parameters).
        alpha_red_init = []
        
        # Init for Q
        if self.estimate_q_scale:
            alpha_red_init.append(1.0) # Start at value 1.0 (meaning Q = 1.0 * Model.Q)
        else:
            # If we were to search for a full matrix, take elements from the model
            Q0 = model.Q.cpu().numpy()
            i, j = np.triu_indices(self.nw)
            alpha_red_init.extend(Q0[i, j])
            
        # Init for R
        R0 = model.R.cpu().numpy()
        if self.diagonal_R:
            alpha_red_init.extend(np.diag(R0)) # Take only the diagonal
        else:
            i, j = np.triu_indices(self.nv)
            alpha_red_init.extend(R0[i, j]) # Take the whole upper triangle
            
        self.alpha_est = np.array(alpha_red_init)
        
        # RLS covariance matrix (parameter estimation uncertainty)
        # Now it has a dimension of only [5x5] instead of [20x20] -> much faster convergence!
        self.Sigma_RLS = np.eye(self.n_params_red) * 100.0

        # Buffers for MDM window
        self.z_buffer = []
        self.u_buffer = []
        self.Upsilon_2 = None

    def _construct_mapping_matrix(self):
        """
        Creates matrix M that transforms reduced parameters to a full vector.
        This matrix mathematically "enforces" the structure of Q and R.
        """
        # --- M part for Q ---
        if self.estimate_q_scale:
            # If we search only for scalar q:
            # Full Q vector (vech Q) = q * vech(Q_base)
            # So the column of matrix M corresponding to Q equals vech(Q_base).
            i, j = np.triu_indices(self.nw)
            vech_Q_base = self.Q_base_np[i, j]
            M_Q = vech_Q_base.reshape(-1, 1) # Column vector [10, 1]
        else:
            M_Q = np.eye(self.n_params_full_Q) # Identity (no reduction)

        # --- M part for R ---
        if self.diagonal_R:
            # If we search only for the diagonal:
            # M_R will have dimensions [n_full, n_diag].
            # Each column corresponds to one diagonal element and has a "1" in the row,
            # which in the full vector corresponds to the position of this diagonal element.
            M_R = np.zeros((self.n_params_full_R, self.nv))
            
            full_idx = 0 # Index in the full vector (vech R)
            diag_idx = 0 # Index in our reduced vector (diagonal)
            
            # Iterate through the upper triangle (same as np.triu_indices)
            for row in range(self.nv):
                for col in range(row, self.nv):
                    if row == col:
                        # This is a diagonal element! 
                        M_R[full_idx, diag_idx] = 1.0
                        diag_idx += 1
                    # Otherwise it's an off-diagonal element -> leave as 0 (enforcing zero)
                    full_idx += 1
        else:
            M_R = np.eye(self.n_params_full_R)

        # --- Assembly of the full M ---
        # M = [ M_Q   0  ]
        #     [  0   M_R ]
        dim_full_Q = M_Q.shape[0]
        dim_red_Q = M_Q.shape[1]
        dim_full_R = M_R.shape[0]
        dim_red_R = M_R.shape[1]
        
        M = np.zeros((dim_full_Q + dim_full_R, dim_red_Q + dim_red_R))
        M[:dim_full_Q, :dim_red_Q] = M_Q
        M[dim_full_Q:, dim_red_Q:] = M_R
        
        return M

    def _get_upsilon(self):
        # Lazy loading for Upsilon matrix (constant for LTI)
        if self.Upsilon_2 is None:
            w2b = [baseMatrix_fun(self.nw, 1)] 
            v2b = [baseMatrix_fun(self.nv, 1)] 
            self.Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        return self.Upsilon_2

    def _reconstruct_qr(self, alpha_red):
        """
        Converts reduced parameters back to Q and R matrices for the Kalman filter.
        """
        # 1. Split vector into Q part and R part
        if self.estimate_q_scale:
            q_scale = alpha_red[0] # The first element is our scalar
            
            # Q Reconstruction: Q_new = scale * Q_base
            Q_est = q_scale * self.Q_base_np
            
            # The rest of the vector belongs to R
            alpha_R = alpha_red[1:]
        else:
            # (Omitted for brevity - full Q logic)
            alpha_R = alpha_red[self.n_params_full_Q:]
            Q_est = np.zeros((self.nw, self.nw)) 

        # 2. R Reconstruction
        if self.diagonal_R:
            # alpha_R directly contains diagonal elements
            # np.diag creates a matrix with these elements on the diagonal and zeros elsewhere
            R_est = np.diag(alpha_R)
        else:
            # Full R logic...
            R_est = np.zeros((self.nv, self.nv))
            i, j = np.triu_indices(self.nv)
            R_est[i, j] = alpha_R
            R_est[j, i] = alpha_R

        return Q_est, R_est

    def step_adaptive(self, y_t, u_t=None):
        """
        One step of online adaptation. The magic of structured estimation happens here.
        """
        # --- 1. Standard buffer update (sliding window) ---
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

        # --- 2. Adaptive algorithm (only if we have a full window) ---
        if len(self.z_buffer) == self.mdm_L:
            try:
                # Prepare data for MDM
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                F_np = self.kf.F.cpu().numpy()
                H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1])) 
                E_np = np.eye(self.nw)
                D_np = np.eye(self.nv)
                nz_np = np.array([self.nv])

                # >>> MDM COMPUTATION <<<
                # Extract "raw" statistics from data. This is independent of Q/R structure.
                # r_list[0] = vector of residuals (data)
                # Awv_matrix = matrix describing the relationship of noises and residuals (model)
                r_list, Awv_matrix = MDM_nullO_LTI(
                    self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
                    z_window, u_window, self.mdm_version
                )
                r_k = r_list[0]
                
                # >>> RLS PREPARATION <<<
                nr = r_k.shape[0]
                Ksi = Ksi_fun(nr) # Selector of unique covariance elements
                
                # Observed vector y_rls (left side of the equation y = H*x)
                # This is the empirical covariance of residuals from the current window
                y_rls = Ksi @ kron2_vec(r_k)
                
                # Upsilon matrix (precomputed)
                Upsilon = self._get_upsilon()
                
                # >>> STRUCTURE APPLICATION (The most important part!) <<<
                
                # 1. H_full: Regression matrix for FULL problem (searching all 20 elements)
                # Corresponds to the equation in the paper: A_2 = Ksi * (Awv (x) Awv) * Upsilon
                H_full = Ksi @ kron2_mat(Awv_matrix) @ Upsilon
                
                # 2. H_red: Projection into STRUCTURED space
                # Apply our mapping matrix M.
                # Equation: y = H_full * alpha_full
                # Substitution: alpha_full = M * alpha_red
                # Result: y = (H_full * M) * alpha_red
                # With this step we tell the RLS algorithm: "Search only for parameters alpha_red, 
                # and we already know how to assemble the full matrices from them."
                H_red = H_full @ self.M 

                # >>> RLS UPDATE (Recursive Least Squares) <<<
                # We now work with significantly smaller matrices (e.g. 5x5 instead of 20x20)
                
                # Innovation covariance S
                S = H_red @ self.Sigma_RLS @ H_red.T + self.lambda_rls * np.eye(H_red.shape[0])
                
                # Kalman Gain for parameters
                # Use pinv for numerical stability
                K_gain = self.Sigma_RLS @ H_red.T @ np.linalg.pinv(S)
                
                # Parameter prediction error
                pred_error = y_rls - H_red @ self.alpha_est
                
                # Parameter estimate update
                self.alpha_est = self.alpha_est + K_gain @ pred_error
                
                # Estimate covariance update (Sigma)
                I_p = np.eye(self.n_params_red)
                self.Sigma_RLS = (1.0 / self.lambda_rls) * (I_p - K_gain @ H_red) @ self.Sigma_RLS
                
                # >>> RECONSTRUCTION AND REGULARIZATION <<<
                Q_new, R_new = self._reconstruct_qr(self.alpha_est)
                
                # Protection against physical nonsense (negative variance)
                if self.estimate_q_scale:
                     # Scalar q must be positive. If RLS drifts into negative, reset to a small positive value.
                     if self.alpha_est[0] < 1e-9: 
                         self.alpha_est[0] = 1e-9
                         # We must reconstruct again to apply this
                         Q_new, R_new = self._reconstruct_qr(self.alpha_est)
                
                if self.diagonal_R:
                    # Diagonal elements of R must be positive.
                    # Use a threshold (e.g. 1e-3), which acts as the "regularization" discussed earlier.
                    # Prevents ANEES explosion.
                    np.fill_diagonal(R_new, np.maximum(np.diag(R_new), 1e-2))
                
                # Insert into the filter
                self.kf.Q = torch.from_numpy(Q_new).float().to(self.device)
                self.kf.R = torch.from_numpy(R_new).float().to(self.device)

            except Exception as e:
                # Fallback: If MDM fails in this step (e.g., numerical issues), 
                # do not change Q and R and continue with previous values.
                # print(f"MDM Fail: {e}")
                pass

        # --- 3. Kalman Filter Step ---
        x, P = self.kf.step(y_t)
        
        return x, P, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Simulates an online run over the entire sequence.
        """
        seq_len = y_seq.shape[0]
        
        # 1. Reset RLS (calls __init__ of this class)
        # Pass all structure parameters to preserve them
        self.__init__(self.kf.model, self.mdm_L, self.mdm_version, self.lambda_rls, 
                      self.estimate_q_scale, self.diagonal_R)
        
        # 2. Filter initial conditions setup
        if Ex0 is None: Ex0 = self.kf.model.Ex0
        if P0 is None: P0 = self.kf.model.P0
        self.kf.reset(Ex0, P0)
        
        # Buffers reset
        self.z_buffer = []
        self.u_buffer = []

        # History logging
        x_hist = torch.zeros(seq_len, self.kf.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.kf.state_dim, self.kf.state_dim, device=self.device)
        Q_hist = []
        R_hist = []

        # Main loop
        for k in range(seq_len):
            y_t = y_seq[k]
            u_t = u_seq[k] if u_seq is not None else None
            
            x, P, Q_curr, R_curr = self.step_adaptive(y_t, u_t)
            
            x_hist[k] = x.squeeze()
            P_hist[k] = P
            Q_hist.append(Q_curr.clone().detach())
            R_hist.append(R_curr.clone().detach())
            
        return {'x_filtered': x_hist, 'P_filtered': P_hist}, Q_hist, R_hist