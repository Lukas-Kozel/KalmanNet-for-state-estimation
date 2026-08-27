import torch
import numpy as np

from tqdm import tqdm

class AdaptiveKalmanFilter_mehra:
    def __init__(self, model, window_size=200):
        self.model = model
        self.dtype = model.F.dtype
        self.device = model.Q.device
        
        self.nx = self.model.F.shape[0]
        self.nz = self.model.H.shape[0]
        
        # Sliding window size setting
        self.window_size = window_size
        self.inov_buffer = []
        
        self.reset()

    def reset(self, Ex0=None, P0=None):
        """Clear buffers and reset internal state before a new trajectory."""
        if Ex0 is None: Ex0 = self.model.Ex0
        if P0 is None: P0 = self.model.P0
            
        self.x_predict = Ex0.clone().detach().reshape(self.nx, 1)
        self.inov_buffer = []
        
        # 1. Choose a stable K for the linear filter
        self.K = 0.1 * torch.linalg.pinv(self.model.H)
    
    
    def step(self, y_t):
        y_t = y_t.reshape(self.nz, 1)
        
        inov = y_t - self.model.H @ self.x_predict
        
        # Sliding window filling
        self.inov_buffer.append(inov)
            
        N_current = len(self.inov_buffer)
        
        # Matrix computation is performed after the window is filled
        if N_current >= self.window_size:
            inov_tensor = torch.cat(self.inov_buffer, dim=1)
            
            # Covariance estimate for l=0
            Py0 = (inov_tensor @ inov_tensor.T) / N_current
            
            Py_list = [Py0] 
            
            # Compute \hat{P}_l^y according to equation (22)
            for l in range(1, self.nx + 1):
                inov_shift = inov_tensor[:, l:]
                inov_orig = inov_tensor[:, :-l]
                Py_l = (inov_shift @ inov_orig.T) / (N_current - l)
                Py_list.append(Py_l)
                
            G_list = []
            V_list = [] 
            
            F = self.model.F
            H = self.model.H
            K = self.K
            
            # Build G according to equation (21) and matrix V according to (23)
            for l in range(1, self.nx + 1):
                # Matrix G: G_l = H * F^l
                G_l = H @ torch.matrix_power(F, l)
                G_list.append(G_l)
                
                # Matrix V (Equation 23): Each row is a sum
                V_l = Py_list[l].clone()
                for i in range(1, l + 1):
                    term = H @ torch.matrix_power(F, i) @ K @ Py_list[l - i]
                    V_l += term
                V_list.append(V_l)
                
            G = torch.cat(G_list, dim=0)
            V = torch.cat(V_list, dim=0)
            
            # Estimate of P^e * H^T according to equation (23)
            PHT = torch.linalg.pinv(G) @ V
            
            # Kalman gain estimate
            Py0_inv = torch.linalg.pinv(Py0 + 1e-6 * torch.eye(self.nz, device=self.device,dtype=self.dtype))
            K_est = PHT @ Py0_inv
            
            if not (torch.isnan(K_est).any() or torch.isinf(K_est).any()):
                A_test = F - F @ K_est @ H
                eigvals = torch.linalg.eigvals(A_test)
                max_eig = torch.max(torch.abs(eigvals))
                
                if max_eig < 1.0:
                    # Gain is stable
                    self.K = K_est
                else:
                    # Statistical deviation in this block caused instability.
                    # This estimate is discarded and continue with current stable K
                    pass
            
            # buffer reset to ensure LTI behavior
            self.inov_buffer = []
            
        # Filtering and prediction for next step
        x_filtered = self.x_predict + self.K @ inov
        self.x_predict = self.model.F @ x_filtered
        
        return x_filtered, self.K

    def process_sequence(self, y_seq, Ex0=None, P0=None):
        seq_len = y_seq.shape[0]
        
        self.reset(Ex0, P0)
        
        x_filtered_history = torch.zeros(seq_len, self.nx, device=self.device,dtype=self.dtype)
        P_filtered_history = torch.zeros(seq_len, self.nx, self.nx, device=self.device,dtype=self.dtype)
        kalman_gain_history = torch.zeros(seq_len, self.nx, self.nz, device=self.device,dtype=self.dtype)
        innovation_history = torch.zeros(seq_len, self.nz, device=self.device,dtype=self.dtype)
        
        for k in tqdm(range(seq_len), desc="Online filtering with adaptive K", leave=False):
            z_t = y_seq[k].reshape(self.nz, 1)
            inov_k = z_t - self.model.H @ self.x_predict
            
            x_filtered, K_current = self.step(y_seq[k])
            
            x_filtered_history[k] = x_filtered.squeeze()
            kalman_gain_history[k] = K_current
            innovation_history[k] = inov_k.squeeze()
            
        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'Kalman_gain': kalman_gain_history,
            'innovation': innovation_history
        }