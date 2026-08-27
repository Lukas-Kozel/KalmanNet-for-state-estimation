import torch
import numpy as np
from tqdm import tqdm

class AdaptiveKalmanFilter_mehra_offline_pretraining:
    def __init__(self, model, num_iterations=5):
        """
        Offline version of Mehra filter.
        Processes entire trajectory at once and repeats gain calculation `num_iterations` times.
        """
        self.device = model.Q.device
        self.model = model
        self.dtype = model.F.dtype
        self.F = model.F
        self.H = model.H
        self.nx = self.F.shape[0]
        self.nz = self.H.shape[0]
        self.num_iterations = num_iterations
        
        self.K = None

    def _filter_sequence(self, y_seq, Ex0):
        """Helper method: re-filters data with current (fixed) gain K and returns innovations and states."""
        seq_len = y_seq.shape[0]
        inov_history = torch.zeros(self.nz, seq_len, device=self.device, dtype=self.dtype)
        x_filt_history = torch.zeros(seq_len, self.nx, device=self.device, dtype=self.dtype)
        
        xp = Ex0.clone().detach().reshape(self.nx, 1)
        
        for i in range(seq_len):
            z_i = y_seq[i].reshape(self.nz, 1)

            # Innovation computation
            inov_i = z_i - self.H @ xp
            inov_history[:, i] = inov_i.squeeze()

            # State update (K is constant for entire sequence)
            x_filtered = xp + self.K @ inov_i
            x_filt_history[i] = x_filtered.squeeze()

            # Prediction for next step
            xp = self.F @ x_filtered
            
        return x_filt_history, inov_history

    def estimate_K_from_data(self, y_seq, Ex0=None):
        """
        PHASE 1: Based on offline data (e.g., long training trajectory)
        iteratively finds optimal steady Kalman gain K.
        """
        if Ex0 is None: Ex0 = self.model.Ex0
        
        seq_len = y_seq.shape[0]
        N = seq_len
        self.K = 0.1 * torch.linalg.pinv(self.H)

        print(f"\n--- Offline Mehra: Start gain adaptation ---")
        print(f"Iteration 0 (Initial guess): K[0,0] = {self.K[0,0].item():.4f}")
        
        for m in tqdm(range(self.num_iterations), desc="Finding optimal K (Mehra)", leave=False):
            # 1. Check stability of current K
            A = self.F - self.F @ self.K @ self.H

            # 2. Filter entire sequence with current K to obtain innovation sequence
            x_filt_hist, inov = self._filter_sequence(y_seq, Ex0)

            # 3. Estimate Py0 (Innovation covariance)
            Py0 = (inov @ inov.T) / N

            # 4. Construct Py and G
            Py_list = []
            G_list = []
            
            for k in range(1, self.nx + 10):
                inov_shift = inov[:, k:]
                inov_orig = inov[:, :-k]
                
                Py_k = (inov_shift @ inov_orig.T) / (N - k)
                Py_list.append(Py_k)
                
                A_pow = torch.matrix_power(A, k - 1)
                G_k = self.H @ A_pow @ self.F
                G_list.append(G_k)
                
            Py = torch.cat(Py_list, dim=0)
            G = torch.cat(G_list, dim=0)

            # 5. Estimate PHT
            PHT = self.K @ Py0 + torch.linalg.pinv(G) @ Py

            # 6. Estimate new K
            K_est = PHT @ torch.linalg.pinv(Py0)
            
            # Check stability of new K
            A_next = self.F - self.F @ K_est @ self.H
            eigvals = torch.linalg.eigvals(A_next)
            max_eig = torch.max(torch.abs(eigvals))
            if max_eig >= 1.0:
                print(f"Warning: In iteration {m} filter became unstable (max_eig={max_eig.item()}). Ending iterations.")
                break

            # Update gain for next iteration
            self.K = K_est

        print(f"--- Offline Mehra: Adaptation completed ---")
        return self.K.clone().detach()

    def process_sequence_with_K(self, y_seq, K_fixed, Ex0=None):
        """
        PHASE 2: Performs inference on test trajectory using fixed,
        pre-trained gain K.
        """
        if Ex0 is None: Ex0 = self.model.Ex0
        self.K = K_fixed
        
        seq_len = y_seq.shape[0]
        final_x_filt, final_inov = self._filter_sequence(y_seq, Ex0)

        # Mehra does not directly estimate matrix P (as standard KF does from Q and R).
        # For output dictionary compatibility, we return zero matrices.
        P_filtered_history = torch.zeros(seq_len, self.nx, self.nx, device=self.device, dtype=self.dtype)
        K_history = self.K.unsqueeze(0).repeat(seq_len, 1, 1) 
        
        return {
            'x_filtered': final_x_filt,
            'P_filtered': P_filtered_history,
            'Kalman_gain': K_history,
            'innovation': final_inov.T
        }

    def process_sequence_adaptively(self, y_seq, Ex0=None):
        """
        Original behavior (adaptation and filtering on same data at once).
        Kept for backward compatibility.
        """
        K_est = self.estimate_K_from_data(y_seq, Ex0)
        return self.process_sequence_with_K(y_seq, K_est, Ex0)
