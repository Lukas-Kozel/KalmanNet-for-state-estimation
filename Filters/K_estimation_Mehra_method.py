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
        
        # Nastavení velikosti klouzavého okna
        self.window_size = window_size
        self.inov_buffer = []
        
        self.reset()

    def reset(self, Ex0=None, P0=None):
        """Vymaže buffery a zresetuje vnitřní stav před novou trajektorií."""
        if Ex0 is None: Ex0 = self.model.Ex0
        if P0 is None: P0 = self.model.P0
            
        self.x_predict = Ex0.clone().detach().reshape(self.nx, 1)
        self.inov_buffer = []
        
        # 1. Zvolme stabilní K lineárního filtru
        self.K = 0.1 * torch.linalg.pinv(self.model.H)
    
    
    def step(self, y_t):
        y_t = y_t.reshape(self.nz, 1)
        
        inov = y_t - self.model.H @ self.x_predict
        
        # Plnění klouzavého okna
        self.inov_buffer.append(inov)
            
        N_current = len(self.inov_buffer)
        
        # Výpočet matic proběhne až po naplnění okna
        if N_current >= self.window_size:
            inov_tensor = torch.cat(self.inov_buffer, dim=1)
            
            # Odhad kovariancí pro l=0
            Py0 = (inov_tensor @ inov_tensor.T) / N_current
            
            Py_list = [Py0] 
            
            # Spočítáme \hat{P}_l^y podle rovnice (22)
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
            
            # Sestavení G podle rovnice (21) a matici V podle (23)
            for l in range(1, self.nx + 1):
                # Matice G: G_l = H * F^l
                G_l = H @ torch.matrix_power(F, l)
                G_list.append(G_l)
                
                # Matice V (Rovnice 23): Každý řádek je součet
                V_l = Py_list[l].clone()
                for i in range(1, l + 1):
                    term = H @ torch.matrix_power(F, i) @ K @ Py_list[l - i]
                    V_l += term
                V_list.append(V_l)
                
            G = torch.cat(G_list, dim=0)
            V = torch.cat(V_list, dim=0)
            
            # Odhad součinu P^e * H^T podle rovnice (23)
            PHT = torch.linalg.pinv(G) @ V
            
            # Odhad Kalmanova zisku
            Py0_inv = torch.linalg.pinv(Py0 + 1e-6 * torch.eye(self.nz, device=self.device,dtype=self.dtype))
            K_est = PHT @ Py0_inv
            
            if not (torch.isnan(K_est).any() or torch.isinf(K_est).any()):
                A_test = F - F @ K_est @ H
                eigvals = torch.linalg.eigvals(A_test)
                max_eig = torch.max(torch.abs(eigvals))
                
                if max_eig < 1.0:
                    # Zisk je stabilní
                    self.K = K_est
                else:
                    # Statistická odchylka v tomto bloku způsobila nestabilitu.
                    # Tento odhad je zahozen a pokračuje se s aktuálním stabilním K
                    pass
            
            # vynulování bufferu kvůli zajištění LTI chování
            self.inov_buffer = []
            
        # Filtrace a predikce na další krok
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
        
        for k in tqdm(range(seq_len), desc="Online filtrace s adaptivním K", leave=False):
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