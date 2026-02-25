import torch
import numpy as np
from tqdm import tqdm

class KalmanFilter:
    # Ponecháno zcela beze změn
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

class AdaptiveKalmanFilter_mehra_offline:
    def __init__(self, model, num_iterations=5):
        """
        Offline (Dávková) verze Mehrova filtru.
        Zpracuje celou trajektorii naráz a opakuje výpočet zisku `num_iterations` krát.
        """
        self.device = model.Q.device
        self.model = model
        
        self.F = model.F
        self.H = model.H
        self.nx = self.F.shape[0]
        self.nz = self.H.shape[0]
        self.num_iterations = num_iterations
        
        self.K = None

    def _filter_sequence(self, y_seq, Ex0):
        """Pomocná metoda: přefiltruje data s aktuálním (fixním) ziskem K a vrátí inovace a stavy."""
        seq_len = y_seq.shape[0]
        inov_history = torch.zeros(self.nz, seq_len, device=self.device)
        x_filt_history = torch.zeros(seq_len, self.nx, device=self.device)
        
        xp = Ex0.clone().detach().reshape(self.nx, 1)
        
        for i in range(seq_len):
            z_i = y_seq[i].reshape(self.nz, 1)
            
            # Výpočet inovace
            inov_i = z_i - self.H @ xp
            inov_history[:, i] = inov_i.squeeze()
            
            # Update stavu (K je konstantní pro celou sekvenci)
            x_filtered = xp + self.K @ inov_i
            x_filt_history[i] = x_filtered.squeeze()
            
            # Predikce pro další krok
            xp = self.F @ x_filtered
            
        return x_filt_history, inov_history

    def process_sequence(self, y_seq, Ex0=None, P0=None):
        """
        Hlavní dávková metoda, přesně podle MATLAB kódu.
        """
        if Ex0 is None: Ex0 = self.model.Ex0
        
        seq_len = y_seq.shape[0]
        N = seq_len
        self.K = 0.7 * torch.eye(self.nz, device=self.device) @ torch.linalg.pinv(self.H)
        
        print(f"\n--- Offline Mehra: Start adaptace ---")
        print(f"Iterace 0 (Nástřel): K[0,0] = {self.K[0,0].item():.4f}")
        # Opakujeme proces filtrace a odhadu M-krát
        for m in range(self.num_iterations):
            # 1. Kontrola stability aktuálního K
            A = self.F - self.F @ self.K @ self.H
            eigvals = torch.linalg.eigvals(A)
            max_eig = torch.max(torch.abs(eigvals))
            
            if max_eig >= 1.0:
                print(f"Varování: V iteraci {m} se filtr stal nestabilním (max_eig={max_eig.item()}). Končím iterace.")
                break # Pokud je K nestabilní, dál už to nemá smysl iterovat
                
            # 2. Filtrace celé sekvence s aktuálním K pro získání inovační posloupnosti
            x_filt_hist, inov = self._filter_sequence(y_seq, Ex0)
            
            # 3. Odhad Py0 (Kovariance inovací)
            Py0 = (inov @ inov.T) / N
            
            # 4. Sestavení Py a G
            Py_list = []
            G_list = []
            
            for k in range(1, self.nx + 1):
                # Odhad autokovariancí (posunuté vs. originál)
                inov_shift = inov[:, k:]
                inov_orig = inov[:, :-k]
                
                Py_k = (inov_shift @ inov_orig.T) / (N - k)
                Py_list.append(Py_k)
                
                # Sestavení G z matice A
                A_pow = torch.matrix_power(A, k - 1)
                G_k = self.H @ A_pow @ self.F
                G_list.append(G_k)
                
            Py = torch.cat(Py_list, dim=0)
            G = torch.cat(G_list, dim=0)
            
            # 5. Odhad PHT
            PHT = self.K @ Py0 + torch.linalg.pinv(G) @ Py
            
            # 6. Odhad nového K
            K_est = PHT @ torch.linalg.pinv(Py0)
            
            # Update zisku pro další iteraci
            self.K = K_est
            
        # Po proběhnutí všech iterací uděláme jeden finální průchod s nejlepším nalezeným K,
        # abychom získali finální filtrované stavy.
        final_x_filt, final_inov = self._filter_sequence(y_seq, Ex0)
        
        # Aby byl výstup kompatibilní se zbytkem tvého frameworku, vrátíme slovník.
        # P_filtered sice Mehra explicitně nevrací (dopočítat ho lze, ale je to pracné),
        # takže vrátíme dummy nuly, jak to děláš doteď, a K zduplikujeme pro celou historii,
        # protože je teď konstantní pro celou trajektorii.
        
        P_filtered_history = torch.zeros(seq_len, self.nx, self.nx, device=self.device)
        # Rozkopírování finálního K do sekvence pro kompatibilitu s tvými grafy
        K_history = self.K.unsqueeze(0).repeat(seq_len, 1, 1) 
        
        return {
            'x_filtered': final_x_filt,
            'P_filtered': P_filtered_history,
            'Kalman_gain': K_history,
            'innovation': final_inov.T # Transponujeme zpět na tvar (seq_len, nz)
        }