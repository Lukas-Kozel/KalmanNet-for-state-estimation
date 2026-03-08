import torch
import numpy as np
from tqdm import tqdm

class AdaptiveKalmanFilter_mehra_offline:
    def __init__(self, model, num_iterations=5):
        """
        Offline (Dávková) verze Mehrova filtru.
        Zpracuje celou trajektorii naráz a opakuje výpočet zisku `num_iterations` krát.
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
        """Pomocná metoda: přefiltruje data s aktuálním (fixním) ziskem K a vrátí inovace a stavy."""
        seq_len = y_seq.shape[0]
        inov_history = torch.zeros(self.nz, seq_len, device=self.device,dtype=self.dtype)
        x_filt_history = torch.zeros(seq_len, self.nx, device=self.device,dtype=self.dtype)
        
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
        self.K = 0.1 * torch.linalg.pinv(self.H)
        
        print(f"\n--- Offline Mehra: Start adaptace ---")
        print(f"Iterace 0 (Nástřel): K[0,0] = {self.K[0,0].item():.4f}")
        # Opakujeme proces filtrace a odhadu M-krát
        for m in tqdm(range(self.num_iterations), desc="Offline filtrace s adaptivním K", leave=False):
            # 1. Kontrola stability aktuálního K
            A = self.F - self.F @ self.K @ self.H

            # 2. Filtrace celé sekvence s aktuálním K pro získání inovační posloupnosti
            x_filt_hist, inov = self._filter_sequence(y_seq, Ex0)
            
            # 3. Odhad Py0 (Kovariance inovací)
            Py0 = (inov @ inov.T) / N
            
            # 4. Sestavení Py a G
            Py_list = []
            G_list = []
            
            for k in range(1, self.nx + 10):
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
            
            A_next = self.F - self.F @ K_est @ self.H
            eigvals = torch.linalg.eigvals(A_next)
            max_eig = torch.max(torch.abs(eigvals))
            if max_eig >= 1.0:
                print(f"Varování: V iteraci {m} se filtr stal nestabilním (max_eig={max_eig.item()}). Končím iterace.")
                break # Pokud je K nestabilní, dál už to nemá smysl iterovat
        
            # Update zisku pro další iteraci
            self.K = K_est
            
        # Průchod s nalezeným nejlepším K
        final_x_filt, final_inov = self._filter_sequence(y_seq, Ex0)
        
        
        P_filtered_history = torch.zeros(seq_len, self.nx, self.nx, device=self.device,dtype=self.dtype)
        K_history = self.K.unsqueeze(0).repeat(seq_len, 1, 1) 
        
        return {
            'x_filtered': final_x_filt,
            'P_filtered': P_filtered_history,
            'Kalman_gain': K_history,
            'innovation': final_inov.T
        }