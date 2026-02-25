import torch
import numpy as np

from tqdm import tqdm

class KalmanFilter:
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

class AdaptiveKalmanFilter_mehra:
    def __init__(self, model, window_size=200):
        self.kf = KalmanFilter(model)
        self.model = model
        self.device = model.Q.device
        
        self.nx = self.kf.F.shape[0]
        self.nz = self.kf.H.shape[0]
        
        # Nastavení velikosti klouzavého okna (musí být větší než nx pro statistickou smysluplnost)
        self.window_size = window_size
        self.inov_buffer = []
        
        self.reset()

    def reset(self, Ex0=None, P0=None):
        """Vymaže buffery a zresetuje vnitřní stav před novou trajektorií."""
        if Ex0 is None: Ex0 = self.model.Ex0
        if P0 is None: P0 = self.model.P0
            
        self.kf.reset(Ex0, P0)
        self.x_predict = Ex0.clone().detach().reshape(self.nx, 1)
        self.inov_buffer = []
        
        # 1. Zvolme stabilní K lineárního filtru
        self.K = 0.7 * torch.linalg.pinv(self.kf.H)
    
    
    def step(self, y_t):
        y_t = y_t.reshape(self.nz, 1)
        
        # 2. Spočteme chybu před měřením [cite: 19]
        inov = y_t - self.kf.H @ self.x_predict
        if torch.isnan(inov).any() or torch.isinf(inov).any():
            print("DEBUG [Krok 1]: Inovace (inov) obsahuje NaN nebo Inf! Zřejmě explodovala predikce x_predict.")
            breakpoint()
            return

        # Přidání do klouzavého okna a ořezání starých dat
        self.inov_buffer.append(inov)
        if len(self.inov_buffer) > self.window_size:
            self.inov_buffer.pop(0)
            
        N_current = len(self.inov_buffer)
        
        # Výpočet matic proběhne, až když máme v okně dostatek dat (min. nx + 1)
        if N_current >= self.window_size:
            # Složení historie inovací do jedné matice pro vektorizovaný výpočet
            inov_tensor = torch.cat(self.inov_buffer, dim=1)
            
            # 3. Odhad kovarianční matice chyby P0_y [cite: 43]
            Py0 = (inov_tensor @ inov_tensor.T) / N_current
            if torch.isnan(Py0).any() or torch.isinf(Py0).any():
                print("DEBUG [Krok 2]: Matice Py0 obsahuje NaN/Inf!")
                breakpoint()
                return
            Py_list = []
            G_list = []
            
            # Matice systému pro fiktivní model [cite: 23]
            A = self.kf.F - self.kf.F @ self.K @ self.kf.H
            
            for l in range(1, self.nx + 1):
                # Zpožděná inovační posloupnost pro dané l 
                inov_shift = inov_tensor[:, l:]
                inov_orig = inov_tensor[:, :-l]
                
                # Výpočet zpožděných autokovariancí
                Py_l = (inov_shift @ inov_orig.T) / (N_current - l)
                Py_list.append(Py_l)
                
                # Sestavení matice G [cite: 70]
                A_pow = torch.matrix_power(A, l - 1)
                G_l = self.kf.H @ A_pow @ self.kf.F
                G_list.append(G_l)
                
            Py = torch.cat(Py_list, dim=0)
            G = torch.cat(G_list, dim=0)
            
            # 4. a 5. Odhad součinu P^e * H^T [cite: 83]
            PHT = self.K @ Py0 + torch.linalg.pinv(G) @ Py
            
            # 6. Odhad Kalmanova zisku (s malým přídavkem pro numerickou stabilitu inverze)
            Py0_inv = torch.linalg.inv(Py0 + 1e-6 * torch.eye(self.nz, device=self.device))
            if torch.isnan(Py0_inv).any() or torch.isinf(Py0_inv).any():
                print("DEBUG [Krok 3]: Inverze Py0_inv selhala (vrací NaN/Inf). Matice je pravděpodobně singulární!")
                breakpoint()
                return
            K_est = PHT @ Py0_inv
            if torch.isnan(K_est).any() or torch.isinf(K_est).any():
                print("DEBUG [Krok 4]: Vypočtený zisk K_est obsahuje NaN/Inf!")
                breakpoint()
                return
            # Update zisku - pro vyšší stabilitu online filtru je možné ho lehce filtrovat 
            # (např. self.K = 0.9 * self.K + 0.1 * K_est), ale zde jedeme čistý přepis:
            self.K = K_est
            
        # 7. Filtrace a predikce na další krok (rovnice lineárního filtru) [cite: 8]
        x_filtered = self.x_predict + self.K @ inov
        self.x_predict = self.kf.F @ x_filtered
        if torch.isnan(self.x_predict).any() or torch.isinf(self.x_predict).any():
            print("DEBUG [Krok 5]: Nový predikovaný stav x_predict obsahuje NaN/Inf! Krok filtrace selhal.")
            breakpoint()
        
        return x_filtered, self.K

    def process_sequence(self, y_seq, Ex0=None, P0=None):
        seq_len = y_seq.shape[0]
        
        # PŘIDÁNO: Reset před trajektorií
        self.reset(Ex0, P0)
        
        x_filtered_history = torch.zeros(seq_len, self.nx, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.nx, self.nx, device=self.device)
        kalman_gain_history = torch.zeros(seq_len, self.nx, self.nz, device=self.device)
        innovation_history = torch.zeros(seq_len, self.nz, device=self.device)
        
        for k in tqdm(range(seq_len), desc="Online filtrace s adaptivním K", leave=False):
            # Uložení inovace ještě před úpravou stavu
            z_t = y_seq[k].reshape(self.nz, 1)
            inov_k = z_t - self.kf.H @ self.x_predict
            
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