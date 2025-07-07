import torch
from torch.func import jacrev 

class ExtendedKalmanFilter:
    def __init__(self, system_model):
        self.f = system_model.f
        self.h = system_model.h
        self.Q = system_model.Q.clone().detach()
        self.R = system_model.R.clone().detach()
        self.Ex0 = system_model.Ex0.clone().detach()
        self.P0 = system_model.P0.clone().detach()
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]

    def predict_step(self, x_filtered, P_filtered):
        # Vypočítáme Jacobiho matici F_t v bodě x_filtered
        F_t = jacrev(self.f)(x_filtered.squeeze()).reshape(self.state_dim, self.state_dim)
        
        # Predikce stavu pomocí nelineární funkce f
        x_predict = self.f(x_filtered)
        
        # Predikce kovariance pomocí linearizované matice F_t
        P_predict = F_t @ P_filtered @ F_t.T + self.Q
        
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        # Jacobiho matice H_t v bodě x_predict
        H_t = jacrev(self.h)(x_predict.squeeze()).reshape(self.obs_dim, self.state_dim)
        
        # Inovace pomocí nelineární funkce h
        innovation = y_t - self.h(x_predict)
        
        # Inovační kovariance
        S = H_t @ P_predict @ H_t.T + self.R
        
        # Kalmanův zisk
        K = P_predict @ H_t.T @ torch.linalg.inv(S)
        
        # Aktualizace odhadu stavu
        x_filtered = x_predict + K @ innovation
        
        # Aktualizace kovariance (stabilní Josephova forma)
        I = torch.eye(self.state_dim)
        P_filtered = (I - K @ H_t) @ P_predict @ (I - K @ H_t).T + K @ self.R @ K.T
        
        return x_filtered, P_filtered, K, innovation

    def apply_filter(self, y_seq):
        seq_len = y_seq.shape[0]

        stored_data = {
            'x_filtered': torch.zeros(seq_len, self.state_dim),
            'P_filtered': torch.zeros(seq_len, self.state_dim, self.state_dim),
            'Kalman_gain': torch.zeros(seq_len, self.state_dim, self.obs_dim),
            'innovation': torch.zeros(seq_len, self.obs_dim),
            'x_predict': torch.zeros(seq_len, self.state_dim),
            'P_predict': torch.zeros(seq_len, self.state_dim, self.state_dim)
        }

        # Inicializace
        x_est = self.Ex0.clone()
        P_est = self.P0.clone()

        for t in range(seq_len):
            # Predikce
            x_predict, P_predict = self.predict_step(x_est, P_est)

            # Filtrace
            y_t = y_seq[t].unsqueeze(-1)
            x_est, P_est, K, innovation = self.update_step(x_predict, y_t, P_predict)

            stored_data['x_filtered'][t] = x_est.squeeze()
            stored_data['P_filtered'][t] = P_est
            stored_data['Kalman_gain'][t] = K
            stored_data['innovation'][t] = innovation.squeeze()
            stored_data['x_predict'][t] = x_predict.squeeze()
            stored_data['P_predict'][t] = P_predict

        return stored_data