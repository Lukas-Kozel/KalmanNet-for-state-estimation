import torch

class KalmanFilter:
    """
    Kalmanův filtr pro t-invaritantní systém s lineární dynamikou.
    """
    def __init__(self,Ex0,P0, F, H, Q, R):
        self.Ex0 = Ex0  # očekávaná hodnota počátečního stavu
        self.P0 = P0  # Počáteční kovarianční matice
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state_dim = F.shape[0]
        self.obs_dim = H.shape[0]
        

    def predict_step(self, x_filtered, P_filtered):
        x_predict = self.F @ x_filtered
        P_predict = self.F @ P_filtered @ self.F.T + self.Q
        return x_predict, P_predict

    def filtration_step(self, x_predict, y_t, P_predict):
        innovation = self.compute_innovation(y_t, x_predict)
        K = self.compute_kalman_gain(P_predict)
        x_filtered = x_predict + K @ innovation
        P_filtered = (torch.eye(self.state_dim)- K @ self.H) @ P_predict @ (torch.eye(self.state_dim)- K @ self.H).T + K @ self.R @ K.T # Joseph form
        return x_filtered, P_filtered, K, innovation

    def compute_kalman_gain(self, P_predict):
        return P_predict @ self.H.T @ torch.linalg.inv(self.H @ P_predict @ self.H.T + self.R)
        
    
    def compute_innovation(self, y_t, x_predict):
        return y_t - self.H @ x_predict
    
    def apply_filter(self, y_seq)->dict:
        seq_len = y_seq.shape[0]

        stored_data = {
            'x_filtered': torch.zeros(seq_len, self.state_dim),
            'P_filtered': torch.zeros(seq_len, self.state_dim, self.state_dim),
            'Kalman_gain': torch.zeros(seq_len, self.state_dim, self.obs_dim),
            'innovation': torch.zeros(seq_len, self.obs_dim),
            'x_predict': torch.zeros(seq_len, self.state_dim),
            'P_predict': torch.zeros(seq_len, self.state_dim, self.state_dim)
        }

        # inicializace
        x_est = self.Ex0.clone()
        P_est = self.P0.clone()

        for t in range(seq_len):
            # predikce
            x_predict, P_predict = self.predict_step(x_est, P_est)

            # filtrace
            y_t = y_seq[t].unsqueeze(-1)
            x_est, P_est, K, innovation = self.filtration_step(x_predict, y_t, P_predict)

            # uložení výsledků
            stored_data['x_filtered'][t] = x_est.squeeze()
            stored_data['P_filtered'][t] = P_est
            stored_data['Kalman_gain'][t] = K
            stored_data['innovation'][t] = innovation.squeeze()
            stored_data['x_predict'][t] = x_predict.squeeze()
            stored_data['P_predict'][t] = P_predict

        return stored_data