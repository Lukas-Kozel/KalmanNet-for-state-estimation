import torch
from torch.func import jacrev 

class ExtendedKalmanFilter:
    def __init__(self, system_model):
   
        self.device = system_model.Q.device
        
        self.f = system_model.f
        self.h = system_model.h
        
 
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        self.Ex0 = system_model.Ex0.clone().detach().to(self.device)
        self.P0 = system_model.P0.clone().detach().to(self.device)
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]

    def predict_step(self, x_filtered, P_filtered):
        F_t = jacrev(self.f)(x_filtered.squeeze()).reshape(self.state_dim, self.state_dim)
        x_predict = self.f(x_filtered)
        P_predict = F_t @ P_filtered @ F_t.T + self.Q
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        H_t = jacrev(self.h)(x_predict.squeeze()).reshape(self.obs_dim, self.state_dim)
        innovation = y_t - self.h(x_predict)
        S = H_t @ P_predict @ H_t.T + self.R
        K = P_predict @ H_t.T @ torch.linalg.inv(S)
        x_filtered = x_predict + K @ innovation
        

        I = torch.eye(self.state_dim, device=self.device)
        P_filtered = (I - K @ H_t) @ P_predict @ (I - K @ H_t).T + K @ self.R @ K.T
        
        return x_filtered, P_filtered, K, innovation

    def apply_filter(self, y_seq):
        seq_len = y_seq.shape[0]

        stored_data = {
            'x_filtered': torch.zeros(seq_len, self.state_dim, device=self.device),
            'P_filtered': torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device),
            'Kalman_gain': torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device),
            'innovation': torch.zeros(seq_len, self.obs_dim, device=self.device),
            'x_predict': torch.zeros(seq_len, self.state_dim, device=self.device),
            'P_predict': torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        }

        x_est = self.Ex0.clone()
        P_est = self.P0.clone()

        for t in range(seq_len):
            x_predict, P_predict = self.predict_step(x_est, P_est)
            y_t = y_seq[t].unsqueeze(-1)
            x_est, P_est, K, innovation = self.update_step(x_predict, y_t, P_predict)

            stored_data['x_filtered'][t] = x_est.squeeze()
            stored_data['P_filtered'][t] = P_est
            stored_data['Kalman_gain'][t] = K
            stored_data['innovation'][t] = innovation.squeeze()
            stored_data['x_predict'][t] = x_predict.squeeze()
            stored_data['P_predict'][t] = P_predict

        return stored_data