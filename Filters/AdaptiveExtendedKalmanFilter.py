import torch
from torch.func import jacrev 

class AdaptiveExtendedKalmanFilter:
    def __init__(self, system_model, Q_init=None, R_init=None,alpha=0.3):
   
        self.device = system_model.Q.device
        
        self.f = system_model.f
        self.h = system_model.h

        self.alpha = alpha
        
        # Pokud je systém lineární, Jakobiány jsou konstantní matice F a H
        self.is_linear_f = getattr(system_model, 'is_linear_f', False)
        self.is_linear_h = getattr(system_model, 'is_linear_h', False)
        if self.is_linear_f:
            self.F = system_model.F
        if self.is_linear_h:
            self.H = system_model.H
        
 
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.x_filtered_prev = None
        self.P_filtered_prev = None

        self.Q_prev = None
        self.R_prev = None

        
        self.reset(system_model.Ex0, system_model.P0,Q_init=Q_init, R_init=R_init)

    def reset(self, Ex0=None, P0=None,Q_init=None, R_init=None):
        """
        Inicializuje nebo resetuje stav filtru.
        """
        if Ex0 is not None:
            self.x_filtered_prev = Ex0.clone().detach().reshape(self.state_dim, 1)
        if P0 is not None:
            self.P_filtered_prev = P0.clone().detach()
        if Q_init is not None:
            self.Q_prev = Q_init.clone().detach()
        if R_init is not None:
            self.R_prev = R_init.clone().detach()

    def predict_step(self, x_filtered, P_filtered):
        if self.is_linear_f:
            F_t = self.F
        else:
            F_t = jacrev(self.f)(x_filtered.squeeze()).reshape(self.state_dim, self.state_dim)

        x_predict = self.f(x_filtered)
        P_predict = F_t @ P_filtered @ F_t.T + self.Q_prev
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        y_t = y_t.reshape(self.obs_dim, 1)

        if self.is_linear_h:
            H_t = self.H
        else:
            H_t = jacrev(self.h)(x_predict.squeeze()).reshape(self.obs_dim, self.state_dim)
            
        innovation = y_t - self.h(x_predict)
        S = H_t @ P_predict @ H_t.T + self.R_prev
        K = P_predict @ H_t.T @ torch.linalg.inv(S)
        x_filtered = x_predict + K @ innovation
        

        I = torch.eye(self.state_dim, device=self.device)
        P_filtered = (I - K @ H_t) @ P_predict @ (I - K @ H_t).T + K @ self.R_prev @ K.T
        
        p_floor = 1e-6
        P_filtered = P_filtered + torch.eye(self.state_dim, device=self.device) * p_floor

        return x_filtered, P_filtered, K, innovation
    
    def measurement_covariance_update(self, y_t,x_filtered, P_filtered):
        residual = y_t - self.h(x_filtered)
        if self.is_linear_h:
            H = self.H
        else:
            H = jacrev(self.h)(x_filtered.squeeze()).reshape(self.obs_dim, self.state_dim)

        R = self.alpha*self.R_prev + (1-self.alpha) *(residual @ residual.T + H @ P_filtered @ H.T)
        r_floor = 1e-6
        R = R + torch.eye(self.obs_dim, device=self.device) * r_floor
        return R
    
    def state_covariance_update(self, K, innovation):
        Q = self.alpha*self.Q_prev + (1-self.alpha) *(K @ innovation @ innovation.T @ K.T)
        q_floor = 1e-7  # Minimální hodnota procesního šumu
        Q = Q + torch.eye(self.state_dim, device=self.device) * q_floor 
        return Q

    def step(self, y_t):
        """
        Provede jeden kompletní krok filtrace (predict + update) pro online použití.
        """
        # 1. Predikce z uloženého interního stavu
        x_predict, P_predict = self.predict_step(self.x_filtered_prev, self.P_filtered_prev)
        
        # 2. Update s novým měřením
        x_filtered, P_filtered, K, innovation = self.update_step(x_predict, y_t, P_predict)
        
        self.Q_prev = self.state_covariance_update(K, innovation)
        self.R_prev = self.measurement_covariance_update(y_t, x_filtered, P_filtered) 
        # 3. Aktualizace interního stavu pro další volání
        self.x_filtered_prev = x_filtered
        self.P_filtered_prev = P_filtered
        
        return x_filtered, P_filtered
    
    def process_sequence(self, y_seq, Ex0=None, P0=None):
            """
            Zpracuje celou sekvenci měření `y_seq` (offline) a vrátí detailní historii.
            """
            # Pokud nejsou zadány, použije defaultní hodnoty z `__init__`
            x_est = Ex0.clone().detach().reshape(self.state_dim, 1) if Ex0 is not None else self.x_filtered_prev.clone()
            P_est = P0.clone().detach() if P0 is not None else self.P_filtered_prev.clone()
            
            seq_len = y_seq.shape[0]

            x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
            P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
            x_predict_history = torch.zeros(seq_len, self.state_dim, device=self.device)
            P_predict_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
            kalman_gain_history = torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device)
            innovation_history = torch.zeros(seq_len, self.obs_dim, device=self.device)

            for t in range(seq_len):
                # 1. Predict
                x_predict, P_predict = self.predict_step(x_est, P_est)
                
                # 2. Update
                x_est, P_est, K, innovation = self.update_step(x_predict, y_seq[t], P_predict)

                self.Q_prev = self.state_covariance_update(K, innovation)
                self.R_prev = self.measurement_covariance_update(y_seq[t], x_est, P_filtered=P_est) 
                
                x_filtered_history[t] = x_est.squeeze()
                P_filtered_history[t] = P_est
                x_predict_history[t] = x_predict.squeeze()
                P_predict_history[t] = P_predict
                kalman_gain_history[t] = K
                innovation_history[t] = innovation.squeeze()

            return {
                'x_filtered': x_filtered_history,
                'P_filtered': P_filtered_history,
                'x_predict': x_predict_history,
                'P_predict': P_predict_history,
                'Kalman_gain': kalman_gain_history,
                'innovation': innovation_history
            }