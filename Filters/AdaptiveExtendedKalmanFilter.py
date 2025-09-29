import torch
from torch.func import jacrev 

class AdaptiveExtendedKalmanFilter:
    """
    Volba parametru alpha velmi závisí na konkrétní aplikaci.
    Hodnota blízká 1 znamená pomalou adaptaci (více váhy na předchozí odhad),
    zatímco hodnota blízká 0 znamená rychlou adaptaci (více váhy na aktuální měření).
    Například pro lineární 2D systém může být vhodná hodnota kolem 0.9,
    zatímco pro nelineární systémy může být lepší hodnota kolem 0
    """
    def __init__(self, system_model, Q_init=None, R_init=None, alpha=0.3):
   
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

        # Interní stavy filtru
        self.x_filtered_prev = None
        self.P_filtered_prev = None
        self.Q_prev = None
        self.R_prev = None
        
        self.reset(system_model.Ex0, system_model.P0, Q_init=Q_init, R_init=R_init)

    def reset(self, Ex0=None, P0=None, Q_init=None, R_init=None):
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
            # jacrev očekává 1D tenzor, proto .squeeze()
            F_t = jacrev(self.f)(x_filtered.squeeze()).reshape(self.state_dim, self.state_dim)
        
        x_predict = self.f(x_filtered.T).T
        
        P_predict = F_t @ P_filtered @ F_t.T + self.Q_prev
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        # Zajistíme, že y_t je sloupcový vektor
        y_t = y_t.reshape(self.obs_dim, 1)

        if self.is_linear_h:
            H_t = self.H
        else:
            # jacrev očekává 1D tenzor, proto .squeeze()
            H_t = jacrev(self.h)(x_predict.squeeze()).reshape(self.obs_dim, self.state_dim)


        y_predict = self.h(x_predict.T).T
        
        innovation = y_t - y_predict
        S = H_t @ P_predict @ H_t.T + self.R_prev
        K = P_predict @ H_t.T @ torch.linalg.inv(S)
        x_filtered = x_predict + K @ innovation

        I = torch.eye(self.state_dim, device=self.device)
        # Použití Joseph form pro numericky stabilnější update kovariance
        P_filtered = (I - K @ H_t) @ P_predict @ (I - K @ H_t).T + K @ self.R_prev @ K.T
        
        # Zajištění pozitivní definitnosti
        p_floor = 1e-6
        P_filtered = P_filtered + torch.eye(self.state_dim, device=self.device) * p_floor

        return x_filtered, P_filtered, K, innovation
    
    def measurement_covariance_update(self, y_t, x_filtered, P_filtered):

        y_predict = self.h(x_filtered.T).T
        
        # y_t [O] -> .reshape -> [O, 1]
        residual = y_t.reshape(self.obs_dim, 1) - y_predict

        if self.is_linear_h:
            H = self.H
        else:
            H = jacrev(self.h)(x_filtered.squeeze()).reshape(self.obs_dim, self.state_dim)

        R = self.alpha * self.R_prev + (1 - self.alpha) * (residual @ residual.T + H @ P_filtered @ H.T)
        
        # Zajištění pozitivní definitnosti
        r_floor = 1e-6
        R = R + torch.eye(self.obs_dim, device=self.device) * r_floor
        return R
    
    def state_covariance_update(self, K, innovation):
        Q = self.alpha * self.Q_prev + (1 - self.alpha) * (K @ innovation @ innovation.T @ K.T)
        
        # Zajištění pozitivní definitnosti
        q_floor = 1e-7
        Q = Q + torch.eye(self.state_dim, device=self.device) * q_floor 
        return Q

    def step(self, y_t):
        """
        Provede jeden kompletní krok filtrace (predict + update) pro online použití.
        """
        x_predict, P_predict = self.predict_step(self.x_filtered_prev, self.P_filtered_prev)
        x_filtered, P_filtered, K, innovation = self.update_step(x_predict, y_t, P_predict)
        
        self.Q_prev = self.state_covariance_update(K, innovation)
        self.R_prev = self.measurement_covariance_update(y_t, x_filtered, P_filtered) 
        
        self.x_filtered_prev = x_filtered
        self.P_filtered_prev = P_filtered
        
        return x_filtered, P_filtered
    
    def process_sequence(self, y_seq, Ex0=None, P0=None):
        """
        Zpracuje celou sekvenci měření `y_seq` (offline) a vrátí detailní historii.
        """
        x_est = Ex0.clone().detach().reshape(self.state_dim, 1)
        P_est = P0.clone().detach()
        
        seq_len = y_seq.shape[0]

        # Inicializace tenzorů pro historii
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)

        # Na začátku (čas t=0) je "filtrovaný" stav roven počátečnímu stavu
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est

        for t in range(seq_len):
            x_predict, P_predict = self.predict_step(x_est, P_est)
            x_est, P_est, K, innovation = self.update_step(x_predict, y_seq[t], P_predict)

            # Adaptace Q a R se počítá AŽ PO update kroku
            self.Q_prev = self.state_covariance_update(K, innovation)
            self.R_prev = self.measurement_covariance_update(y_seq[t], x_est, P_filtered=P_est) 
            
            x_filtered_history[t] = x_est.squeeze()
            P_filtered_history[t] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history
        }