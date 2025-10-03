import torch
from torch.func import jacrev

class ExtendedKalmanFilter:
    def __init__(self, system_model):

        self.device = system_model.Q.device

        self.f = system_model.f
        self.h = system_model.h

        # Pokud je systém lineární, Jakobiány jsou konstantní matice F a H
        self.is_linear_f = getattr(system_model, 'is_linear_f', False)
        self.is_linear_h = getattr(system_model, 'is_linear_h', False)
        if self.is_linear_f:
            self.F = system_model.F
        if self.is_linear_h:
            self.H = system_model.H


        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)

        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]

        self.x_filtered_prev = None
        self.P_filtered_prev = None

        # Interní stav pro online použití: predikce pro aktuální krok
        self.x_predict_current = None
        self.P_predict_current = None

        self.reset(system_model.Ex0, system_model.P0)

    def reset(self, Ex0, P0):
        """
        Inicializuje nebo resetuje interní stav filtru pro online použití.
        """
        self.x_predict_current = Ex0.clone().detach().reshape(self.state_dim, 1)
        self.P_predict_current = P0.clone().detach()

    def predict_step(self, x_filtered, P_filtered):
        if self.is_linear_f:
            F_t = self.F
        else:
            F_t = jacrev(self.f)(x_filtered.squeeze()).reshape(self.state_dim, self.state_dim)
        
        # [State_Dim, 1] -> .T -> [1, State_Dim] -> f() -> [1, State_Dim] -> .T -> [State_Dim, 1]
        x_predict = self.f(x_filtered.T).T

        P_predict = F_t @ P_filtered @ F_t.T + self.Q
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        y_t = y_t.reshape(self.obs_dim, 1)

        if self.is_linear_h:
            H_t = self.H
        else:
            H_t = jacrev(self.h)(x_predict.squeeze()).reshape(self.obs_dim, self.state_dim)

        y_predict = self.h(x_predict.T).T
        innovation = y_t - y_predict
        
        S = H_t @ P_predict @ H_t.T + self.R
        K = P_predict @ H_t.T @ torch.linalg.inv(S)
        x_filtered = x_predict + K @ innovation

        I = torch.eye(self.state_dim, device=self.device)
        P_filtered = (I - K @ H_t) @ P_predict @ (I - K @ H_t).T + K @ self.R @ K.T

        return x_filtered, P_filtered, K, innovation

    def step(self, y_t):
        """
        Provede jeden kompletní krok filtrace pro ONLINE použití.
        Vrací nejlepší odhad pro aktuální čas 't'.
        """
        x_filtered, P_filtered, _, _ = self.update_step(self.x_predict_current, y_t, self.P_predict_current)

        x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered)

        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next

        return x_filtered, P_filtered

    def process_sequence(self, y_seq, Ex0=None, P0=None):
            """
            Zpracuje celou sekvenci měření `y_seq` (offline) a vrátí detailní historii.
            """

            seq_len = y_seq.shape[0]

            x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
            P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
            kalman_gain_history = torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device)
            innovation_history = torch.zeros(seq_len, self.obs_dim, device=self.device)

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