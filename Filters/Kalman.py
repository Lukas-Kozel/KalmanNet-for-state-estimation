import torch

class KalmanFilter:
    """
    Kalmanův filtr pro t-invaritantní systém s lineární dynamikou.
    """
    def __init__(self,model):
        self.device = model.Q.device
        self.model = model
        self.Ex0 = model.Ex0  # očekávaná hodnota počátečního stavu
        self.P0 = model.P0  # Počáteční kovarianční matice
        self.F = model.F
        self.H = model.H
        self.Q = model.Q
        self.R = model.R
        self.state_dim = self.F.shape[0]
        self.obs_dim = self.H.shape[0]

        # Interní stav pro online použití: predikce pro aktuální krok
        self.x_predict_current = None
        self.P_predict_current = None

        self.reset(model.Ex0, model.P0)

    def reset(self, Ex0, P0):
        """
        Inicializuje nebo resetuje interní stav filtru pro online použití.
        """
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
        P_filtered = (I - K @ self.H) @ P_predict @ (I - K @ self.H).T + K @ self.R @ K.T # Joseph form
        return x_filtered, P_filtered, K, innovation

    def compute_kalman_gain(self, P_predict):
        return P_predict @ self.H.T @ torch.linalg.inv(self.H @ P_predict @ self.H.T + self.R)
        
    
    def compute_innovation(self, y_t, x_predict):
        return y_t - self.H @ x_predict
    
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