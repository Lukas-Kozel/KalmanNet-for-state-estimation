import torch

class UnscentedKalmanFilterTAN:
    def __init__(self, system_model):
        """
        Inicializace zjednodušeného, ale robustního a vektorizovaného UKF.
        Tato verze nepoužívá parametry alpha, beta, kappa.
        """
        self.device = system_model.Q.device
        
        self.system = system_model
        self.f = system_model.f
        self.h = system_model.h
        
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]
        
        self.n_sigma_points = 2 * self.state_dim + 1
        self.weights = torch.full((self.n_sigma_points,), 1.0 / self.n_sigma_points, device=self.device)

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


    def _get_sigma_points(self, x_mean, P):
        """
        Vypočítá sigma body na základě průměru a kovariance.
        """
        if not torch.all(torch.isfinite(P)):
            raise ValueError("Covariance matrix P contains NaNs or Infs!")

        try:
            # Přidání malého jitteru pro numerickou stabilitu
            jitter = torch.eye(self.state_dim, device=self.device) * 1e-6
            S = torch.linalg.cholesky(P + jitter, upper=False)
        except torch.linalg.LinAlgError:
            # print("Cholesky selhal! Používám vlastní odmocninu matice.")
            eigvals, eigvecs = torch.linalg.eigh(P)
            S = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T

        # scale = torch.sqrt(torch.tensor(self.state_dim, device=self.device, dtype=torch.float32))
        n = self.state_dim
        scale = torch.sqrt(torch.tensor((n * 2 + 1) / 2.0, device=self.device, dtype=torch.float32))
        # Vytvoření sigma bodů (x_mean musí být sloupcový vektor [state_dim, 1])
        sigma_points = x_mean.repeat(1, self.n_sigma_points)
        sigma_points[:, 1:self.state_dim + 1] += scale * S
        sigma_points[:, self.state_dim + 1:] -= scale * S
        
        return sigma_points

    def predict_step(self, x_filtered, P_filtered, u_current=None):
        # Krok 1: Výpočet sigma bodů (zůstává stejný)
        # sigma_points má tvar [state_dim, n_sigma_points]
        sigma_points = self._get_sigma_points(x_filtered, P_filtered)
        
        # Krok 2: Propagace sigma bodů přes deterministickou část dynamiky 'f'
        # self.f očekává dávku [batch_size, state_dim], proto musíme transponovat
        # propagated_points má tvar [n_sigma_points, state_dim]
        propagated_points_f = self.f(sigma_points.T)

        # ### ZDE JE KLÍČOVÁ OPRAVA ###
        # Krok 3: Přičtení vstupu 'u' ke KAŽDÉMU propagovanému bodu
        if u_current is not None:
            # u_current má tvar [state_dim]. Rozšíříme ho, aby se dal sečíst s maticí.
            # PyTorch se postará o broadcasting (rozšíření) u_current na všechny řádky.
            propagated_points = propagated_points_f + u_current
        else:
            propagated_points = propagated_points_f

        # Zpětná transpozice pro další výpočty
        # propagated_points má nyní tvar [state_dim, n_sigma_points]
        propagated_points = propagated_points.T
        
        # Krok 4: Výpočet predikovaného stavu a kovariance (Unscented Transform)
        # (Tato část je již správně)
        x_predict = (propagated_points @ self.weights).unsqueeze(1)
        diff = propagated_points - x_predict
        P_predict = (diff * self.weights) @ diff.T + self.Q

        return x_predict, P_predict

    def update_step(self, x_predict, P_predict, y_t):
        y_t = y_t.reshape(self.obs_dim, 1)

        sigma_points = self._get_sigma_points(x_predict, P_predict)

        measurement_points = self.h(sigma_points.T).T
        
        y_hat = (measurement_points @ self.weights).unsqueeze(1)
        
        y_diff = measurement_points - y_hat
        x_diff = sigma_points - x_predict
        
        P_yy = (y_diff * self.weights) @ y_diff.T + self.R
        P_xy = (x_diff * self.weights) @ y_diff.T
        
        innovation = y_t - y_hat
        K = P_xy @ torch.linalg.inv(P_yy)
        
        x_filtered = x_predict + K @ innovation
        P_filtered = P_predict - K @ P_xy.T
        
        return x_filtered, P_filtered, K, innovation


    def step(self, y_t, u_t=None):
        """
        Provede jeden kompletní krok filtrace pro ONLINE použití.
        Vrací nejlepší odhad pro aktuální čas 't'.
        """
        x_filtered, P_filtered, _, _ = self.update_step(self.x_predict_current, self.P_predict_current, y_t)

        if u_t is not None:
            x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered, u_t)
        else:
            x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered)

        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next

        return x_filtered, P_filtered
    

    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None):
        """
        Zpracuje celou sekvenci měření.
        Tato metoda je nyní plně kompatibilní s vaším vyhodnocovacím skriptem.
        """
        x_est = Ex0.clone().detach().reshape(self.state_dim, 1)
        P_est = P0.clone().detach()

        seq_len = y_seq.shape[0]

        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        kalman_gain_history = torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device)
        innovation_history = torch.zeros(seq_len, self.obs_dim, device=self.device)

        x_predict_k = Ex0.clone().detach().reshape(self.state_dim, 1)
        P_predict_k = P0.clone().detach()

        for k in range(seq_len):
            x_filtered, P_filtered, K, innovation = self.update_step(x_predict_k, P_predict_k, y_seq[k])

            u_k = u_sequence[k, :] if u_sequence is not None else None
            x_predict_k_plus_1, P_predict_k_plus_1 = self.predict_step(x_filtered, P_filtered,u_k)

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