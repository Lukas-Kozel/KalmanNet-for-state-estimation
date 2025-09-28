import torch

class UnscentedKalmanFilter:
    def __init__(self, system_model, alpha=1e-3, beta=2., kappa=0.):
        """
        Inicializace Unscented Kalman Filtru.
        
        Args:
            system_model: Objekt obsahující dynamiku systému (f, h, Q, R, ...).
            alpha, beta, kappa: Parametry pro Unscented Transform.
        """
        self.device = system_model.Q.device
        
        self.f = system_model.f
        self.h = system_model.h
        
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]
        
        # Parametry pro Unscented Transform
        self.L = self.state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Výpočet vah pro sigma body
        self.lambda_ = self.alpha**2 * (self.L + self.kappa) - self.L
        
        # Váhy pro výpočet průměru
        self.Wm = torch.full((2 * self.L + 1,), 0.5 / (self.L + self.lambda_)).to(self.device)
        self.Wm[0] = self.lambda_ / (self.L + self.lambda_)
        
        # Váhy pro výpočet kovariance
        self.Wc = self.Wm.clone()
        self.Wc[0] += (1 - self.alpha**2 + self.beta)

        # Interní stavy filtru
        self.x_filtered_prev = None
        self.P_filtered_prev = None
        self.reset(system_model.Ex0, system_model.P0)

    def reset(self, Ex0=None, P0=None):
        """
        Inicializuje nebo resetuje stav filtru. Stejné jako v EKF.
        """
        if Ex0 is not None:
            self.x_filtered_prev = Ex0.clone().detach().reshape(self.state_dim, 1)
        if P0 is not None:
            self.P_filtered_prev = P0.clone().detach()

    def _get_sigma_points(self, x_mean, P):
        """
        Vypočítá sigma body na základě průměru a kovariance.
        """
        # Přidání malého jitteru pro numerickou stabilitu Choleského rozkladu
        jitter = torch.eye(self.L, device=self.device) * 1e-6
        P_stable = P + jitter
        
        try:
            S = torch.linalg.cholesky(P_stable)
        except torch.linalg.LinAlgError:
            print("Cholesky selhal! Používám vlastní odmocninu matice.")
            # Záložní metoda, pokud Cholesky selže
            eigvals, eigvecs = torch.linalg.eigh(P_stable)
            S = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T

        scale = torch.sqrt(self.L + self.lambda_)
        
        # Vytvoření sigma bodů
        sigma_points = x_mean.repeat(1, 2 * self.L + 1)
        sigma_points[:, 1:self.L + 1] += scale * S
        sigma_points[:, self.L + 1:] -= scale * S
        
        return sigma_points

    def predict_step(self, x_filtered, P_filtered):
        sigma_points = self._get_sigma_points(x_filtered, P_filtered)
        
        # Propagace sigma bodů přes nelineární funkci f (PLNĚ VEKTORIZOVANĚ)
        # Tvar sigma_points: [L, 2L+1]. Tvar pro f: [2L+1, L]
        propagated_points = self.f(sigma_points.T).T
        
        # Výpočet predikovaného stavu a kovariance
        x_predict = (propagated_points @ self.Wm).unsqueeze(1)
        diff = propagated_points - x_predict
        P_predict = (diff * self.Wc) @ diff.T + self.Q

        return x_predict, P_predict

    def update_step(self, x_predict, P_predict, y_t):
        sigma_points = self._get_sigma_points(x_predict, P_predict)

        # Transformace sigma bodů do měřícího prostoru (PLNĚ VEKTORIZOVANĚ)
        # Tvar sigma_points: [L, 2L+1]. Tvar pro h: [2L+1, L]
        measurement_points = self.h(sigma_points.T).T
        
        # Predikce měření
        y_hat = (measurement_points @ self.Wm).unsqueeze(1)
        
        # Výpočet kovariancí
        y_diff = measurement_points - y_hat
        x_diff = sigma_points - x_predict
        
        P_yy = (y_diff * self.Wc) @ y_diff.T + self.R
        P_xy = (x_diff * self.Wc) @ y_diff.T
        
        # Výpočet Kalmanova zisku a update stavu
        innovation = y_t.reshape(self.obs_dim, 1) - y_hat
        K = P_xy @ torch.linalg.inv(P_yy)
        
        x_filtered = x_predict + K @ innovation
        P_filtered = P_predict - K @ P_yy @ K.T
        
        return x_filtered, P_filtered, K, innovation

    def step(self, y_t):
        """
        Provede jeden kompletní krok filtrace (predict + update). Stejné jako v EKF.
        """
        x_predict, P_predict = self.predict_step(self.x_filtered_prev, self.P_filtered_prev)
        x_filtered, P_filtered, _, _ = self.update_step(x_predict, P_predict, y_t)
        
        self.x_filtered_prev = x_filtered
        self.P_filtered_prev = P_filtered

        return x_filtered, P_filtered

    def process_sequence(self, y_seq, Ex0=None, P0=None):
        """
        Zpracuje celou sekvenci měření `y_seq`. Stejné jako v EKF.
        """
        x_est = Ex0.clone().detach().reshape(self.state_dim, 1) if Ex0 is not None else self.x_filtered_prev.clone()
        P_est = P0.clone().detach() if P0 is not None else self.P_filtered_prev.clone()

        seq_len = y_seq.shape[0]

        # Inicializace tenzorů pro historii
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
            x_est, P_est, K, innovation = self.update_step(x_predict, P_predict, y_seq[t])

            # Uložení výsledků
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