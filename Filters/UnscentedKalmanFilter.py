import torch

class UnscentedKalmanFilter:
    def __init__(self, system_model):
        """
        Inicializace zjednodušeného, ale robustního a vektorizovaného UKF.
        Tato verze nepoužívá parametry alpha, beta, kappa.
        """
        self.device = system_model.Q.device
        
        self.f = system_model.f
        self.h = system_model.h
        
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]
        
        # --- ZJEDNODUŠENÍ: Odstranění alpha, beta, kappa ---
        # Počet sigma bodů
        self.n_sigma_points = 2 * self.state_dim + 1
        # Jednoduché, rovnoměrné váhy pro všechny body
        self.weights = torch.full((self.n_sigma_points,), 1.0 / self.n_sigma_points, device=self.device)

    def _get_sigma_points(self, x_mean, P):
        """
        Vypočítá sigma body na základě průměru a kovariance.
        """
        # --- ZACHOVÁNA ROBUSTNOST: Ochrana proti selhání Choleského rozkladu ---
        try:
            # Přidání malého jitteru pro numerickou stabilitu
            jitter = torch.eye(self.state_dim, device=self.device) * 1e-6
            S = torch.linalg.cholesky(P + jitter)
        except torch.linalg.LinAlgError:
            print("Cholesky selhal! Používám vlastní odmocninu matice.")
            eigvals, eigvecs = torch.linalg.eigh(P)
            S = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=0))) @ eigvecs.T

        # --- ZJEDNODUŠENÍ: Jednoduchý scaling faktor ---
        scale = torch.sqrt(torch.tensor(self.state_dim, device=self.device, dtype=torch.float32))
        
        # Vytvoření sigma bodů (x_mean musí být sloupcový vektor [state_dim, 1])
        sigma_points = x_mean.repeat(1, self.n_sigma_points)
        sigma_points[:, 1:self.state_dim + 1] += scale * S
        sigma_points[:, self.state_dim + 1:] -= scale * S
        
        return sigma_points

    def predict_step(self, x_filtered, P_filtered):
        sigma_points = self._get_sigma_points(x_filtered, P_filtered)
        
        # --- OPRAVA: Plně vektorizovaná propagace bodů ---
        propagated_points = self.f(sigma_points.T).T
        
        x_predict = (propagated_points @ self.weights).unsqueeze(1)
        diff = propagated_points - x_predict
        P_predict = (diff * self.weights) @ diff.T + self.Q

        return x_predict, P_predict

    def update_step(self, x_predict, P_predict, y_t):
        sigma_points = self._get_sigma_points(x_predict, P_predict)

        # --- OPRAVA: Plně vektorizovaná transformace bodů (bez for-smyčky) ---
        measurement_points = self.h(sigma_points.T).T
        
        y_hat = (measurement_points @ self.weights).unsqueeze(1)
        
        y_diff = measurement_points - y_hat
        x_diff = sigma_points - x_predict
        
        P_yy = (y_diff * self.weights) @ y_diff.T + self.R
        P_xy = (x_diff * self.weights) @ y_diff.T
        
        innovation = y_t.reshape(self.obs_dim, 1) - y_hat
        K = P_xy @ torch.linalg.inv(P_yy)
        
        x_filtered = x_predict + K @ innovation
        P_filtered = P_predict - K @ P_yy @ K.T
        
        return x_filtered, P_filtered, K, innovation

    def process_sequence(self, y_seq, Ex0=None, P0=None):
        """
        Zpracuje celou sekvenci měření.
        Tato metoda je nyní plně kompatibilní s vaším vyhodnocovacím skriptem.
        """
        x_est = Ex0.clone().detach().reshape(self.state_dim, 1)
        P_est = P0.clone().detach()

        seq_len = y_seq.shape[0]

        # --- OPRAVA: Správná inicializace a ukládání historie ---
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)

        # Na začátku (čas t=0) je "filtrovaný" stav roven počátečnímu stavu
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est

        # Pro t=1...N-1 provádíme predict a update
        # Poznámka: v EKF a UKF se typicky predikuje stav v čase 't' z 't-1' a pak se updatuje s měřením 't'
        # Vaše smyčka v eval skriptu ale počítá odhady pro y[0]...y[N-1], proto je tato implementace správná
        for t in range(seq_len):
            # 1. Predict
            x_predict, P_predict = self.predict_step(x_est, P_est)

            # 2. Update
            x_est, P_est, _, _ = self.update_step(x_predict, P_predict, y_seq[t])

            # Uložení výsledků
            x_filtered_history[t] = x_est.squeeze()
            P_filtered_history[t] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
        }