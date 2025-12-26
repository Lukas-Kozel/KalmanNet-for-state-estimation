import torch

class UnscentedKalmanFilterNCLT:
    def __init__(self, system_model, alpha=0.1, beta=2.0, kappa=0.0):
        """
        Robustní UKF pro NCLT s podporou vstupu u (control) a NaN měření.
        
        Args:
            alpha: Rozptyl sigma bodů (0.1 - 1.0 je ok pro float32)
            beta: Parametr pro prior (2.0 je optimální pro Gaussovské rozdělení)
            kappa: Sekundární škálovací parametr (obvykle 0 nebo 3-n)
        """
        self.device = system_model.Q.device
        
        self.system = system_model
        # f: (x, u) -> x_next (Musí umět zpracovat batch sigma bodů!)
        self.f = system_model.f
        # h: (x) -> y (Musí umět zpracovat batch sigma bodů!)
        self.h = system_model.h
        
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]
        
        # --- Nastavení parametrů Sigma bodů ---
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lam = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.n_sigma_points = 2 * self.state_dim + 1
        
        # Váhy (W_mean a W_cov)
        self.Wm = torch.full((self.n_sigma_points,), 0.5 / (self.state_dim + self.lam), device=self.device)
        self.Wc = self.Wm.clone()
        
        # První váha je speciální
        self.Wm[0] = self.lam / (self.state_dim + self.lam)
        self.Wc[0] = self.lam / (self.state_dim + self.lam) + (1 - self.alpha**2 + self.beta)

        # Interní stavy
        self.x_predict_current = None
        self.P_predict_current = None
        
        # Inicializace
        self.reset(system_model.Ex0, system_model.P0)

    def reset(self, Ex0, P0):
        self.x_predict_current = Ex0.clone().detach().reshape(self.state_dim, 1)
        self.P_predict_current = P0.clone().detach()

    def _ensure_positive_definite(self, P):
        """Vynutí symetrii a PD vlastnost matice P."""
        P = 0.5 * (P + P.transpose(-1, -2))
        jitter = 1e-6 * torch.eye(self.state_dim, device=self.device)
        return P + jitter

    def _get_sigma_points(self, x, P):
        """Generování sigma bodů pomocí Choleského rozkladu."""
        # Pojistka pro stabilitu
        P = self._ensure_positive_definite(P)
        
        try:
            L = torch.linalg.cholesky(P)
        except torch.linalg.LinAlgError:
            # Fallback: Eigendecomposition pokud Cholesky selže
            eigvals, eigvecs = torch.linalg.eigh(P)
            eigvals = torch.clamp(eigvals, min=1e-8)
            L = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

        factor = torch.sqrt(torch.tensor(self.state_dim + self.lam, device=self.device))
        
        sigma_points = torch.zeros((self.n_sigma_points, self.state_dim), device=self.device)
        sigma_points[0] = x.squeeze()
        
        for i in range(self.state_dim):
            sigma_points[i + 1] = x.squeeze() + factor * L[:, i]
            sigma_points[i + 1 + self.state_dim] = x.squeeze() - factor * L[:, i]
            
        return sigma_points # [2n+1, state_dim]

    def predict_step(self, x_curr, P_curr, u_curr=None):
        """
        Krok predikce (Time Update).
        """
        # 1. Generování Sigma bodů
        sigma_points = self._get_sigma_points(x_curr, P_curr)
        
        # 2. Propagace bodů přes nelineární funkci f(x, u)
        if u_curr is not None:
            # Vstup u [Input_dim] -> [1, Input_dim] -> repeat -> [2n+1, Input_dim]
            # Tím zajistíme, že f() dostane správný tvar pro batch processing
            u_batch = u_curr.reshape(1, -1).repeat(self.n_sigma_points, 1)
            propagated_sigmas = self.f(sigma_points, u_batch)
        else:
            propagated_sigmas = self.f(sigma_points, None)
            
        # 3. Výpočet apriorního průměru x_pred
        # Vážený součet sigma bodů
        x_pred = torch.sum(self.Wm.unsqueeze(1) * propagated_sigmas, dim=0).reshape(self.state_dim, 1)
        
        # 4. Výpočet apriorní kovariance P_pred
        diff = propagated_sigmas - x_pred.T
        
        # POZNÁMKA: Zde by teoreticky mělo být ošetření úhlů (wrap_to_pi) pro rozdíly stavů,
        # pokud sys_model obsahuje úhly (theta). Standardní odčítání může udělat chybu kolem +/- PI.
        # Pro NCLT to obvykle projde, pokud se auto netočí extrémně rychle v jednom kroku.
        
        P_pred = diff.T @ (torch.diag(self.Wc) @ diff) + self.Q
        P_pred = self._ensure_positive_definite(P_pred)
        
        return x_pred, P_pred

    def update_step(self, x_pred, P_pred, y_meas):
        """
        Krok korekce (Measurement Update).
        Obsahuje NaN handling identický s EKF.
        """
        # === HANDLING NaN (Výpadek GPS) ===
        if torch.any(torch.isnan(y_meas)):
            # Vracíme predikci beze změny (Skip Update)
            # Nulová inovace a gain pro logování
            return x_pred, P_pred, torch.zeros(self.state_dim, self.obs_dim, device=self.device), torch.zeros(self.obs_dim, device=self.device)

        y_meas = y_meas.reshape(self.obs_dim, 1)

        # 1. Znovu generujeme sigma body (z predikce)
        # Některé varianty UKF používají staré body, ale standard je generovat nové z P_pred
        sigma_points = self._get_sigma_points(x_pred, P_pred)
        
        # 2. Transformace sigma bodů do prostoru měření h(x)
        meas_sigmas = self.h(sigma_points)
        
        # 3. Odhad měření y_hat (Vážený průměr)
        y_hat = torch.sum(self.Wm.unsqueeze(1) * meas_sigmas, dim=0).reshape(self.obs_dim, 1)
        
        # 4. Kovariance měření (S) a křížová kovariance (P_xy)
        y_diff = meas_sigmas - y_hat.T
        x_diff = sigma_points - x_pred.T
        
        S = y_diff.T @ (torch.diag(self.Wc) @ y_diff) + self.R
        P_xy = x_diff.T @ (torch.diag(self.Wc) @ y_diff)
        
        # 5. Výpočet Kalman Gain a Update
        # Používáme pinv pro stabilitu
        K = P_xy @ torch.linalg.pinv(S)
        
        innovation = y_meas - y_hat
        x_new = x_pred + K @ innovation
        
        # P_new = P_pred - K * S * K'
        P_new = P_pred - K @ S @ K.T
        P_new = self._ensure_positive_definite(P_new)
        
        return x_new, P_new, K, innovation

    def process_sequence(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Dávkové zpracování celé trajektorie.
        Struktura: Loop { Update -> Save -> Predict }
        """
        seq_len = y_seq.shape[0]
        
        x_hist = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        # Inicializace
        # Pokud nebylo zadáno Ex0, použijeme interní stav z minula (nebo reset)
        if Ex0 is not None:
            x_curr = Ex0.clone().detach().reshape(self.state_dim, 1)
            P_curr = P0.clone().detach()
        else:
            x_curr = self.x_predict_current
            P_curr = self.P_predict_current
        
        for k in range(seq_len):
            # --- 1. UPDATE STEP (Korekce) ---
            # Zkusíme opravit aktuální stav měřením (pokud není NaN)
            x_upd, P_upd, _, _ = self.update_step(x_curr, P_curr, y_seq[k])
            
            # Uložíme výsledek (Posterior)
            x_hist[k] = x_upd.squeeze()
            P_hist[k] = P_upd
            
            # --- 2. PREDICT STEP (Posun do budoucna) ---
            if k < seq_len - 1:
                # Načtení vstupu u pro tento krok
                u_k = u_seq[k] if u_seq is not None else None
                
                # Predikce x_{k+1|k}
                x_curr, P_curr = self.predict_step(x_upd, P_upd, u_k)
            
        return {'x_filtered': x_hist, 'P_filtered': P_hist}