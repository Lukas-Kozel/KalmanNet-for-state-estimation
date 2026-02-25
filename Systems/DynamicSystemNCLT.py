import torch

class DynamicSystemNCLT:
    def __init__(self, state_dim=6, obs_dim=2, dt=1.0, Q=None, R=None, Ex0=None, P0=None, f=None, h=None, device=None, noise_type='gaussian', gmm_params=None):
        """
        Dynamický systém pro NCLT Navigaci podle článku:
        'Practical Implementation of KalmanNet for Accurate Data Fusion'.
        
        State: [px, py, vx, vy, theta, omega] (6D)
        Input: [v_left, v_right, theta_imu, omega_imu] (4D)
        Meas:  [gps_x, gps_y] (2D)
        """
        if device is None:
            self.device = Q.device if Q is not None else torch.device('cpu')
        else:
            self.device = device
            
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        
        # Defaultní matice, pokud nejsou zadány (dle článku/datasetu)
        if Q is None:
            # Q pro 6 stavů. Pozice a rychlost mají malý šum, úhel větší.
            # Hodnoty jsou placeholder, v notebooku je definuješ přesněji.
            self.Q = torch.eye(state_dim).to(self.device) * 0.1
        else:
            self.Q = Q.to(self.device)

        if R is None:
            # R pro GPS (cca 1-5m chyba)
            self.R = torch.eye(obs_dim).to(self.device) * 1.0
        else:
            self.R = R.to(self.device)

        if Ex0 is None:
            self.Ex0 = torch.zeros(state_dim, 1).to(self.device)
        else:
            self.Ex0 = Ex0.to(self.device)

        if P0 is None:
            self.P0 = torch.eye(state_dim).to(self.device)
        else:
            self.P0 = P0.to(self.device)
        # --- DEFINICE LINEARITY PRO RKN/FILTRY ---
        self.is_linear_h = True
        # Matice měření H vybere jen px (index 0) a py (index 1) ze 6D stavu
        self.H = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ], device=self.device)

        # Funkce f je vzhledem ke stavu x prakticky vzato lineární (bere jen px, py a zbytek přepisuje),
        # ale pro jistotu (a protože závisí na u_in) můžeme RKN donutit počítat Jakobián.
        self.is_linear_f = False
        
        # --- DEFINICE MODELU ---
        # Pokud uživatel nezadá f/h, použijeme model z článku (rovnice 5 a 6)
        if f is None:
            self._f_func = self._f_paper_dynamics
        else:
            self._f_func = f
            
        if h is None:
            self._h_func = self._h_gps_measurement
        else:
            self._h_func = h

        self.noise_type = noise_type
        if self.noise_type == 'gmm':
            if gmm_params is None:
                self.gmm_params = {'prob_outlier': 0.05, 'outlier_scale': 10}
            else:
                self.gmm_params = gmm_params
        
        # Choleského rozklad pro generování šumu
        self._compute_cholesky()

    def _compute_cholesky(self):
        try:
            self.L_q = torch.linalg.cholesky(self.Q)
            self.L_r = torch.linalg.cholesky(self.R)
            self.L_p0 = torch.linalg.cholesky(self.P0)
        except torch.linalg.LinAlgError as e:
            print(f"⚠️ Chyba Choleského rozkladu (matice nejsou PD): {e}")
            self.L_q, self.L_r, self.L_p0 = None, None, None

    def _prepare_input(self, x_in):
        """Zajistí, že vstup je [Batch, Dim]"""
        if x_in.dim() == 0: x_in = x_in.unsqueeze(0).unsqueeze(0)
        if x_in.dim() == 1: x_in = x_in.unsqueeze(0)
        return x_in

    # =========================================================================
    # FYZIKÁLNÍ MODELY (Dle článku)
    # =========================================================================
    
    def _f_paper_dynamics(self, x, u):
        """
        Přechodová funkce dle rovnice (5) v článku.
        x: [Batch, 6] -> [px, py, vx, vy, theta, omega] (minulý stav)
        u: [Batch, 4] -> [v_left, v_right, theta_imu, omega_imu] (vstup)
        """
        # Pokud nemáme vstup, předpokládáme stání (nebo CV model)
        if u is None:
            return x 

        # 1. Extrakce vstupů z u
        v_l = u[:, 0]
        v_r = u[:, 1]
        theta_imu = u[:, 2] # Orientace přímo z IMU (neintegrace)
        omega_imu = u[:, 3] # Úhlová rychlost z IMU

        # 2. Výpočet rychlosti těžiště vozidla [cite: 725]
        v_c = 0.5 * (v_r + v_l)

        # 3. Extrakce stavů (používáme jen pozici z minula, zbytek se přepíše vstupem)
        px_prev = x[:, 0]
        py_prev = x[:, 1]
        
        # 4. Aplikace dynamiky (Rovnice 5) 
        # Pozice se integruje pomocí rychlosti a úhlu z IMU
        px_new = px_prev + v_c * torch.cos(theta_imu) * self.dt
        py_new = py_prev + v_c * torch.sin(theta_imu) * self.dt
        
        # Rychlosti vx, vy se vypočítají z vc a theta (neintegrují se, jsou deterministické z vstupu)
        vx_new = v_c * torch.cos(theta_imu)
        vy_new = v_c * torch.sin(theta_imu)
        
        # Úhel a omega se berou přímo ze vstupu (IMU je "master" pro orientaci v tomto modelu)
        theta_new = theta_imu
        omega_new = omega_imu

        return torch.stack((px_new, py_new, vx_new, vy_new, theta_new, omega_new), dim=1)

    def _h_gps_measurement(self, x):
        """
        Měření GPS dle rovnice (6).
        Vracíme jen pozici [px, py].
        Matice H je [1 0 0 0 0 0; 0 1 0 0 0 0].
        """
        return x[:, :2] # První dva sloupce

    # =========================================================================
    # HLAVNÍ METODY (f, h, step, measure)
    # =========================================================================

    def f(self, x_in, u_in=None):
        """Aplikuje přechodovou funkci f(x, u)."""
        x_batch = self._prepare_input(x_in)
        
        if u_in is not None:
            u_batch = self._prepare_input(u_in)
            return self._f_func(x_batch, u_batch)
        else:
            # Fallback pro volání bez u (např. při inicializaci, i když pro tento model nedává smysl)
            try:
                return self._f_func(x_batch, None)
            except TypeError:
                return self._f_func(x_batch)

    def h(self, x_in):
        """Aplikuje funkci měření h(x)."""
        x_batch = self._prepare_input(x_in)
        return self._h_func(x_batch)

    def step(self, x_prev_in, u_in=None):
        """
        Jeden krok simulace: x_k = f(x_{k-1}, u_{k-1}) + q_{k-1}
        """
        x_prev_batch = self._prepare_input(x_prev_in)
        batch_size = x_prev_batch.shape[0]
        
        # 1. Deterministická část (Physics)
        x_next_det = self.f(x_prev_batch, u_in)
        
        # 2. Stochastická část (Process Noise)
        # Generujeme šum pro všech 6 stavů
        w = self._generate_noise(self.Q, self.L_q, batch_size).squeeze(-1)
        
        return x_next_det + w

    def measure(self, x_in):
        """
        Simulace měření: y_k = h(x_k) + r_k
        """
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]
        
        # 1. Deterministická část
        y_det = self.h(x_batch)
        
        # 2. Šum měření
        v = self._generate_noise(self.R, self.L_r, batch_size).squeeze(-1)
        
        return y_det + v

    # =========================================================================
    # POMOCNÉ METODY (Initial State, Noise)
    # =========================================================================

    def get_initial_state(self):
        """Vrátí jeden náhodný počáteční stav (1D vektor)."""
        if self.L_p0 is None: raise RuntimeError("Cholesky P0 selhal.")
        z = torch.randn(self.state_dim, device=self.device)
        return self.Ex0.squeeze() + self.L_p0 @ z

    def get_initial_state_batch(self, batch_size):
        """Vrátí dávku náhodných počátečních stavů [B, Dim]."""
        if self.L_p0 is None: raise RuntimeError("Cholesky P0 selhal.")
        z = torch.randn(batch_size, self.state_dim, device=self.device)
        Ex0_batch = self.Ex0.squeeze().unsqueeze(0)
        return Ex0_batch + z @ self.L_p0.T

    def _generate_noise(self, cov, L, batch_size):
        dim = cov.shape[0]
        if self.noise_type == 'gaussian':
            if L is None: raise RuntimeError("Cholesky selhal.")
            return L @ torch.randn(batch_size, dim, 1, device=self.device)
        elif self.noise_type == 'gmm':
            std_normal = torch.sqrt(torch.diag(cov))
            std_outlier = std_normal * self.gmm_params['outlier_scale']
            is_outlier = torch.rand(batch_size, dim, 1, device=self.device) < self.gmm_params['prob_outlier']
            n_norm = torch.randn(batch_size, dim, 1, device=self.device) * std_normal.view(1, -1, 1)
            n_out = torch.randn(batch_size, dim, 1, device=self.device) * std_outlier.view(1, -1, 1)
            return torch.where(is_outlier, n_out, n_norm)
        else:
            raise ValueError(f"Neznámý noise_type: {self.noise_type}")