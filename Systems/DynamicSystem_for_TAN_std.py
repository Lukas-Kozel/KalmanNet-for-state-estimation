import torch

class DynamicSystemTAN_std:
    def __init__(self, state_dim, obs_dim, Q, R, Ex0, P0, x_axis_unique, y_axis_unique, f=None, h=None, F=None, H=None, device=None, noise_type='gaussian', gmm_params=None):
        if device is None:
            self.device = Q.device
        else:
            self.device = device
            
        self.min_x = x_axis_unique.min()
        self.max_x = x_axis_unique.max()
        self.min_y = y_axis_unique.min()
        self.max_y = y_axis_unique.max()
        print(f"INFO: DynamicSystemTAN inicializován s hranicemi mapy:")
        print(f"  X: [{self.min_x:.2f}, {self.max_x:.2f}]")
        print(f"  Y: [{self.min_y:.2f}, {self.max_y:.2f}]")
        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.Q, self.R, self.Ex0, self.P0 = Q.to(self.device), R.to(self.device), Ex0.to(self.device), P0.to(self.device)

        if f is not None and F is not None: raise ValueError("Zadejte buď `f` nebo `F`.")
        if h is not None and H is not None: raise ValueError("Zadejte buď `h` nebo `H`.")
        
        # --- ZMĚNA ZDE: Uložíme si fyzikální lambda funkce ---
        self._f_phys_func = f
        self._h_phys_func = h
        
        # --- ZMĚNA ZDE: Přejmenování fyzikálních matic ---
        self.F_phys = F.to(self.device) if F is not None else None
        self.H_phys = H.to(self.device) if H is not None else None
        
        self.is_linear_f = (self.F_phys is not None)
        self.is_linear_h = (self.H_phys is not None)

        if not self.is_linear_f and self._f_phys_func is None: raise ValueError("Chybí `f` nebo `F`.")
        if not self.is_linear_h and self._h_phys_func is None: raise ValueError("Chybí `h` nebo `H`.")

        # --- PŘIDÁNO: Atributy pro statistiky Z-score ---
        # Inicializujeme je jako identity (průměr 0, odchylka 1)
        # Tyto budou přepsány zvenčí po výpočtu z trénovacích dat
        self.x_mean = torch.zeros(state_dim, device=self.device)
        self.x_std = torch.ones(state_dim, device=self.device)
        self.y_mean = torch.zeros(obs_dim, device=self.device)
        self.y_std = torch.ones(obs_dim, device=self.device)
        # --- KONEC PŘIDÁNÍ ---

        self.noise_type = noise_type
        if self.noise_type == 'gmm':
            if gmm_params is None:
                # Defaultní hodnoty, pokud nejsou zadány
                self.gmm_params = {'prob_outlier': 0.05, 'outlier_scale': 10}
                print(f"INFO: Používám defaultní GMM parametry: {self.gmm_params}")
            else:
                self.gmm_params = gmm_params
        
        # Choleského rozklad pro Gaussovský případ
        try:
            self.L_q = torch.linalg.cholesky(self.Q)
            self.L_r = torch.linalg.cholesky(self.R)
            self.L_p0 = torch.linalg.cholesky(self.P0)
        except torch.linalg.LinAlgError as e:
            print(f"Varování při Choleského rozkladu: {e}")
            self.L_q, self.L_r, self.L_p0 = None, None, None

    def _prepare_input(self, x_in):
        """Pomocná funkce pro robustní zpracování vstupu."""
        # Pokud je vstup skalár (0D), uděláme z něj 1D vektor
        if x_in.dim() == 0:
            x_in = x_in.unsqueeze(0)
        # Pokud je vstup 1D vektor, uděláme z něj dávku o velikosti 1
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)
        return x_in

    # --- ZMĚNA ZDE: Přejmenování na f_phys ---
    def f_phys(self, x_in):
        """FYZIKÁLNÍ stavová funkce."""
        # Připravíme vstup, aby byl vždy dávka ([B, D_state])
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]

        if self.is_linear_f:
            # Plně vektorizovaná operace
            F_batch = self.F_phys.unsqueeze(0).expand(batch_size, -1, -1)
            # .unsqueeze(-1) převede [B, D] na [B, D, 1] pro maticové násobení
            return torch.bmm(F_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        else:
            # Aplikujeme nelineární funkci přímo na celý dávkový tenzor.
            return self._f_phys_func(x_batch)

    # --- ZMĚNA ZDE: Přejmenování na h_phys ---
    def h_phys(self, x_in):
        """FYZIKÁLNÍ funkce pozorování."""
        # Připravíme vstup, aby byl vždy dávka ([B, D_state])
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]

        if self.is_linear_h:
            # Plně vektorizovaná operace
            H_batch = self.H_phys.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(H_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        else:
            # Aplikujeme nelineární funkci přímo na celý dávkový tenzor.
            return self._h_phys_func(x_batch)

    # --- PŘIDÁNO: Nová veřejná funkce f (Z-score wrapper) ---
    def f(self, x_norm):
        """
        NORMALIZOVANÝ wrapper pro f.
        Vstup: x_norm (normalizovaný stav)
        Výstup: x_pred_norm (normalizovaný predikovaný stav)
        """
        # 1. De-normalizovat na fyzikální stav
        x_phys = (x_norm * self.x_std) + self.x_mean
        
        # 2. Zavolat původní fyzikální model
        x_pred_phys = self.f_phys(x_phys)
        
        # 3. Normalizovat výstup zpět
        x_pred_norm = (x_pred_phys - self.x_mean) / self.x_std
        return x_pred_norm

    # --- PŘIDÁNO: Nová veřejná funkce h (Z-score wrapper) ---
    def h(self, x_norm):
        """
        NORMALIZOVANÝ wrapper pro h.
        Vstup: x_norm (normalizovaný stav)
        Výstup: y_pred_norm (normalizované pozorování)
        """
        # 1. De-normalizovat na fyzikální stav
        x_phys = (x_norm * self.x_std) + self.x_mean
        
        # 2. Zavolat původní fyzikální model (který volá terMap)
        y_pred_phys = self.h_phys(x_phys)
        
        # 3. Normalizovat výstup (pozorování) zpět
        y_pred_norm = (y_pred_phys - self.y_mean) / self.y_std
        return y_pred_norm
    # --- KONEC PŘIDÁNÍ ---

    def get_initial_state(self):
        # Tato funkce vrací FYZIKÁLNÍ počáteční stav
        if self.L_p0 is None: raise RuntimeError("Choleského rozklad P0 selhal.")
        z = torch.randn(self.state_dim, 1, device=self.device)
        return (self.Ex0.unsqueeze(1) + self.L_p0 @ z).squeeze()

    def get_initial_state_input(self):
        # Tato funkce vrací FYZIKÁLNÍ počáteční stav
        if self.L_p0 is None: raise RuntimeError("Choleského rozklad P0 selhal.")
        z = torch.randn(self.state_dim, 1, device=self.device)
        return (self.Ex0.unsqueeze(1) + self.L_p0 @ z).squeeze()

    def get_deterministic_initial_state(self):
        # Tato funkce vrací FYZIKÁLNÍ počáteční stav
        return self.Ex0.clone().squeeze()

    def step(self, x_prev_in, u_in=None):
        """Provede jeden krok dynamiky (pro generování dat). Vstup i výstup jsou FYZIKÁLNÍ."""
        x_prev_batch = self._prepare_input(x_prev_in)
        batch_size = x_prev_batch.shape[0]
        w = self._generate_noise(self.Q, self.L_q, batch_size)    
        if u_in is None:    
            # --- ZMĚNA ZDE: Volá f_phys ---
            return self.f_phys(x_prev_batch) + w.squeeze(-1)
        else:
            u_batch = self._prepare_input(u_in)
            # --- ZMĚNA ZDE: Volá f_phys ---
            return self.f_phys(x_prev_batch) + u_batch + w.squeeze(-1)
            

    def measure(self, x_in):
        """Provede měření stavu (pro generování dat). Vstup i výstup jsou FYZIKÁLNÍ."""
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]
        
        v = self._generate_noise(self.R, self.L_r, batch_size)
        # --- ZMĚNA ZDE: Volá h_phys ---
        y_noiseless = self.h_phys(x_batch)
        return y_noiseless + v.squeeze(-1)
    
    def _generate_noise(self, cov_matrix, cholesky_factor, num_samples):
        """Pomocná funkce pro generování šumu podle self.noise_type."""
        dim = cov_matrix.shape[0]
        
        if self.noise_type == 'gaussian':
            if cholesky_factor is None: raise RuntimeError("Choleského rozklad selhal.")
            # --- O oprava: přidána dimenze dávky ---
            return (cholesky_factor @ torch.randn(num_samples, dim, 1, device=self.device))
        
        elif self.noise_type == 'gmm':
            # Předpokládáme diagonální kovarianční matici pro jednoduchost GMM
            std_normal = torch.sqrt(torch.diag(cov_matrix))
            std_outlier = std_normal * self.gmm_params['outlier_scale']
            
            is_outlier = torch.rand(num_samples, dim, 1, device=self.device) < self.gmm_params['prob_outlier']
            
            noise_normal = torch.randn(num_samples, dim, 1, device=self.device) * std_normal.view(1, -1, 1)
            noise_outlier = torch.randn(num_samples, dim, 1, device=self.device) * std_outlier.view(1, -1, 1)
            
            return torch.where(is_outlier, noise_outlier, noise_normal)
        
        else:
            raise ValueError(f"Neznámý typ šumu: '{self.noise_type}'")