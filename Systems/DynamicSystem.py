import torch

class DynamicSystem:
    def __init__(self, state_dim, obs_dim, Q, R, Ex0, P0, f=None, h=None, F=None, H=None, device=None, noise_type='gaussian', gmm_params=None):
        if device is None:
            self.device = Q.device
        else:
            self.device = device
            
        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.Q, self.R, self.Ex0, self.P0 = Q.to(self.device), R.to(self.device), Ex0.to(self.device), P0.to(self.device)

        if f is not None and F is not None: raise ValueError("Zadejte buď `f` nebo `F`.")
        if h is not None and H is not None: raise ValueError("Zadejte buď `h` nebo `H`.")
        
        # Uložíme si lambda funkce přímo
        self._f_func = f
        self._h_func = h
        
        self.F = F.to(self.device) if F is not None else None
        self.H = H.to(self.device) if H is not None else None
        
        self.is_linear_f = (self.F is not None)
        self.is_linear_h = (self.H is not None)

        if not self.is_linear_f and self._f_func is None: raise ValueError("Chybí `f` nebo `F`.")
        if not self.is_linear_h and self._h_func is None: raise ValueError("Chybí `h` nebo `H`.")

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

    def f(self, x_in):
        # Připravíme vstup, aby byl vždy dávka ([B, D_state])
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]

        if self.is_linear_f:
            # Plně vektorizovaná operace
            F_batch = self.F.unsqueeze(0).expand(batch_size, -1, -1)
            # .unsqueeze(-1) převede [B, D] na [B, D, 1] pro maticové násobení
            return torch.bmm(F_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        else:
            # Aplikujeme nelineární funkci přímo na celý dávkový tenzor.
            # Lambda funkce musí být napsána tak, aby to podporovala (např. torch.sin).
            return self._f_func(x_batch)

    def h(self, x_in):
        # Připravíme vstup, aby byl vždy dávka ([B, D_state])
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]

        if self.is_linear_h:
            # Plně vektorizovaná operace
            H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(H_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        else:
            # Aplikujeme nelineární funkci přímo na celý dávkový tenzor.
            return self._h_func(x_batch)

    def get_initial_state(self):
        if self.L_p0 is None: raise RuntimeError("Choleského rozklad P0 selhal.")
        z = torch.randn(self.state_dim, 1, device=self.device)
        # Squeeze() je bezpečnější než squeeze(-1), odstraní všechny dimenze velikosti 1
        return (self.Ex0.unsqueeze(1) + self.L_p0 @ z).squeeze()

    def get_initial_state_input(self):
        if self.L_p0 is None: raise RuntimeError("Choleského rozklad P0 selhal.")
        z = torch.randn(self.state_dim, 1, device=self.device)
        return (self.Ex0.unsqueeze(1) + self.L_p0 @ z).squeeze()

    def get_deterministic_initial_state(self):
        return self.Ex0.clone().squeeze()

    def step(self, x_prev_in, u_in=None):
        """Provede jeden krok dynamiky. Vstup i výstup jsou 2D: [B, D]."""
        x_prev_batch = self._prepare_input(x_prev_in)
        batch_size = x_prev_batch.shape[0]
        w = self._generate_noise(self.Q, self.L_q, batch_size)    
        if u_in is None:    
            return self.f(x_prev_batch) + w.squeeze(-1)
        else:
            u_batch = self._prepare_input(u_in)
            return self.f(x_prev_batch) + u_batch + w.squeeze(-1)
            

    def measure(self, x_in):
        """Provede měření stavu. Vstup i výstup jsou 2D: [B, D] a [B, O]."""
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]
        
        v = self._generate_noise(self.R, self.L_r, batch_size)
        y_noiseless = self.h(x_batch)
        return y_noiseless + v.squeeze(-1)
    
    def _generate_noise(self, cov_matrix, cholesky_factor, num_samples):
        """Pomocná funkce pro generování šumu podle self.noise_type."""
        dim = cov_matrix.shape[0]
        
        if self.noise_type == 'gaussian':
            if cholesky_factor is None: raise RuntimeError("Choleského rozklad selhal.")
            return cholesky_factor @ torch.randn(num_samples, dim, 1, device=self.device)
        
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