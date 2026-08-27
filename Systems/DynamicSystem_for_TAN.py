import torch

class DynamicSystemTAN:
    def __init__(self, state_dim, obs_dim, Q, R, Ex0, P0,x_axis_unique,y_axis_unique, f=None, h=None, F=None, H=None, device=None, noise_type='gaussian', gmm_params=None):
        if device is None:
            self.device = Q.device
        else:
            self.device = device
            
        self.min_x = x_axis_unique.min()
        self.max_x = x_axis_unique.max()
        self.min_y = y_axis_unique.min()
        self.max_y = y_axis_unique.max()
        print(f"  X: [{self.min_x:.2f}, {self.max_x:.2f}]")
        print(f"  Y: [{self.min_y:.2f}, {self.max_y:.2f}]")
        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.Q, self.R, self.Ex0, self.P0 = Q.to(self.device), R.to(self.device), Ex0.to(self.device), P0.to(self.device)

        if f is not None and F is not None: raise ValueError("Zadejte buď `f` nebo `F`.")
        if h is not None and H is not None: raise ValueError("Zadejte buď `h` nebo `H`.")
        
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
                self.gmm_params = {'prob_outlier': 0.05, 'outlier_scale': 10}
            else:
                self.gmm_params = gmm_params
        
        try:
            self.L_q = torch.linalg.cholesky(self.Q)
            self.L_r = torch.linalg.cholesky(self.R)
            self.L_p0 = torch.linalg.cholesky(self.P0)
        except torch.linalg.LinAlgError as e:
            self.L_q, self.L_r, self.L_p0 = None, None, None

    def _prepare_input(self, x_in):
        if x_in.dim() == 0:
            x_in = x_in.unsqueeze(0)
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)
        return x_in

    def f(self, x_in):
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]

        if self.is_linear_f:
            F_batch = self.F.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(F_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        else:
            return self._f_func(x_batch)

    def h(self, x_in):
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]

        if self.is_linear_h:
            H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(H_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        else:
            return self._h_func(x_batch)

    def get_initial_state(self):
        if self.L_p0 is None: raise RuntimeError("Choleského rozklad P0 selhal.")
        z = torch.randn(self.state_dim, 1, device=self.device)
        return (self.Ex0.unsqueeze(1) + self.L_p0 @ z).squeeze()

    def get_initial_state_input(self):
        if self.L_p0 is None: raise RuntimeError("Choleského rozklad P0 selhal.")
        z = torch.randn(self.state_dim, 1, device=self.device)
        return (self.Ex0.unsqueeze(1) + self.L_p0 @ z).squeeze()

    def get_deterministic_initial_state(self):
        return self.Ex0.clone().squeeze()

    def step(self, x_prev_in, u_in=None):
        x_prev_batch = self._prepare_input(x_prev_in)
        batch_size = x_prev_batch.shape[0]
        w = self._generate_noise(self.Q, self.L_q, batch_size)    
        if u_in is None:    
            return self.f(x_prev_batch) + w.squeeze(-1)
        else:
            u_batch = self._prepare_input(u_in)
            return self.f(x_prev_batch) + u_batch + w.squeeze(-1)
            

    def measure(self, x_in):
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]
        
        v = self._generate_noise(self.R, self.L_r, batch_size)
        y_noiseless = self.h(x_batch)
        return y_noiseless + v.squeeze(-1)
    
    def _generate_noise(self, cov_matrix, cholesky_factor, num_samples):
        dim = cov_matrix.shape[0]
        
        if self.noise_type == 'gaussian':
            if cholesky_factor is None: raise RuntimeError("Choleského rozklad selhal.")
            return cholesky_factor @ torch.randn(num_samples, dim, 1, device=self.device)
        
        elif self.noise_type == 'gmm':
            std_normal = torch.sqrt(torch.diag(cov_matrix))
            std_outlier = std_normal * self.gmm_params['outlier_scale']
            
            is_outlier = torch.rand(num_samples, dim, 1, device=self.device) < self.gmm_params['prob_outlier']
            
            noise_normal = torch.randn(num_samples, dim, 1, device=self.device) * std_normal.view(1, -1, 1)
            noise_outlier = torch.randn(num_samples, dim, 1, device=self.device) * std_outlier.view(1, -1, 1)
            
            return torch.where(is_outlier, noise_outlier, noise_normal)
        
        else:
            raise ValueError(f"Neznámý typ šumu: '{self.noise_type}'")
        
    def calculate_terrain_gradient(self, x_in):
        with torch.enable_grad():
            
            x_batch = self._prepare_input(x_in).detach().clone()
            x_batch.requires_grad_(True)
            
            y_out = self.h(x_batch)
            z_height = y_out[:, 0]
            
            gradients = torch.autograd.grad(
                outputs=z_height, 
                inputs=x_batch,
                grad_outputs=torch.ones_like(z_height),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            terrain_grad = gradients[:, :2] # [dx, dy]
            
            terrain_grad = torch.clamp(terrain_grad, -5.0, 5.0)
            
        return terrain_grad