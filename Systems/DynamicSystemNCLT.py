import torch

class DynamicSystemNCLT:
    def __init__(self, state_dim=6, obs_dim=2, dt=1.0, Q=None, R=None, Ex0=None, P0=None, f=None, h=None, device=None, noise_type='gaussian', gmm_params=None):
        """
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
        
        if Q is None:
            self.Q = torch.eye(state_dim).to(self.device) * 0.1
        else:
            self.Q = Q.to(self.device)

        if R is None:
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
        self.is_linear_h = True
        self.H = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ], device=self.device)

        self.is_linear_f = False
        
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
        
        self._compute_cholesky()

    def _compute_cholesky(self):
        try:
            self.L_q = torch.linalg.cholesky(self.Q)
            self.L_r = torch.linalg.cholesky(self.R)
            self.L_p0 = torch.linalg.cholesky(self.P0)
        except torch.linalg.LinAlgError as e:
            self.L_q, self.L_r, self.L_p0 = None, None, None

    def _prepare_input(self, x_in):
        if x_in.dim() == 0: x_in = x_in.unsqueeze(0).unsqueeze(0)
        if x_in.dim() == 1: x_in = x_in.unsqueeze(0)
        return x_in

    def _f_paper_dynamics(self, x, u):
        if u is None:
            return x 

        v_l = u[:, 0]
        v_r = u[:, 1]
        theta_imu = u[:, 2]
        omega_imu = u[:, 3]

        v_c = 0.5 * (v_r + v_l)

        px_prev = x[:, 0]
        py_prev = x[:, 1]
        
        px_new = px_prev + v_c * torch.cos(theta_imu) * self.dt
        py_new = py_prev + v_c * torch.sin(theta_imu) * self.dt
        
        vx_new = v_c * torch.cos(theta_imu)
        vy_new = v_c * torch.sin(theta_imu)
        
        theta_new = theta_imu
        omega_new = omega_imu

        return torch.stack((px_new, py_new, vx_new, vy_new, theta_new, omega_new), dim=1)

    def _h_gps_measurement(self, x):
        return x[:, :2]

    def f(self, x_in, u_in=None):
        x_batch = self._prepare_input(x_in)
        
        if u_in is not None:
            u_batch = self._prepare_input(u_in)
            return self._f_func(x_batch, u_batch)
        else:
            try:
                return self._f_func(x_batch, None)
            except TypeError:
                return self._f_func(x_batch)

    def h(self, x_in):
        x_batch = self._prepare_input(x_in)
        return self._h_func(x_batch)

    def step(self, x_prev_in, u_in=None):
        x_prev_batch = self._prepare_input(x_prev_in)
        batch_size = x_prev_batch.shape[0]
        
        x_next_det = self.f(x_prev_batch, u_in)
        
        w = self._generate_noise(self.Q, self.L_q, batch_size).squeeze(-1)
        
        return x_next_det + w

    def measure(self, x_in):
        x_batch = self._prepare_input(x_in)
        batch_size = x_batch.shape[0]
        
        y_det = self.h(x_batch)
        
        v = self._generate_noise(self.R, self.L_r, batch_size).squeeze(-1)
        
        return y_det + v


    def get_initial_state(self):
        if self.L_p0 is None: raise RuntimeError("Cholesky P0 failed.")
        z = torch.randn(self.state_dim, device=self.device)
        return self.Ex0.squeeze() + self.L_p0 @ z

    def get_initial_state_batch(self, batch_size):
        if self.L_p0 is None: raise RuntimeError("Cholesky P0 failed.")
        z = torch.randn(batch_size, self.state_dim, device=self.device)
        Ex0_batch = self.Ex0.squeeze().unsqueeze(0)
        return Ex0_batch + z @ self.L_p0.T

    def _generate_noise(self, cov, L, batch_size):
        dim = cov.shape[0]
        if self.noise_type == 'gaussian':
            if L is None: raise RuntimeError("Cholesky failed.")
            return L @ torch.randn(batch_size, dim, 1, device=self.device)
        elif self.noise_type == 'gmm':
            std_normal = torch.sqrt(torch.diag(cov))
            std_outlier = std_normal * self.gmm_params['outlier_scale']
            is_outlier = torch.rand(batch_size, dim, 1, device=self.device) < self.gmm_params['prob_outlier']
            n_norm = torch.randn(batch_size, dim, 1, device=self.device) * std_normal.view(1, -1, 1)
            n_out = torch.randn(batch_size, dim, 1, device=self.device) * std_outlier.view(1, -1, 1)
            return torch.where(is_outlier, n_out, n_norm)
        else:
            raise ValueError(f"Unknowń noise_type: {self.noise_type}")