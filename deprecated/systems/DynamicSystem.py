import torch

class DynamicSystem:
    def __init__(self, state_dim, obs_dim, Q, R, Ex0, P0, f=None, h=None, F=None, H=None, device=None):
        if device is None: self.device = Q.device
        else: self.device = device
        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.Q, self.R, self.Ex0, self.P0 = Q.to(self.device), R.to(self.device), Ex0.to(self.device), P0.to(self.device)
        if f is not None and F is not None: raise ValueError("Zadejte buď `f` nebo `F`.")
        if h is not None and H is not None: raise ValueError("Zadejte buď `h` nebo `H`.")
        self._f_func, self._h_func = f, h
        self.F = F.to(self.device) if F is not None else None
        self.H = H.to(self.device) if H is not None else None
        self.is_linear_f, self.is_linear_h = (self.F is not None), (self.H is not None)
        if not self.is_linear_f and self._f_func is None: raise ValueError("Chybí `f` nebo `F`.")
        if not self.is_linear_h and self._h_func is None: raise ValueError("Chybí `h` nebo `H`.")
        try:
            self.L_q, self.L_r, self.L_p0 = torch.linalg.cholesky(self.Q), torch.linalg.cholesky(self.R), torch.linalg.cholesky(self.P0)
        except torch.linalg.LinAlgError as e:
            print(f"Varování při Choleského rozkladu: {e}")

    def f(self, x_in):
        if x_in.dim() == 1: x_in = x_in.unsqueeze(0)
        batch_size = x_in.shape[0]
        x_batch = x_in.unsqueeze(-1)
        if self.is_linear_f:
            F_batch = self.F.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(F_batch, x_batch).squeeze(-1)
        else:
            res = [self._f_func(x_batch[i]) for i in range(batch_size)]
            return torch.stack(res).squeeze(-1)

    def h(self, x_in):
        if x_in.dim() == 1: x_in = x_in.unsqueeze(0)
        batch_size = x_in.shape[0]
        x_batch = x_in.unsqueeze(-1)
        if self.is_linear_h:
            H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(H_batch, x_batch).squeeze(-1)
        else:
            res = [self._h_func(x_batch[i]) for i in range(batch_size)]
            return torch.stack(res).squeeze(-1)

    def get_initial_state(self):
        z = torch.randn(self.state_dim, 1, device=self.device)
        return (self.Ex0 + self.L_p0 @ z).squeeze(-1)

    def get_deterministic_initial_state(self):
        return self.Ex0.clone().squeeze(-1)

    def step(self, x_prev_in):
        """Provede jeden krok dynamiky. Vstup i výstup jsou 2D: [B, D]."""
        if x_prev_in.dim() == 1:
            x_prev_in = x_prev_in.unsqueeze(0)

        batch_size = x_prev_in.shape[0]
        w = self.L_q @ torch.randn(batch_size, self.state_dim, 1, device=self.device)
        return self.f(x_prev_in) + w.squeeze(-1)

    def measure(self, x_in):
        """Provede měření stavu. Vstup i výstup jsou 2D: [B, D] a [B, O]."""
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)

        batch_size = x_in.shape[0]
        v = self.L_r @ torch.randn(batch_size, self.obs_dim, 1, device=self.device)
        y_noiseless = self.h(x_in)
        return y_noiseless + v.squeeze(-1)