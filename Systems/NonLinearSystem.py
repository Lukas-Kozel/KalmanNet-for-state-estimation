import torch

class NonlinearSystem:
    def __init__(self, f, h, Q, R, Ex0, P0):

        self.device = Q.device


        self.f = f
        self.h = h
        self.Q = Q.to(self.device)
        self.R = R.to(self.device)
        self.Ex0 = Ex0.to(self.device)
        self.P0 = P0.to(self.device)
        
        self.state_dim = Ex0.shape[0]
        self.obs_dim = R.shape[0]

        self.L_q = torch.linalg.cholesky(self.Q)
        self.L_r = torch.linalg.cholesky(self.R)
        self.L_p0 = torch.linalg.cholesky(self.P0)

    def get_initial_state(self):
        z = torch.randn(self.state_dim, 1, device=self.device)
        return self.Ex0 + self.L_p0 @ z

    def step(self, x_prev):
        z = torch.randn(self.state_dim, 1, device=self.device)
        w = self.L_q @ z
        return self.f(x_prev) + w

    def measure(self, x):
        z = torch.randn(self.obs_dim, 1, device=self.device)
        v = self.L_r @ z
        
        y_noiseless = self.h(x).view(self.obs_dim, 1)
        return y_noiseless + v