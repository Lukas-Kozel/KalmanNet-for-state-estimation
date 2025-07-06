import torch

class NonlinearSystem:
    def __init__(self, f, h, Q, R, Ex0, P0):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.Ex0 = Ex0
        self.P0 = P0
        
        self.state_dim = Ex0.shape[0]
        self.obs_dim = R.shape[0]

    def get_initial_state(self):
        return torch.randn(self.state_dim, 1) * torch.sqrt(self.P0) + self.Ex0

    def step(self, x_prev):
        w = torch.randn(self.state_dim, 1) * torch.sqrt(self.Q)
        return self.f(x_prev) + w

    def measure(self, x):
        v = torch.randn(self.obs_dim, 1) * torch.sqrt(self.R)
        return self.h(x) + v