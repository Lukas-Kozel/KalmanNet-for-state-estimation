import torch

class LinearSystem:
    def __init__(self,Ex0,P0, F, H, Q, R):
        self.Ex0 = Ex0  # očekávaná hodnota počátečního stavu
        self.P0 = P0  # Počáteční kovarianční matice
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state_dim = F.shape[0]
        self.obs_dim = H.shape[0]

    def get_initial_state(self):
        """
        Vrátí počáteční stav systému.
        """
        return torch.randn(self.state_dim, 1) * torch.sqrt(self.P0) + self.Ex0
    

    def get_deterministic_initial_state(self):
        """
        Vrátí počáteční stav systému jako deterministickou hodnotu.
        """
        return self.Ex0.unsqueeze(1)

    def step(self, x_prev):
        w = torch.randn(self.state_dim, 1) * torch.sqrt(self.Q)
        return self.F @ x_prev + w

    def measure(self, x):
        v = torch.randn(self.obs_dim, 1) * torch.sqrt(self.R)
        return self.H @ x + v