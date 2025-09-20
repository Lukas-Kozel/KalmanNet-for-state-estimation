import torch

class DynamicSystem:
    def __init__(self, state_dim, obs_dim, Q, R, Ex0, P0, f=None, h=None, F=None, H=None, device=None):
        """
        Sjednocená třída pro lineární i nelineární dynamické systémy.
        
        Pro lineární systém zadejte matice F a H.
        Pro nelineární systém zadejte funkce f a h.
        """
        if device is None:
            self.device = Q.device
        else:
            self.device = device

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        self.Q = Q.to(self.device)
        self.R = R.to(self.device)
        self.Ex0 = Ex0.to(self.device)
        self.P0 = P0.to(self.device)
        
        if f is not None and F is not None:
            raise ValueError("Zadejte buď funkci `f` nebo matici `F`, ne obojí.")
        if h is not None and H is not None:
            raise ValueError("Zadejte buď funkci `h` nebo matici `H`, ne obojí.")

        if F is not None:
            self.F = F.to(self.device)
            self.f = lambda x: self.F @ x
            self.is_linear_f = True
        else:
            self.f = f
            self.is_linear_f = False

        if H is not None:
            self.H = H.to(self.device)
            self.h = lambda x: self.H @ x
            self.is_linear_h = True
        else:
            self.h = h
            self.is_linear_h = False

        if self.f is None or self.h is None:
            raise ValueError("Musí být zadána buď funkce (f,h) nebo matice (F,H).")

        try:
            self.L_q = torch.linalg.cholesky(self.Q)
        except torch.linalg.LinAlgError:
            print("Varování: Matice Q není pozitivně definitní.")
            
        try:
            self.L_r = torch.linalg.cholesky(self.R)
        except torch.linalg.LinAlgError:
            print("Varování: Matice R není pozitivně definitní.")

        try:
            self.L_p0 = torch.linalg.cholesky(self.P0)
        except torch.linalg.LinAlgError:
            print("Varování: Matice P0 není pozitivně definitní.")

    def get_initial_state(self):
        """Generuje náhodný počáteční stav."""
        z = torch.randn(self.state_dim, 1, device=self.device)
        return self.Ex0 + self.L_p0 @ z

    def get_deterministic_initial_state(self):
        """Vrací deterministickou střední hodnotu počátečního stavu."""
        return self.Ex0.clone()

    def step(self, x_prev):
        """Provede jeden krok dynamiky systému."""
        w = self.L_q @ torch.randn(self.state_dim, 1, device=self.device)
        return self.f(x_prev) + w

    def measure(self, x):
        """Provede měření stavu."""
        v = self.L_r @ torch.randn(self.obs_dim, 1, device=self.device)
        y_noiseless = self.h(x).view(self.obs_dim, 1)
        return y_noiseless + v