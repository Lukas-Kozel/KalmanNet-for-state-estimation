import torch
import numpy as np
from tqdm import tqdm

class ParticleFilterNCLT:
    
    def __init__(self, system_model, num_particles=1000, jitter=None, resampling_threshold=0.5):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles

        self.jitter = jitter
        self.resampling_threshold = resampling_threshold
        
        self.state_dim = system_model.Q.shape[0]
        self.obs_dim = system_model.R.shape[0]
        
        self.Ex0 = system_model.Ex0.clone().detach().flatten().to(self.device)
        self.P0 = system_model.P0.clone().detach().to(self.device)
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        # Ochrana inverze
        try:
            self.L_R = torch.linalg.cholesky(self.R)
            self.R_inv = torch.cholesky_inverse(self.L_R)
        except RuntimeError:
            self.R_inv = torch.inverse(self.R)

        self.dist_Q = torch.distributions.MultivariateNormal(
            torch.zeros(self.state_dim, device=self.device), 
            self.Q
        )
        
        self.dist_init = torch.distributions.MultivariateNormal(self.Ex0, self.P0)
        
        if self.N <= 0:
            raise ValueError("Počet částic (num_particles) musí být kladné celé číslo.")
        
    def _propagate_vectorized(self, particles, Q=None, u_current=None):
        if u_current is not None:
            # KLÍČOVÁ OPRAVA: Expandování vektoru řízení u_t pro všechny částice!
            # Pokud přijde u_current o velikosti [4], naklonujeme ho na [100000, 4]
            if u_current.dim() == 1:
                u_batch = u_current.unsqueeze(0).expand(self.N, -1)
            else:
                u_batch = u_current
                
            predicted_particles = self.f(particles, u_batch)
        else:
            predicted_particles = self.f(particles)
        
        if Q is None:
            noise = self.dist_Q.sample((self.N,))
        else:
            noise = torch.distributions.MultivariateNormal(
                torch.zeros(self.state_dim, device=self.device), Q
            ).sample((self.N,))
            
        return predicted_particles + noise

    def _compute_likelihood_vectorized(self, particles, z, R=None):
        expected_measurements = self.h(particles) # (N, obs_dim)
        residuals = z - expected_measurements     # (N, obs_dim)
        
        if R is not None and R is not self.R:
             R_inv_local = torch.inverse(R)
             term1 = residuals @ R_inv_local
        else:
             term1 = residuals @ self.R_inv

        mahalanobis = (term1 * residuals).sum(dim=1) # (N,)
        log_weights = -0.5 * mahalanobis
        return torch.exp(log_weights)

    def _vectorized_systematic_resample(self, particles, weights):
        cumsum = torch.cumsum(weights, dim=0)
        cumsum[-1] = 1.0  
        
        step = 1.0 / self.N
        u0 = torch.rand(1, device=self.device).item() * step 
        u = torch.arange(self.N, device=self.device) * step + u0 
        
        indices = torch.searchsorted(cumsum, u)
        indices = torch.clamp(indices, 0, self.N - 1)
        return particles[indices]
    
    def _normalize_weights(self, weights):
        sum_weights = torch.sum(weights)
        if sum_weights < 1e-15:
            weights.fill_(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def get_estimate(self, particles, weights):
        mean = torch.sum(particles * weights.unsqueeze(1), dim=0) 
        diff = particles - mean 
        cov = (diff.T * weights) @ diff
        return mean.reshape(self.state_dim, 1), cov
    
    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None, Q=None, R=None):
        if y_seq.device != self.device:
            y_seq = y_seq.to(self.device)
            
        if u_sequence is not None and u_sequence.device != self.device:
            u_sequence = u_sequence.to(self.device)

        seq_len = y_seq.shape[0]

        curr_Q = torch.tensor(Q, device=self.device).float() if (Q is not None and not torch.is_tensor(Q)) else (Q.to(self.device) if Q is not None else None)
        curr_R = torch.tensor(R, device=self.device).float() if (R is not None and not torch.is_tensor(R)) else (R.to(self.device) if R is not None else None)

        start_mean = Ex0.flatten().to(self.device) if Ex0 is not None else self.Ex0
        start_cov = P0.to(self.device) if P0 is not None else self.P0
        
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        if Ex0 is not None or P0 is not None:
             current_particles = torch.distributions.MultivariateNormal(start_mean, start_cov).sample((self.N,))
        else:
             current_particles = self.dist_init.sample((self.N,))
             
        current_weights = torch.full((self.N,), 1.0 / self.N, device=self.device)
        
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est

        n_threshold = self.N * self.resampling_threshold

        with torch.no_grad():
            for k in tqdm(range(1, seq_len), desc="Processing PF (GPU)"):
                
                u_k = u_sequence[k] if u_sequence is not None else None
                
                # 1. Propagace
                current_particles = self._propagate_vectorized(current_particles, curr_Q, u_current=u_k)
                
                # 2. Likelihood a váhy (pouze pokud měření neobsahuje NaN)
                y_k = y_seq[k]
                if not torch.isnan(y_k).any():
                    likelihoods = self._compute_likelihood_vectorized(current_particles, y_k, curr_R)
                    weights_unnorm = current_weights * likelihoods
                    current_weights = self._normalize_weights(weights_unnorm)
                
                # 3. Resampling
                effective_N = 1.0 / torch.sum(current_weights**2)
                
                if effective_N < n_threshold:
                    current_particles = self._vectorized_systematic_resample(current_particles, current_weights)
                    current_weights.fill_(1.0 / self.N)
                    if self.jitter is not None:
                        jitter_noise = torch.randn_like(current_particles) * self.jitter
                        current_particles = current_particles + jitter_noise
                        
                # 4. Ukládání                
                x_est, P_est = self.get_estimate(current_particles, current_weights)
                x_filtered_history[k] = x_est.squeeze()
                P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history
        }