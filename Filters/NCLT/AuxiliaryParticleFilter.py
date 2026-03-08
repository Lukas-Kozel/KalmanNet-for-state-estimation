import torch
import numpy as np
from tqdm import tqdm

class AuxiliaryParticleFilterNCLT:
    """
    Extrémně rychlá (GPU Vectorized) verze Adaptive Auxiliary Particle Filter (AAPF).
    
    Klíčové optimalizace:
    1. Předpočítané Choleskyho rozklady pro okamžité generování procesního šumu.
    2. Mahalanobisova vzdálenost místo pomalých objektů torch.distributions.
    3. Plná vektorizace resampling fáze.
    """
    
    def __init__(self, system_model, num_particles=2000, jitter_strength=2.0, r_inflation=20.0, resample_threshold=0.5):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        self.obs_dim = system_model.R.shape[0]
        
        # Hyperparametry
        self.jitter_strength = jitter_strength      
        self.r_inflation = r_inflation              
        self.resample_threshold = resample_threshold 
        
        # Tenzory na GPU
        self.Ex0 = system_model.Ex0.clone().detach().flatten().to(self.device)
        self.P0 = system_model.P0.clone().detach().to(self.device)
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        # Záchranná brzda pro kovarianci
        self.min_cov_matrix = torch.diag(torch.diag(self.Q)) * 0.01

        # --- OPTIMALIZACE PRO EXTRÉMNÍ RYCHLOST ---
        # 1. Předpočítaný Cholesky pro procesní šum Q
        try:
            self.L_Q = torch.linalg.cholesky(self.Q)
        except RuntimeError:
            self.L_Q = torch.linalg.cholesky(self.Q + torch.eye(self.state_dim, device=self.device)*1e-6)

        # 2. Předpočítaná inverze pro nafouknuté R (r_inflation)
        self.inflated_R = self.R * self.r_inflation
        try:
            L_R_inf = torch.linalg.cholesky(self.inflated_R)
            self.inv_inflated_R = torch.cholesky_inverse(L_R_inf)
        except RuntimeError:
            self.inv_inflated_R = torch.inverse(self.inflated_R)

        # 3. Distribuce pouze pro krok k=0
        self.dist_init = torch.distributions.MultivariateNormal(self.Ex0, self.P0)

        print(f"Initialized FAST Adaptive APF: N={self.N}, Jitter={self.jitter_strength}, R-Infl={self.r_inflation}, ResThr={self.resample_threshold}")

    def _sample_process_noise(self):
        """Bleskové vzorkování procesního šumu Q."""
        standard_noise = torch.randn(self.N, self.state_dim, device=self.device)
        return standard_noise @ self.L_Q.T

    def _compute_log_likelihood_vectorized(self, particles, z):
        """
        Extrémně rychlý výpočet log-likelihood pomocí Mahalanobisovy vzdálenosti.
        Zpracovává pouze validní (ne-NaN) dimenze měření.
        """
        mask = ~torch.isnan(z)
        if not mask.any():
            return torch.zeros(self.N, device=self.device)
            
        z_valid = z[mask]
        
        # Výběr validní sub-matice z předpočítané inverze
        # V PyTorch to elegantně řeší dvojité maskování
        inv_R_valid = self.inv_inflated_R[mask][:, mask]
        
        expected_z = self.h(particles) # (N, obs_dim)
        expected_valid = expected_z[:, mask] # (N, valid_obs_dim)
        
        residuals = z_valid - expected_valid # (N, valid_obs_dim)
        
        # Mahalanobis = (res @ R^-1) * res
        term1 = residuals @ inv_R_valid
        mahalanobis = (term1 * residuals).sum(dim=1) # (N,)
        
        # Log likelihood je úměrná -0.5 * Mahalanobis
        return -0.5 * mahalanobis

    def _normalize_log_weights(self, log_weights):
        """Stabilní Log-Sum-Exp normalizace vah."""
        max_log = torch.max(log_weights)
        if torch.isinf(max_log):
            return torch.full((self.N,), 1.0 / self.N, device=self.device)
            
        weights_unnorm = torch.exp(log_weights - max_log)
        sum_w = torch.sum(weights_unnorm)
        
        if sum_w < 1e-15:
            return torch.full((self.N,), 1.0 / self.N, device=self.device)
            
        return weights_unnorm / sum_w

    def _systematic_resample(self, weights):
        """Plně vektorizovaný Systematic Resampling bez .item() bariéry."""
        cumsum = torch.cumsum(weights, dim=0)
        cumsum[-1] = 1.0  
        
        step = 1.0 / self.N
        u0 = torch.rand(1, device=self.device) * step 
        u = torch.arange(self.N, device=self.device) * step + u0 
        
        indexes = torch.searchsorted(cumsum, u)
        return torch.clamp(indexes, 0, self.N - 1)

    def get_estimate(self, particles, weights):
        """Vrátí vážený průměr a kovarianci s minimální bezpečnostní podlahou."""
        mean = torch.sum(particles * weights.unsqueeze(1), dim=0)
        diff = particles - mean
        cov = (diff.T * weights) @ diff
        cov = cov + self.min_cov_matrix
        return mean.reshape(self.state_dim, 1), cov

    def apply_jitter(self, particles):
        """Direct Roughening bleskově."""
        if self.jitter_strength > 0:
            noise = self._sample_process_noise() # Znovupoužití rychlého generátoru
            return particles + noise * self.jitter_strength
        return particles

    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None):
        if y_seq.device != self.device:
            y_seq = y_seq.to(self.device)
            
        if u_sequence is not None and u_sequence.device != self.device:
            u_sequence = u_sequence.to(self.device)
            
        seq_len = y_seq.shape[0]
        
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

        # Cache pro batch vstupu u_k, aby se neexpandoval zbytečně uvnitř if-else větví
        u_batch_cache = None

        with torch.no_grad():
            for k in tqdm(range(1, seq_len), desc="Fast APF (GPU)"):
                
                u_k = u_sequence[k] if u_sequence is not None else None
                y_k = y_seq[k]
                
                # Příprava batche pro vstupní vektor
                if u_k is not None:
                    u_batch_cache = u_k.unsqueeze(0).expand(self.N, -1)
                else:
                    u_batch_cache = None

                is_measurement_valid = not torch.isnan(y_k).all()

                if not is_measurement_valid:
                    # ==============================
                    # A. OUTAGE MODE (Výpadek GPS)
                    # ==============================
                    if u_batch_cache is not None:
                        pred_particles = self.f(current_particles, u_batch_cache)
                    else:
                        pred_particles = self.f(current_particles)
                    
                    current_particles = pred_particles + self._sample_process_noise()
                    current_particles = self.apply_jitter(current_particles)
                    
                else:
                    # ==============================
                    # B. MEASUREMENT MODE
                    # ==============================
                    n_eff = 1.0 / torch.sum(current_weights ** 2)
                    
                    if n_eff < (self.N * self.resample_threshold):
                        # --- B1. APF UPDATE (Resampling nutný) ---
                        
                        # 1. Lookahead (mu)
                        if u_batch_cache is not None:
                            mu_particles = self.f(current_particles, u_batch_cache)
                        else:
                            mu_particles = self.f(current_particles)
                        
                        # 2. Auxiliary Weights
                        log_weights_prev = torch.log(current_weights + 1e-30)
                        log_lik_aux = self._compute_log_likelihood_vectorized(mu_particles, y_k)
                        
                        log_weights_aux = log_weights_prev + log_lik_aux
                        weights_aux = self._normalize_log_weights(log_weights_aux)
                        
                        # 3. Resampling indexů
                        parent_indices = self._systematic_resample(weights_aux)
                        
                        # 4. Propagace přeživších rodičů (s využitím indexovaných rodičů)
                        parents = current_particles[parent_indices]
                        if u_batch_cache is not None:
                            pred_parents = self.f(parents, u_batch_cache)
                        else:
                            pred_parents = self.f(parents)
                            
                        current_particles = pred_parents + self._sample_process_noise()
                        current_particles = self.apply_jitter(current_particles)

                        # 5. APF Weight Update
                        log_lik_final = self._compute_log_likelihood_vectorized(current_particles, y_k)
                        log_lik_aux_parents = log_lik_aux[parent_indices]
                        
                        log_weights_new = log_lik_final - log_lik_aux_parents
                        current_weights = self._normalize_log_weights(log_weights_new)
                        
                    else:
                        # --- B2. SIS UPDATE (Resampling není nutný) ---
                        if u_batch_cache is not None:
                            pred_particles = self.f(current_particles, u_batch_cache)
                        else:
                            pred_particles = self.f(current_particles)
                            
                        current_particles = pred_particles + self._sample_process_noise()
                        current_particles = self.apply_jitter(current_particles)
                        
                        log_weights_prev = torch.log(current_weights + 1e-30)
                        log_lik = self._compute_log_likelihood_vectorized(current_particles, y_k)
                        
                        log_weights_new = log_weights_prev + log_lik
                        current_weights = self._normalize_log_weights(log_weights_new)

                # Odhad
                x_est, P_est = self.get_estimate(current_particles, current_weights)
                x_filtered_history[k] = x_est.squeeze()
                P_filtered_history[k] = P_est
            
        return {'x_filtered': x_filtered_history, 'P_filtered': P_filtered_history}