import torch
import numpy as np
from scipy.stats import multivariate_normal

class AuxiliaryParticleFilterTAN:
    """
    Vectorized Auxiliary Particle Filter (APF) with NaN and Control Input support.
    Suitable for the NCLT dataset.
    """
    
    def __init__(self, system_model, num_particles=1000):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        
        # Cache pro Numpy
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np = system_model.Q.cpu().numpy()
        self.R_np = system_model.R.cpu().numpy()
        
    def _compute_log_likelihood_vectorized(self, particles, z_np, R_np):
        """
        Returns log(p(z|x)) for all particles at once.
        """
        mask = ~np.isnan(z_np)
        if not mask.any():
            # If the measurement is NaN, the likelihood is uniform (log(1) = 0, but relatively it doesn't matter)
            # We return zeros, which means we add no information.
            return np.zeros(self.N)

        z_valid = z_np[mask]
        R_valid = R_np[np.ix_(mask, mask)]

        # Measurement prediction (h)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        expected_measurements = self.h(particles_torch).detach().cpu().numpy()
        expected_valid = expected_measurements[:, mask]

        # Log-PDF computation
        try:
            log_lik = multivariate_normal.logpdf(expected_valid, mean=z_valid, cov=R_valid)
        except np.linalg.LinAlgError:
            return np.full(self.N, -np.inf)
            
        return log_lik

    def _normalize_log_weights(self, log_weights):
        """Stable softmax: exp(log_w - max) / sum(...)"""
        max_log = np.max(log_weights)
        if np.isinf(max_log):
            return np.full(self.N, 1.0 / self.N)
            
        weights_unnorm = np.exp(log_weights - max_log)
        weights = weights_unnorm / np.sum(weights_unnorm)
        return weights

    def _systematic_resample(self, weights):
        """Returns parent indices."""
        N = len(weights)
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def get_estimate(self, particles, weights):
        # 1. Classic mean computation
        mean = np.average(particles, weights=weights, axis=0)
        
        # 2. Covariance computation
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        
        
        # Problem: With PF it happens that all particles become identical (sample impoverishment),
        # then cov = 0. ANEES divides by zero -> blows up.
        # Solution: Add a small "safety" diagonal. 
        # A value of 1e-3 means we are certain at most to millimeters, no more.
        
        min_cov_val = 1e-3 
        cov = cov + np.eye(self.state_dim) * min_cov_val
        
        # Conversion to torch
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        
        return mean_torch.reshape(self.state_dim, 1), cov_torch
    
    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None):
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()
        u_seq_np = u_sequence.cpu().numpy() if u_sequence is not None else None
        
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
        
        # Historie
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        # Inicializace
        
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        # We keep weights in linear space because resampling needs them linear
        current_weights = np.full(self.N, 1.0 / self.N)
        
        # Storing the initial state
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est

        for k in range(1, seq_len):
            u_k = u_seq_np[k] if u_seq_np is not None else None
            y_k = y_seq_np[k]
            is_measurement_valid = not np.all(np.isnan(y_k))

            # --- APF PHASE 1: Lookahead & Auxiliary Weights ---
            
            # 1. Deterministic prediction of the mean (mu) for all particles
            # mu_k^i = f(x_{k-1}^i, u_k)
            parts_torch = torch.from_numpy(current_particles).float().to(self.device)
            if u_k is not None:
                u_torch = torch.from_numpy(u_k).float().to(self.device).reshape(1, -1).repeat(self.N, 1)
                mu_particles_torch = self.f(parts_torch, u_torch)
            else:
                mu_particles_torch = self.f(parts_torch)
            
            mu_particles = mu_particles_torch.detach().cpu().numpy()

            # 2. Computation of auxiliary weights
            # If we have a measurement: log_w_aux = log(w_{k-1}) + log p(y_k | mu_k)
            # If we have no measurement: log_w_aux = log(w_{k-1})  (no new info)
            
            log_weights_prev = np.log(current_weights + 1e-300)
            
            if is_measurement_valid:
                log_lik_aux = self._compute_log_likelihood_vectorized(mu_particles, y_k, self.R_np)
                log_weights_aux = log_weights_prev + log_lik_aux
            else:
                # If there is no measurement, the auxiliary weight is just the original weight
                log_weights_aux = log_weights_prev
                
            # Normalizace pro resampling
            weights_aux = self._normalize_log_weights(log_weights_aux)

            # --- APF PHASE 2: Resampling ---
            # Select indices of "promising" particles
            parent_indices = self._systematic_resample(weights_aux)
            
            # Select parent particles
            parents = current_particles[parent_indices]
            
            # --- APF PHASE 3: Final propagation ---
            # x_k^j ~ N(f(parents^j, u_k), Q)
            # We must run the SELECTED parents through the model f() again (vectorized)
            
            parents_torch = torch.from_numpy(parents).float().to(self.device)
            if u_k is not None:
                # We already have u_torch, just check the shape
                u_torch = torch.from_numpy(u_k).float().to(self.device).reshape(1, -1).repeat(self.N, 1)
                pred_parents_torch = self.f(parents_torch, u_torch)
            else:
                pred_parents_torch = self.f(parents_torch)
                
            pred_parents = pred_parents_torch.detach().cpu().numpy()
            
            # Adding noise Q
            noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.Q_np, size=self.N)
            current_particles = pred_parents + noise

            # --- APF PHASE 4: Weight correction ---
            # Weight w_k^j = p(y_k | x_k^j) / p(y_k | mu_{parent})
            # V logaritmech: log_w_new = log_lik_final - log_lik_aux_parent
            
            if is_measurement_valid:
                # 1. Likelihood of the final particles
                log_lik_final = self._compute_log_likelihood_vectorized(current_particles, y_k, self.R_np)
                
                # 2. Likelihood of the parents (we already have it from Phase 1, we just need to take the correct indices)
                log_lik_aux_parents = log_lik_aux[parent_indices]
                
                
                log_weights_new = log_lik_final - log_lik_aux_parents
                
                # Handling: Due to noise Q, it may happen that x_k is 'out' while mu_k was 'ok'
                # or vice versa. APF weights should be around 1.0 (log 0).
            else:
                # If we have no measurement, the weights are uniform (1/N)
                # Because we resampled according to w_{k-1}, the particles are already distributed
                # according to the posterior from the past. Now they only diffuse with noise Q.
                log_weights_new = np.zeros(self.N) # log(1) = 0, po normalizaci to bude 1/N

            current_weights = self._normalize_log_weights(log_weights_new)
            
            # Estimate and save
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est
            
        return {'x_filtered': x_filtered_history, 'P_filtered': P_filtered_history}