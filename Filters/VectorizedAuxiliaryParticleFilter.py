import torch
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class VectorizedAuxiliaryParticleFilter:
    """
    Fully vectorized implementation of Auxiliary Particle Filter (APF).
    """
    
    def __init__(self, system_model, num_particles=1000):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        # Note: We assume H maps to obs_dim. We obtain it from R.
        self.obs_dim = system_model.R.shape[0] 
        
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np_default = system_model.Q.cpu().numpy()
        self.R_np_default = system_model.R.cpu().numpy()
        
        # For likelihood vectorization we prepare a fixed object for R (if R is const.)
        # If R changes, we will have to create it in each step.
        self.mvn_R = multivariate_normal(mean=np.zeros(self.obs_dim), cov=self.R_np_default)

    def _propagate_particles(self, particles, Q_np):
        """
        VECTORIZED propagation.
        Input: particles (N, state_dim)
        Output: propagated_particles (N, state_dim)
        """
        # 1. Batch conversion to torch (N, state_dim)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        
        # 2. Apply model f to entire batch at once
        # Assumes self.f can work with (N, dim), which NN usually can
        with torch.no_grad():
            predicted_particles_torch = self.f(particles_torch)
        
        predicted_particles = predicted_particles_torch.cpu().numpy()
        
        # 3. Generate noise for all particles at once
        # size=self.N ensures generation of matrix (N, state_dim)
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np, size=self.N)
        
        return predicted_particles + noise

    def _compute_likelihoods(self, particles, z_np, R_np):
        """
        VECTORIZED likelihood computation.
        Input: particles (N, state_dim), z_np (obs_dim,)
        Output: likelihoods (N,)
        """
        # 1. Transform all particles into measurement space h(x)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        
        with torch.no_grad():
            expected_measurements_torch = self.h(particles_torch) # (N, obs_dim)
        
        expected_measurements = expected_measurements_torch.cpu().numpy()
        
        # 2. Compute residuals (innovations) for all particles
        # Broadcasting: (obs_dim,) - (N, obs_dim) -> (N, obs_dim)
        residuals = z_np - expected_measurements
        
        # 3. Evaluate PDF
        # Mathematical trick: p(z | x) ~ N(z; h(x), R) is equivalent to N(z - h(x); 0, R)
        # Therefore we can use one distribution N(0, R) and just feed residuals into it.
        # If R is constant, we use self.mvn_R, otherwise we create a new one.
        if np.array_equal(R_np, self.R_np_default):
            likelihoods = self.mvn_R.pdf(residuals)
        else:
            likelihoods = multivariate_normal.pdf(residuals, mean=np.zeros(self.obs_dim), cov=R_np)
            
        return likelihoods

    def _normalize_weights(self, weights):
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            weights.fill(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def _multinomial_resample_indices(self, weights):
        """
        Remains the same, it was already efficient.
        """
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        random_picks = np.random.rand(self.N)
        indexes = np.searchsorted(cumulative_sum, random_picks)
        return indexes

    def get_estimate(self, particles, weights):
        """
        Vectorized computation of weighted mean and covariance.
        """
        # Mean: (dim,)
        mean = np.average(particles, weights=weights, axis=0)
        
        # Cov: (N, dim)
        diff = particles - mean
        # (dim, N) @ (N, dim) -> (dim, dim)
        cov = (diff.T * weights) @ diff
        
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        return mean_torch.reshape(self.state_dim, 1), cov_torch
    
    def process_sequence(self, y_seq, Ex0=None, P0=None, Q=None, R=None):
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()

        Q_np = Q.cpu().numpy() if Q is not None else self.Q_np_default
        R_np = R.cpu().numpy() if R is not None else self.R_np_default
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
            
        # If R differs from default, we update mvn object for speed
        if not np.array_equal(R_np, self.R_np_default):
             self.mvn_R = multivariate_normal(mean=np.zeros(self.obs_dim), cov=R_np)

        # Pre-allocation of history for speed
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        particles_history = []
        
        # Initialization
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        
        # Save initial state
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())

        # MAIN LOOP
        for k in tqdm(range(1, seq_len), desc="APF Vectorized"):
            
            # --- APF Phase 1: Characterization (Mean prediction) ---
            # Vectorized computation of "mu" for all particles
            char_particles = self._propagate_particles(current_particles, Q_np)
            
            # Vectorized computation of p(z_k | mu)
            char_likelihoods = self._compute_likelihoods(char_particles, y_seq_np[k], R_np)
            
            # Auxiliary weights
            aux_weights = current_weights * char_likelihoods
            aux_weights = self._normalize_weights(aux_weights)

            # --- APF Phase 2: Resampling indices ---
            parent_indices = self._multinomial_resample_indices(aux_weights)
            
            # Select parents using "Fancy Indexing" (fast)
            # parents shape: (N, state_dim)
            parents = current_particles[parent_indices]
            
            # We also need parent likelihoods for denominator in phase 4
            # parent_char_likelihoods shape: (N,)
            parent_char_likelihoods = char_likelihoods[parent_indices]

            # --- APF Phase 3: Final Propagation ---
            # Propagate selected parents (full stochastic step)
            new_particles = self._propagate_particles(parents, Q_np)

            # --- APF Phase 4: Weights ---
            # p(z_k | x_k)
            final_likelihoods = self._compute_likelihoods(new_particles, y_seq_np[k], R_np)
            
            # Guard against division by zero
            safe_denom = np.maximum(parent_char_likelihoods, 1e-10)
            
            # w_k = p(z|x) / p(z|mu)
            new_weights = final_likelihoods / safe_denom
            
            # Normalization and update
            current_weights = self._normalize_weights(new_weights)
            current_particles = new_particles
            
            # Saving (optional: particles_history uses a lot of memory, consider disabling for large N)
            particles_history.append(current_particles.copy())
            
            # State estimate
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'particles_history': particles_history
        }
