import torch
import numpy as np
from scipy.stats import multivariate_normal

class ParticleFilterMH:
    """
    Implementace SIR filtru s Resample-Move krokem (Metropolis-Hastings).
    Tato třída implementuje Algorithm 3 a Algorithm 4 z článku.
    """
    
    def __init__(self, system_model, num_particles=1000):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np_default = system_model.Q.cpu().numpy()
        self.R_np_default = system_model.R.cpu().numpy()
        self.particles = None
        self.weights = None

    def _propagate_particle(self, particle, Q_np):
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        predicted_particle_torch = self.f(particle_torch)
        predicted_particle = predicted_particle_torch.cpu().numpy().flatten()
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np)
        return predicted_particle + noise

    def _compute_likelihood(self, particle, z_np, R_np):
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        expected_measurement_torch = self.h(particle_torch)
        expected_measurement = expected_measurement_torch.cpu().numpy().flatten()
        likelihood = multivariate_normal.pdf(z_np, mean=expected_measurement, cov=R_np)
        return likelihood

    def _normalize_weights(self, weights):
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            weights.fill(1.0 / len(weights))
        else:
            weights /= sum_weights
        return weights

    def _systematic_resample(self, particles, weights):
        N = len(particles)
        Q = np.cumsum(weights)
        u0 = np.random.uniform(0.0, 1.0 / N)
        new_particles = np.empty_like(particles)
        n, m = 0, 0
        while n < N:
            u = u0 + n / N
            while u > Q[m]:
                m += 1
            new_particles[n] = particles[m]
            n += 1
        return new_particles
        
    def _metropolis_hastings_move(self, resampled_particles, parent_particles_before_resampling, z_np, Q_np, R_np):
        """
        implementace Algorithm 4 z článku.
        """
        final_particles = resampled_particles.copy()
        
        for i in range(self.N):
            # Step 1: sample a threshold from a uniform distribution
            u_prime = np.random.rand()

            # původní resamplovaná částice (x_k^i)
            particle_old = resampled_particles[i]
            
            # Nalezení "rodiče" z času k-1 (x_{k-1}^j). 
            # Předpoklad, že i-tá resamplovaná částice
            # pochází z i-tého rodiče před resampligem.
            parent_particle = parent_particles_before_resampling[i]
            
            # Step 2: propagate particle previous time step
            # Vytvoříme nového kandidáta x_k' z rodiče
            particle_new = self._propagate_particle(parent_particle, Q_np)

            # Step 3: likelihood for propagated particle step 2
            l1 = self._compute_likelihood(particle_new, z_np, R_np)
            
            # Step 4: likelihood resampled particle (input to MH step)
            l2 = self._compute_likelihood(particle_old, z_np, R_np)

            # Step 5: determine whether or not to keep move
            if l2 == 0:
                acceptance_ratio = 1.0
            else:
                acceptance_ratio = min(1.0, l1 / l2)
            
            if u_prime < acceptance_ratio:
                final_particles[i] = particle_new
            
        return final_particles


    def get_estimate(self, particles, weights):
        mean = np.average(particles, weights=weights, axis=0)
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        return mean_torch.reshape(self.state_dim, 1), cov_torch
    
    def process_sequence(self, y_seq, Ex0=None, P0=None, Q=None, R=None, resampling_threshold=0.5, use_mh_step=True):
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()
        Q_np = Q.cpu().numpy() if Q is not None else self.Q_np_default
        R_np = R_np = R.cpu().numpy() if R is not None else self.R_np_default
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        particles_history = []
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())
        n_threshold = self.N * resampling_threshold

        for k in range(1, seq_len):
            
            # Uchováme si rodiče z k-1 pro MH krok
            parent_particles_for_mh = current_particles.copy()

            temp_propagated_particles = np.empty((self.N, self.state_dim))
            temp_new_weights = np.empty(self.N)
            for i in range(self.N):
                temp_propagated_particles[i] = self._propagate_particle(current_particles[i], Q_np)
                likelihood = self._compute_likelihood(temp_propagated_particles[i], y_seq_np[k], R_np)
                temp_new_weights[i] = current_weights[i] * likelihood
            
            propagated_particles = temp_propagated_particles
            propagated_weights = self._normalize_weights(temp_new_weights)
            
            effective_N = 1. / np.sum(propagated_weights**2)
            if effective_N < n_threshold:
                resampled_particles = self._systematic_resample(propagated_particles, propagated_weights)
                
                if use_mh_step:
                    moved_particles = self._metropolis_hastings_move(
                        resampled_particles, parent_particles_for_mh, y_seq_np[k], Q_np, R_np)
                    current_particles = moved_particles
                else:
                    current_particles = resampled_particles
                
                current_weights = np.full(self.N, 1.0 / self.N)
            else:
                current_particles = propagated_particles
                current_weights = propagated_weights
                
            particles_history.append(current_particles.copy()) 
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'particles_history': particles_history
        }