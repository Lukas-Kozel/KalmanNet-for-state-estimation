import torch
import numpy as np
from scipy.stats import multivariate_normal
import copy

class ParticleFilter:
    """
    Implementace SIR (Sequential Importance Resampling) Particle Filtru.
    Tato třída se drží logiky a struktury z tutoriálu, ale pro vysoký výkon
    používá vektorizované operace místo pomalých `for` cyklů.
    """
    
    def __init__(self, system_model, num_particles=1000):
        # ... (tato metoda zůstává beze změny)
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
        if self.N <= 0:
            raise ValueError("Počet částic (num_particles) musí být kladné celé číslo.")

    def _propagate_vectorized(self, particles, Q_np, u_current_np=None):
        """Vektorizovaná predikce pro celý mrak částic."""
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        predicted_particles_torch = self.f(particles_torch)
        predicted_particles = predicted_particles_torch.cpu().numpy()
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np, size=self.N)
        if u_current_np is None:
            return predicted_particles + noise
        else:
            return predicted_particles + noise + u_current_np

    def _compute_likelihood_vectorized(self, particles, z_np, R_np):
        """Vektorizovaný výpočet likelihood pro celý mrak částic."""
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        expected_measurements_torch = self.h(particles_torch)
        expected_measurements = expected_measurements_torch.cpu().numpy()
        likelihoods = multivariate_normal.pdf(expected_measurements, mean=z_np, cov=R_np)
        return likelihoods

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
    
    def _normalize_weights(self, weights):
        """Normalizuje váhy tak, aby jejich součet byl 1. Ekvivalent 'normalize_weights' z reference."""
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            print("Varování: Součet vah je téměř nulový. Resetuji na uniformní rozdělení.")
            weights.fill(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def get_estimate(self, particles, weights):
        mean = np.average(particles, weights=weights, axis=0)
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        return mean_torch.reshape(self.state_dim, 1), cov_torch

    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None, Q=None, R=None, resampling_threshold=0.5):
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()
        u_sequence_np = u_sequence.cpu().numpy()
        Q_np = Q.cpu().numpy() if Q is not None else self.Q_np_default
        R_np = R.cpu().numpy() if R is not None else self.R_np_default
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
            
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        particles_history = []
        
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        
        assert current_particles.shape == (self.N, self.state_dim), "Chyba inicializace!"
        
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())

        n_threshold = self.N * resampling_threshold

        for k in range(1, seq_len):
            u_k_np = u_sequence_np[:, k]
            propagated_particles = self._propagate_vectorized(current_particles, Q_np, u_current_np=u_k_np)
            
            likelihoods = self._compute_likelihood_vectorized(propagated_particles, y_seq_np[k], R_np)
            new_weights = current_weights * likelihoods
            
            current_particles = propagated_particles
            current_weights = self._normalize_weights(new_weights)
            
            assert current_particles.shape == (self.N, self.state_dim), f"Chyba po propagaci v kroku {k}!"

            effective_N = 1. / np.sum(current_weights**2)
            if effective_N < n_threshold:
                current_particles = self._systematic_resample(current_particles, current_weights)
                current_weights.fill(1.0 / self.N)
                
            assert current_particles.shape == (self.N, self.state_dim), f"Chyba na konci kroku {k}!"

            particles_history.append(current_particles.copy()) 
            
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'particles_history': particles_history
        }