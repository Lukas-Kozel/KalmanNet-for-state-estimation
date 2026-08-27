import torch
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class AuxiliaryParticleFilter:
    """
    Implementation of Auxiliary Particle Filter (APF).
    See Section 5.2.2 in "Particle Filters: A Hands-On Tutorial" (Elfring et al.).
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
        """
        Propagates ONE particle. This is the implementation of the "draw sample" step from Algorithm 2 (p. 14).
        Generates a new particle x_k^i from old x_{k-1}^i using the process model.
        x_k^i ~ p(x_k | x_{k-1}^i)

        In the paper, this step is described in Example 3, equation (15) (p. 9):
        x_k^i ~ N(f(x_{k-1}^i), Q)
        """
        # Convert one particle to Torch tensor to use the model function f
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)

        # Apply deterministic part of motion model: f(x_{k-1}^i)
        predicted_particle_torch = self.f(particle_torch)
        predicted_particle = predicted_particle_torch.cpu().numpy().flatten()

        # Add stochastic part: random sample of process noise w_k
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np) # multivariate because we want just a sample

        # Result is new particle position at time k
        return predicted_particle + noise # x_k^i ~ N(x_{k-1}^i + B_k*u_k, Q) from equation (15)
    

    def _compute_likelihood(self, particle, z_np, R_np):
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        expected_measurement_torch = self.h(particle_torch)
        expected_measurement = expected_measurement_torch.cpu().numpy().flatten()
        likelihood = multivariate_normal.pdf(z_np, mean=expected_measurement, cov=R_np)
        return likelihood

    def _normalize_weights(self, weights):
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            weights.fill(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def _multinomial_resample_indices(self, weights):
        """
        Multinomial resampling returning only indices. APF in the reference uses
        this method to select "promising parents".
        """
        N = len(weights)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        random_picks = np.random.rand(N)
        indexes = np.searchsorted(cumulative_sum, random_picks)
        return indexes

    def get_estimate(self, particles, weights):
        mean = np.average(particles, weights=weights, axis=0)
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        return mean_torch.reshape(self.state_dim, 1), cov_torch
    
    def process_sequence(self, y_seq, Ex0=None, P0=None, Q=None, R=None):
        """
        Processes entire measurement sequence using APF algorithm.
        Logic corresponds to 4 steps described in Section 5.2.2 of the paper.
        """
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()

        Q_np = Q.cpu().numpy() if Q is not None else self.Q_np_default
        R_np = R.cpu().numpy() if R is not None else self.R_np_default
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
            
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        particles_history = []
        
        # Initialization
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())

        for k in tqdm(range(1,seq_len), desc="Processing sequence APF"):

            # --- APF Phase 1: Compute auxiliary weights ---
            # "Compute Ns point estimates... Then compute weights for these characterizations"

            # Array for storing first phase results
            char_likelihoods = np.empty(self.N)
            aux_weights = np.empty(self.N)

            for i in range(self.N):
                # compute μ
                char_particle = self._propagate_particle(current_particles[i], Q_np)

                # Evaluate it against new measurement z_k and store likelihood
                char_likelihoods[i] = self._compute_likelihood(char_particle, y_seq_np[k], R_np)

                # Compute auxiliary weight: w_aux ∝ w_{k-1} * p(z_k|μ_i)
                aux_weights[i] = current_weights[i] * char_likelihoods[i]

            # Normalize auxiliary weights
            aux_weights = self._normalize_weights(aux_weights)

            # --- APF Phase 2: Resampling "promising parents" ---
            # "Use weights w_i from step 1 in a resampling step... store the indices i^j"
            parent_indices = self._multinomial_resample_indices(aux_weights)

            # --- APF Phase 3 and 4: Final prediction and weight computation ---
            # "Perform a prediction step for each of the... indices... Compute the weights..."
            
            new_particles = np.empty((self.N, self.state_dim))
            new_weights = np.empty(self.N)

            for i in range(self.N):
                # Get index of selected "parent"
                # does not iterate over all particles, only over selected indices
                parent_index = parent_indices[i]

                # Perform full stochastic prediction from this parent
                new_particles[i] = self._propagate_particle(current_particles[parent_index], Q_np)

                # Compute final likelihood for new particle
                final_likelihood = self._compute_likelihood(new_particles[i], y_seq_np[k], R_np)

                char_likelihood_of_parent = char_likelihoods[parent_index]
                if char_likelihood_of_parent < 1e-10:
                    char_likelihood_of_parent = 1e-10

                # Compute final weight according to equation (19) in paper: w_k ∝ p(z_k|x_k) / p(z_k|μ)
                new_weights[i] = final_likelihood / char_likelihood_of_parent

            # Replace old particles and weights with new, normalized ones
            current_particles = new_particles
            current_weights = self._normalize_weights(new_weights)
            
            particles_history.append(current_particles.copy())
            
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'particles_history': particles_history
        }
