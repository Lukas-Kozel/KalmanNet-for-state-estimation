import torch
import numpy as np
from scipy.stats import multivariate_normal
import copy

class ParticleFilterSIR:
    """
    Implementation of SIR (Sequential Importance Resampling) Particle Filter.
    This class follows as closely as possible the logic and structure from the tutorial
    "Particle Filters: A Hands-On Tutorial" (Elfring et al.).
    
    The goal is to exactly replicate the algorithm described in the paper, especially
    Algorithm 2: "Particle filter with resampling" (page 14).
    """
    
    def __init__(self, system_model, num_particles=1000):
        """
        Initialization of SIR filter.
        Here we set all necessary parameters that the filter will use.
        """
        self.device = system_model.Q.device
        
        # General model functions that we will call.
        # In the paper these functions are denoted as f_k and h_k in equations (1) and (2) (page 4).
        self.f = system_model.f  # Process model (motion model)
        self.h = system_model.h  # Measurement model
        
        # Filter parameters
        self.N = num_particles         # Number of particles, denoted as N_s in the paper
        self.state_dim = system_model.Q.shape[0]
        
        # Store initial conditions (prior) and noise matrices from the model.
        # These values will be used for initialization and during filtering.
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np_default = system_model.Q.cpu().numpy() # Covariance of process noise 'v' from equation (1)
        self.R_np_default = system_model.R.cpu().numpy() # Covariance of measurement noise 'n' from equation (2)
        
        # Internal filter state that will change at each step.
        # Corresponds to the set {x_k^i, w_k^i} from the pseudocode in the paper.
        self.particles = None  # NumPy array (N, state_dim), states of all particles x_k^i
        self.weights = None    # NumPy array (N,), weights of all particles w_k^i

    def _propagate_particle(self, particle, Q_np):
        """
        Propagates ONE particle. This is implementation of the "draw sample" step from Algorithm 2 (p. 14).
        Generates new particle x_k^i from old x_{k-1}^i using process model.
        x_k^i ~ p(x_k | x_{k-1}^i)
        
        In the paper this step is described in Example 3, equation (15) (p. 9):
        x_k^i ~ N(f(x_{k-1}^i), Q)
        """
        # Convert one particle to Torch tensor to use model function f
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        
        # Apply deterministic part of motion model: f(x_{k-1}^i)
        predicted_particle_torch = self.f(particle_torch)
        predicted_particle = predicted_particle_torch.cpu().numpy().flatten()
        
        # Add stochastic part: random sample of process noise w_k
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np) # here is multivariate because I want only a sample
        
        # Result is new particle position at time k
        return predicted_particle + noise # x_k^i ~ N(x_{k-1}^i + B_k*u_k, Q) from equation (15)

    def _compute_likelihood(self, particle, z_np, R_np):
        """
        Computes probability (likelihood) p(z_k|x_k^i) for ONE particle.
        This value indicates how much the particle state 'particle' agrees with actual measurement 'z_np'.
        It is a key component for weight update (equations 16 and 17, p. 9).
        """
        # Convert particle to Torch tensor for use of model function h
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        
        # Compute expected measurement for given particle: h(x_k^i)
        expected_measurement_torch = self.h(particle_torch)
        expected_measurement = expected_measurement_torch.cpu().numpy().flatten()
        
        # Compute PDF value of Gaussian distribution of measurement noise.
        # We ask: "What is the probability that we would measure 'z_np' if the
        # true state were 'particle' and our sensor had noise with covariance R?"
        likelihood = multivariate_normal.pdf(z_np, mean=expected_measurement, cov=R_np) # here is pdf because I want specific realization at given point
        return likelihood

    def _normalize_weights(self, weights):
        """Normalizes weights so their sum is 1. Equivalent to 'normalize_weights' from reference."""
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            # Protection against numerical collapse when all weights are practically zero.
            # In such case reset weights to uniform distribution.
            print("Warning: Sum of weights is almost zero. Resetting to uniform distribution.")
            weights.fill(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def _systematic_resample(self, particles, weights):
        """
        Implementation of Systematic Resampling. This algorithm fights against degeneracy
        (Challenge I, section 4.2.1) by creating a new particle population where particles
        with high weight are cloned and those with low weight are removed.
        Logic is exact replica of reference code.
        """
        N = len(particles)
        
        # Create cumulative sum of weights (CDF). Imagine it as a "pie"
        # where size of slice for each particle corresponds to its weight.
        Q = np.cumsum(weights)

        # Generate only ONE random starting point in first segment [1e-10, 1/N).
        u0 = np.random.uniform(1e-10, 1.0 / N)
        
        new_particles = np.empty_like(particles)
        n = 0  # Index for new (resampled) particles
        m = 0  # Index for old particles (for traversing CDF)
        
        while n < N:
            # Compute position of "pointer" for current new particle.
            # All pointers are exactly 1/N apart.
            u = u0 + float(n)/ N
            
            # Find first particle in CDF whose "pie slice" extends beyond our pointer.
            while u > Q[m]:
                m += 1
            
            # Copy this selected particle into new set.
            new_particles[n] = particles[m]
            
            n += 1
            
        return new_particles

    def get_estimate(self, particles, weights):
        """
        Computes final state estimate as weighted average of all particles.
        Paper mentions that particle cloud represents posterior (equation 14),
        and this is way to obtain one point estimate from it.
        """
        mean = np.average(particles, weights=weights, axis=0) # weighted average over particles and weights
        # Weighted covariance for uncertainty estimate (needed for ANEES)
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        
        # Convert to Torch tensors
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        return mean_torch.reshape(self.state_dim, 1), cov_torch

    def process_sequence(self, y_seq, Ex0=None, P0=None, Q=None, R=None, resampling_threshold=0.5):
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()

        Q_np = Q.cpu().numpy() if Q is not None else self.Q_np_default
        R_np = R.cpu().numpy() if R is not None else self.R_np_default
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
            
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        particles_history = []
        
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        
        # Safety check
        assert current_particles.shape == (self.N, self.state_dim), "Initialization error!"
        
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())

        n_threshold = self.N * resampling_threshold

        for k in range(1, seq_len):
            temp_propagated_particles = np.empty((self.N, self.state_dim))
            temp_new_weights = np.empty(self.N)

            for i in range(self.N):
                temp_propagated_particles[i] = self._propagate_particle(current_particles[i], Q_np)
                likelihood = self._compute_likelihood(temp_propagated_particles[i], y_seq_np[k], R_np)
                temp_new_weights[i] = current_weights[i] * likelihood
            
            # Weighted particles BEFORE resampling
            propagated_particles = temp_propagated_particles
            propagated_weights = self._normalize_weights(temp_new_weights)
            
            assert propagated_particles.shape == (self.N, self.state_dim), f"Error after propagation in step {k}!"

            effective_N = 1. / np.sum(propagated_weights**2)
            if effective_N < n_threshold:
                # 2. Perform resampling
                resampled_particles = self._systematic_resample(propagated_particles, propagated_weights)
                
                # Final particles for this step
                current_particles = resampled_particles
                current_weights = np.full(self.N, 1.0 / self.N)
            else:
                # If we do not resample, transfer weighted particles to next step
                current_particles = propagated_particles
                current_weights = propagated_weights
                
            assert current_particles.shape == (self.N, self.state_dim), f"Error at end of step {k}!"

            particles_history.append(current_particles.copy()) 
            
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'particles_history': particles_history
        }
