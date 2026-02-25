import torch
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class AuxiliaryParticleFilter:
    """
    Implementace Auxiliary Particle Filtru (APF).
    Viz sekce 5.2.2. v "Particle Filters: A Hands-On Tutorial" (Elfring et al.).
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
        Propaguje JEDNU částici. Toto je implementace kroku "draw sample" z Algorithm 2 (str. 14).
        Generuje novou částici x_k^i ze staré x_{k-1}^i pomocí process modelu.
        x_k^i ~ p(x_k | x_{k-1}^i)
        
        V článku je tento krok popsán v Example 3, rovnice (15) (str. 9):
        x_k^i ~ N(f(x_{k-1}^i), Q)
        """
        # Převedeme jednu částici na Torch tenzor, abychom mohli použít modelovou funkci f
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        
        # Aplikujeme deterministickou část modelu pohybu: f(x_{k-1}^i)
        predicted_particle_torch = self.f(particle_torch)
        predicted_particle = predicted_particle_torch.cpu().numpy().flatten()
        
        # Přidáme stochastickou část: náhodný vzorek procesního šumu w_k
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np) # zde je multivariate, protože chci jen vzorek
        
        # Výsledkem je nová pozice částice v čase k
        return predicted_particle + noise # x_k^i ~ N(x_{k-1}^i + B_k*u_k, Q) z rovnice (15)
    

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
        Multinomial Resampling vracející pouze indexy. APF v referenci používá
        tuto metodu pro výběr "nadějných rodičů".
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
        Zpracuje celou sekvenci měření pomocí APF algoritmu.
        Logika odpovídá 4 krokům popsaným v sekci 5.2.2 článku.
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
        
        # Inicializace
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())

        for k in tqdm(range(1,seq_len), desc="Processing sequence APF"):
            
            # --- APF Fáze 1: Výpočet pomocných vah ---
            # "Compute Ns point estimates... Then compute weights for these characterizations"
            
            # Pole pro uložení výsledků první fáze
            char_likelihoods = np.empty(self.N)
            aux_weights = np.empty(self.N)

            for i in range(self.N):
                # výpočet μ
                char_particle = self._propagate_particle(current_particles[i], Q_np)
                
                # Ohodnotíme ji podle nového měření z_k a uložíme si likelihood
                char_likelihoods[i] = self._compute_likelihood(char_particle, y_seq_np[k], R_np)
                
                # Spočítáme pomocnou váhu: w_pomocná ∝ w_{k-1} * p(z_k|μ_i)
                aux_weights[i] = current_weights[i] * char_likelihoods[i]

            # Normalizujeme pomocné váhy
            aux_weights = self._normalize_weights(aux_weights)

            # --- APF Fáze 2: Resampling "nadějných rodičů" ---
            # "Use weights w_i from step 1 in a resampling step... store the indices i^j"
            parent_indices = self._multinomial_resample_indices(aux_weights)

            # --- APF Fáze 3 a 4: Finální predikce a výpočet vah ---
            # "Perform a prediction step for each of the... indices... Compute the weights..."
            
            new_particles = np.empty((self.N, self.state_dim))
            new_weights = np.empty(self.N)

            for i in range(self.N):
                # Získáme index vybraného "rodiče"
                # neiteruje se skrz všechny částice, ale jen přes vybrané indexy
                parent_index = parent_indices[i]
                
                # Provedeme plnou stochastickou predikci z tohoto rodiče
                new_particles[i] = self._propagate_particle(current_particles[parent_index], Q_np)
                
                # Vypočítáme finální likelihood pro novou částici
                final_likelihood = self._compute_likelihood(new_particles[i], y_seq_np[k], R_np)
                
                char_likelihood_of_parent = char_likelihoods[parent_index]
                if char_likelihood_of_parent < 1e-10:
                    char_likelihood_of_parent = 1e-10
                    
                # Vypočítáme finální váhu podle rovnice (19) v článku: w_k ∝ p(z_k|x_k) / p(z_k|μ)
                new_weights[i] = final_likelihood / char_likelihood_of_parent
            
            # Přepíšeme staré částice a váhy novými, normalizovanými
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