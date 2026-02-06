import torch
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

class VectorizedAuxiliaryParticleFilter:
    """
    Plně vektorizovaná implementace Auxiliary Particle Filtru (APF).
    """
    
    def __init__(self, system_model, num_particles=1000):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        # Pozor: Předpokládáme, že H mapuje na obs_dim. Získáme ho z R.
        self.obs_dim = system_model.R.shape[0] 
        
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np_default = system_model.Q.cpu().numpy()
        self.R_np_default = system_model.R.cpu().numpy()
        
        # Pro vektorizaci likelihoodu si připravíme fixní objekt pro R (pokud je R konst.)
        # Pokud se R mění, budeme ho muset tvořit v každém kroku.
        self.mvn_R = multivariate_normal(mean=np.zeros(self.obs_dim), cov=self.R_np_default)

    def _propagate_particles(self, particles, Q_np):
        """
        VEKTORIZOVANÁ propagace.
        Vstup: particles (N, state_dim)
        Výstup: propagated_particles (N, state_dim)
        """
        # 1. Dávkový převod na torch (N, state_dim)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        
        # 2. Aplikace modelu f na celý batch najednou
        # Předpokládá, že self.f umí pracovat s (N, dim), což NN obvykle umí
        with torch.no_grad():
            predicted_particles_torch = self.f(particles_torch)
        
        predicted_particles = predicted_particles_torch.cpu().numpy()
        
        # 3. Generování šumu pro všechny částice najednou
        # size=self.N zajistí vygenerování matice (N, state_dim)
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np, size=self.N)
        
        return predicted_particles + noise

    def _compute_likelihoods(self, particles, z_np, R_np):
        """
        VEKTORIZOVANÝ výpočet likelihoodu.
        Vstup: particles (N, state_dim), z_np (obs_dim,)
        Výstup: likelihoods (N,)
        """
        # 1. Transformace všech částic do prostoru měření h(x)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        
        with torch.no_grad():
            expected_measurements_torch = self.h(particles_torch) # (N, obs_dim)
        
        expected_measurements = expected_measurements_torch.cpu().numpy()
        
        # 2. Výpočet reziduí (inovací) pro všechny částice
        # Broadcasting: (obs_dim,) - (N, obs_dim) -> (N, obs_dim)
        residuals = z_np - expected_measurements
        
        # 3. Vyhodnocení PDF
        # Matematický trik: p(z | x) ~ N(z; h(x), R) je ekvivalentní N(z - h(x); 0, R)
        # Tím pádem můžeme použít jedno rozdělení N(0, R) a jen do něj sypat rezidua.
        # Pokud je R konstantní, použijeme self.mvn_R, jinak vytvoříme nové.
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
        Zůstává stejné, už to bylo efektivní.
        """
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        random_picks = np.random.rand(self.N)
        indexes = np.searchsorted(cumulative_sum, random_picks)
        return indexes

    def get_estimate(self, particles, weights):
        """
        Vektorizovaný výpočet váženého průměru a kovariance.
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
            
        # Pokud se R liší od defaultu, aktualizujeme mvn objekt pro rychlost
        if not np.array_equal(R_np, self.R_np_default):
             self.mvn_R = multivariate_normal(mean=np.zeros(self.obs_dim), cov=R_np)

        # Pre-alokace historie pro rychlost
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        particles_history = []
        
        # Inicializace
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        current_weights = np.full(self.N, 1.0 / self.N)
        
        # Uložení počátečního stavu
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        particles_history.append(current_particles.copy())

        # HLAVNÍ SMYČKA
        for k in tqdm(range(1, seq_len), desc="APF Vectorized"):
            
            # --- APF Fáze 1: Charakterizace (Mean predikce) ---
            # Vektorizovaně spočítáme "mu" pro všechny částice
            char_particles = self._propagate_particles(current_particles, Q_np)
            
            # Vektorizovaně spočítáme p(z_k | mu)
            char_likelihoods = self._compute_likelihoods(char_particles, y_seq_np[k], R_np)
            
            # Pomocné váhy
            aux_weights = current_weights * char_likelihoods
            aux_weights = self._normalize_weights(aux_weights)

            # --- APF Fáze 2: Resampling indexů ---
            parent_indices = self._multinomial_resample_indices(aux_weights)
            
            # Vybereme rodiče pomocí "Fancy Indexing" (rychlé)
            # parents shape: (N, state_dim)
            parents = current_particles[parent_indices]
            
            # Potřebujeme i likelihoods rodičů pro jmenovatele ve fázi 4
            # parent_char_likelihoods shape: (N,)
            parent_char_likelihoods = char_likelihoods[parent_indices]

            # --- APF Fáze 3: Finální Propagace ---
            # Propagujeme vybrané rodiče (plný stochastický krok)
            new_particles = self._propagate_particles(parents, Q_np)

            # --- APF Fáze 4: Váhy ---
            # p(z_k | x_k)
            final_likelihoods = self._compute_likelihoods(new_particles, y_seq_np[k], R_np)
            
            # Ošetření dělení nulou
            safe_denom = np.maximum(parent_char_likelihoods, 1e-10)
            
            # w_k = p(z|x) / p(z|mu)
            new_weights = final_likelihoods / safe_denom
            
            # Normalizace a update
            current_weights = self._normalize_weights(new_weights)
            current_particles = new_particles
            
            # Ukládání (volitelné: particles_history bere hodně paměti, u velkých N zvaž vypnout)
            particles_history.append(current_particles.copy())
            
            # Odhad stavu
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            'particles_history': particles_history
        }