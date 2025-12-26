import torch
import numpy as np
from scipy.stats import multivariate_normal

class ParticleFilterNCLT:
    """
    Robustní implementace SIR Particle Filtru používající Log-Likelihood
    pro numerickou stabilitu při práci s vícerozměrnými nebo velmi přesnými senzory.
    """
    
    def __init__(self, system_model, num_particles=2000):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        
        # Převedení tenzorů modelu na numpy pro PF (PF běží efektivněji na CPU pro indexing)
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np_default = system_model.Q.cpu().numpy()
        self.R_np_default = system_model.R.cpu().numpy()
        
        if self.N <= 0:
            raise ValueError("Počet částic musí být kladné celé číslo.")

    def _propagate_vectorized(self, particles, Q_np, u_current_np=None):
        """
        Vektorizovaná predikce (Time Update).
        f() je voláno přes Torch (pokud je NN), šum je přidán v Numpy.
        """
        # Převod částic na torch pro průchod dynamickým modelem (f)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        
        if u_current_np is not None:
            u_torch = torch.from_numpy(u_current_np).float().to(self.device)
            # Předpokládáme broadcasting u_torch pro N částic
            predicted_particles_torch = self.f(particles_torch, u_torch)
        else:
            predicted_particles_torch = self.f(particles_torch)
            
        predicted_particles = predicted_particles_torch.detach().cpu().numpy()
        
        # Přidání procesního šumu (aditivní)
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), Q_np, size=self.N)
        return predicted_particles + noise

    def _compute_log_likelihood_vectorized(self, particles, z_np, R_np):
        # 1. OŠETŘENÍ NUL: Pokud '0.0' znamená chybějící data, přidej podmínku!
        # Předpokládám, že pro NCLT souřadnice 0.0, 0.0 nejsou validní, pokud jsi v terénu.
        # Pokud používáš NaN pro chybějící data, tato řádka je OK. 
        # Pokud máš nuly, změň to např. na: mask = ~np.isnan(z_np) & (np.abs(z_np) > 1e-6)
        mask = ~np.isnan(z_np)
        
        if not mask.any():
            return np.zeros(self.N)
            
        z_valid = z_np[mask]
        R_valid = R_np[np.ix_(mask, mask)] 

        particles_torch = torch.from_numpy(particles).float().to(self.device)
        expected_measurements_torch = self.h(particles_torch)
        expected_measurements = expected_measurements_torch.detach().cpu().numpy()
        expected_valid = expected_measurements[:, mask]
        
        # Ochrana proti singulární matici R (pro jistotu)
        try:
            log_likelihoods = multivariate_normal.logpdf(expected_valid, mean=z_valid, cov=R_valid)
        except np.linalg.LinAlgError:
            # Fallback pro numerickou nestabilitu
            return np.full(self.N, -np.inf)
        
        return log_likelihoods

    def _weights_from_log_weights(self, log_weights):
        max_log = np.max(log_weights)
        
        # OPRAVA: Ošetření případu, kdy jsou všechny váhy -inf
        if np.isinf(max_log):
            # Všechny částice mají nulovou pravděpodobnost -> Particle Deprivation.
            # Řešení: Resetovat na uniformní váhy (ignorovat měření) 
            # nebo (pokročilé) injektovat nové částice kolem měření.
            # Pro teď resetujeme na uniformní:
            return np.full(self.N, 1.0 / self.N)
            
        # exp(log_w - max) = w / exp(max)
        weights_unnormalized = np.exp(log_weights - max_log)
        
        sum_w = np.sum(weights_unnormalized)
        if sum_w < 1e-15:
            return np.full(self.N, 1.0 / self.N)
            
        return weights_unnormalized / sum_w

    def _systematic_resample(self, particles, weights):
        """Standardní systematický resampling."""
        N = len(particles)
        # Kumulativní součet vah
        Q = np.cumsum(weights)
        Q[-1] = 1.0  # Ujištění proti float chybám
        
        u0 = np.random.uniform(0.0, 1.0 / N)
        # Vektorizovaný resampling indexů
        points = u0 + np.arange(N) / N
        indices = np.searchsorted(Q, points)
        
        return particles[indices].copy() # Copy je důležité, aby se nerozbila paměť

    def _weights_from_log_weights(self, log_weights):
        """
        Log-Sum-Exp trik pro převod log-vah na normalizované lineární váhy.
        Zabraňuje podtečení (underflow).
        """
        max_log = np.max(log_weights)
        # Odečtení maxima (posun do stabilní oblasti, např. z -700 na 0)
        # exp(log_w - max) = w / exp(max)
        weights_unnormalized = np.exp(log_weights - max_log)
        
        sum_w = np.sum(weights_unnormalized)
        if sum_w < 1e-15:
            # Fallback pro extrémní degeneraci
            return np.full(self.N, 1.0 / self.N)
            
        return weights_unnormalized / sum_w

    def get_estimate(self, particles, weights):
        """Vypočte vážený průměr a kovarianci."""
        mean = np.average(particles, weights=weights, axis=0)
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        
        mean_torch = torch.from_numpy(mean).float().to(self.device)
        cov_torch = torch.from_numpy(cov).float().to(self.device)
        return mean_torch.reshape(self.state_dim, 1), cov_torch

    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None, Q=None, R=None, resampling_threshold=0.5):
        seq_len = y_seq.shape[0]
        y_seq_np = y_seq.cpu().numpy()
        u_sequence_np = u_sequence.cpu().numpy() if u_sequence is not None else None
        
        # Flexibilní parametry (pokud nejsou zadány, berou se z modelu)
        Q_np = Q.cpu().numpy() if Q is not None else self.Q_np_default
        R_np = R.cpu().numpy() if R is not None else self.R_np_default
        Ex0_np = Ex0.cpu().numpy().flatten() if Ex0 is not None else self.Ex0_np
        P0_np = P0.cpu().numpy() if P0 is not None else self.P0_np
            
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        # Inicializace mraku částic
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        # Log-váhy inicializujeme jako 0 (protože log(1/N) je konstanta, stačí relativní)
        # Ale pro resampling potřebujeme lineární. Začneme s uniformními.
        current_weights = np.full(self.N, 1.0 / self.N)
        
        # Uložení počátečního stavu
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        
        n_threshold = self.N * resampling_threshold

        for k in range(1, seq_len):
            u_k_np = u_sequence_np[k, :] if u_sequence_np is not None else None
            
            # 1. PREDIKCE (Propagace)
            # Částice se pohnou, váhy zůstávají stejné (pokud nepoužíváme proposal distribution q != p)
            current_particles = self._propagate_vectorized(current_particles, Q_np, u_current_np=u_k_np)
            
            # 2. UPDATE (Věrohodnost)
            y_k = y_seq_np[k]
            
            if not np.all(np.isnan(y_k)):
                # Získáme log likelihoods: log p(y|x)
                log_likelihoods = self._compute_log_likelihood_vectorized(current_particles, y_k, R_np)
                
                # Update vah: log(w_new) = log(w_prev) + log(likelihood)
                # Protože po resamplingu jsou váhy uniformní (1/N), můžeme jen přičíst likelihood
                # Pokud by nebyl resampling v každém kroku, museli bychom přičítat k log(current_weights)
                # Pro jistotu pracujeme s logaritmem předchozích vah:
                log_weights_prev = np.log(current_weights + 1e-300)
                log_weights_new = log_weights_prev + log_likelihoods
                
                # Normalizace zpět do lineárního prostoru (Log-Sum-Exp)
                current_weights = self._weights_from_log_weights(log_weights_new)
            
            # 3. ODHAD STAVU (před resamplingem pro lepší přesnost, ale lze i po)
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est

            # 4. RESAMPLING
            # Efektivní počet částic
            effective_N = 1.0 / np.sum(current_weights**2)
            
            if effective_N < n_threshold:
                current_particles = self._systematic_resample(current_particles, current_weights)
                # Po resamplingu mají všechny částice stejnou váhu
                current_weights.fill(1.0 / self.N)
        
        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history
        }