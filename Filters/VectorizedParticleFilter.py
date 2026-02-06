import torch
import numpy as np
from tqdm import tqdm

class VectorizedParticleFilter:
    """
    Plně vektorizovaná implementace SIR (Sequential Importance Resampling) Particle Filtru.
    Běží kompletně na GPU (PyTorch) pro maximální výkon.
    """
    
    def __init__(self, system_model, num_particles=1000):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        
        # Načtení parametrů přímo do Torch tensorů na Device
        self.state_dim = system_model.Q.shape[0]
        self.obs_dim = system_model.R.shape[0]
        
        # Uložíme si parametry jako tensory, abychom se vyhnuli přesunům v loopu
        self.Ex0 = system_model.Ex0.clone().detach().flatten().to(self.device)
        self.P0 = system_model.P0.clone().detach().to(self.device)
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        
        # --- Optimalizace Likelihoodu (Předpočítání inverze a determinantu) ---
        # Log-Likelihood: -0.5 * (res.T @ R_inv @ res)
        # Používáme Choleskyho rozklad pro numerickou stabilitu, pokud je R pozitivně definitní
        try:
            self.L_R = torch.linalg.cholesky(self.R)
            # R_inv potřebujeme pro Mahalanobisovu vzdálenost
            self.R_inv = torch.cholesky_inverse(self.L_R)
        except RuntimeError:
            # Fallback pro případ, že R není PD (např. nuly na diagonále)
            self.R_inv = torch.inverse(self.R)

        # Distribuční objekty pro procesní šum (vytvoříme jednou)
        self.dist_Q = torch.distributions.MultivariateNormal(
            torch.zeros(self.state_dim, device=self.device), 
            self.Q
        )
        
        # Startovní distribuce
        self.dist_init = torch.distributions.MultivariateNormal(self.Ex0, self.P0)

        self.particles = None
        self.weights = None
        
        if self.N <= 0:
            raise ValueError("Počet částic (num_particles) musí být kladné celé číslo.")

    def _propagate_vectorized(self, particles, Q=None):
        """
        Vektorizovaná predikce pro celý mrak částic na GPU.
        """
        # 1. Deterministická část (Model f)
        # Předpokládáme, že f umí batch (N, dim)
        predicted_particles = self.f(particles)
        
        # 2. Stochastická část (Šum)
        # Generujeme šum přímo na GPU
        if Q is None:
            noise = self.dist_Q.sample((self.N,))
        else:
            # Pokud se Q mění dynamicky (např. zvenčí), musíme vytvořit novou distribuci
            # Ale pokud je to jen tensor Q na GPU, použijeme ho.
            # Pro rychlost předpokládáme standardní Q, pokud není zadáno jinak.
            # Zde pro kompatibilitu s voláním:
            noise = torch.distributions.MultivariateNormal(
                torch.zeros(self.state_dim, device=self.device), Q
            ).sample((self.N,))
            
        return predicted_particles + noise

    def _compute_likelihood_vectorized(self, particles, z, R=None):
        """
        Vektorizovaný výpočet likelihood na GPU pomocí log-likelihood.
        """
        # 1. Transformace do prostoru měření
        expected_measurements = self.h(particles) # (N, obs_dim)
        
        # 2. Rezidua (z - h(x))
        # z shape: (obs_dim,), expected shape: (N, obs_dim) -> broadcasting funguje
        residuals = z - expected_measurements 
        
        # 3. Mahalanobis distance (vektorizovaně)
        # (res @ R_inv) * res -> sum over dim
        
        # Pokud je R zadané externě a liší se, musíme přepočítat inverzi
        if R is not None and R is not self.R:
             R_inv_local = torch.inverse(R)
             term1 = residuals @ R_inv_local
        else:
             term1 = residuals @ self.R_inv

        mahalanobis = (term1 * residuals).sum(dim=1) # (N,)
        
        # 4. Log Likelihood (konstanty pro normalizaci vah nejsou potřeba, vykrátí se)
        log_weights = -0.5 * mahalanobis
        
        # Převedeme zpět na pravděpodobnost (s ošetřením underflow pomocí max triku)
        # To se obvykle děje až při update vah, ale pro zachování struktury vracíme likelihoods
        # Zde vracíme log_likelihoods, protože je to bezpečnější, 
        # ale metoda se jmenuje compute_likelihood, tak vrátíme exp().
        # Aby to nebylo 0, odečteme max (trik se udělá v update kroku, tady vrátíme raw exp)
        
        return torch.exp(log_weights)

    def _vectorized_systematic_resample(self, particles, weights):
        """
        Extrémně rychlý Systematic Resampling pomocí PyTorch operací.
        """
        # 1. Kumulativní suma vah (CDF)
        cumsum = torch.cumsum(weights, dim=0)
        cumsum[-1] = 1.0  # Ošetření numerické chyby
        
        # 2. Generování systematických bodů
        step = 1.0 / self.N
        u0 = torch.rand(1, device=self.device).item() * step
        u = torch.arange(self.N, device=self.device) * step + u0
        
        # 3. Nalezení indexů (searchsorted na GPU)
        indices = torch.searchsorted(cumsum, u)
        
        # Ošetření přetečení indexů (pro jistotu)
        indices = torch.clamp(indices, 0, self.N - 1)
        
        # 4. Výběr částic (Gather/Indexing)
        new_particles = particles[indices]
        
        return new_particles
    
    def _normalize_weights(self, weights):
        sum_weights = torch.sum(weights)
        if sum_weights < 1e-15:
            # Degenerace -> Uniformní reset
            weights.fill_(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def get_estimate(self, particles, weights):
        """Vektorizovaný odhad na GPU."""
        # Mean: vážený průměr
        mean = torch.sum(particles * weights.unsqueeze(1), dim=0) # (dim,)
        
        # Covariance
        diff = particles - mean # (N, dim)
        # (dim, N) @ (N, dim) -> (dim, dim)
        cov = (diff.T * weights) @ diff
        
        return mean.reshape(self.state_dim, 1), cov
    
    def process_sequence(self, y_seq, Ex0=None, P0=None, Q=None, R=None, resampling_threshold=0.5):
        # Zajistíme, že data jsou na GPU
        if y_seq.device != self.device:
            y_seq = y_seq.to(self.device)
            
        seq_len = y_seq.shape[0]

        # Pokud jsou Q a R None, použijeme interní self.Q/self.R (které jsou už na GPU)
        # Pokud jsou zadané, musíme zajistit, že jsou to tensory na GPU
        if Q is not None and not torch.is_tensor(Q):
             Q = torch.tensor(Q, device=self.device).float()
        elif Q is not None and Q.device != self.device:
             Q = Q.to(self.device)
             
        if R is not None and not torch.is_tensor(R):
             R = torch.tensor(R, device=self.device).float()
        elif R is not None and R.device != self.device:
             R = R.to(self.device)

        # Inicializace stavu
        start_mean = Ex0.flatten().to(self.device) if Ex0 is not None else self.Ex0
        start_cov = P0.to(self.device) if P0 is not None else self.P0
        
        # Pre-alokace historie (na GPU)
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        # particles_history = [] # Ukládání historie žere VRAM, zapni jen pokud nutné
        
        # Inicializace částic
        if Ex0 is not None or P0 is not None:
             current_particles = torch.distributions.MultivariateNormal(start_mean, start_cov).sample((self.N,))
        else:
             current_particles = self.dist_init.sample((self.N,))
             
        current_weights = torch.full((self.N,), 1.0 / self.N, device=self.device)
        
        # Uložení t=0
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est
        # particles_history.append(current_particles.cpu().numpy()) # Pro vizualizaci převod na CPU

        n_threshold = self.N * resampling_threshold

        # Vypneme gradienty pro maximální rychlost
        with torch.no_grad():
            for k in tqdm(range(1, seq_len), desc="Processing PF (GPU)"):
                
                # 1. Propagace
                # Pokud je Q None, použije se self.Q (rychlé), jinak externí Q
                current_particles = self._propagate_vectorized(current_particles, Q if Q is not None else None)
                
                # 2. Likelihood
                # Log-likelihood trick pro numerickou stabilitu
                # Spočítáme log likelihoody ručně (rychlejší než přes distributions)
                # Používáme metodu z _compute_likelihood_vectorized ale upravenou pro log
                
                expected_meas = self.h(current_particles)
                residuals = y_seq[k] - expected_meas
                
                # Mahalanobis
                curr_R = R if R is not None else self.R
                curr_R_inv = torch.inverse(curr_R) if R is not None else self.R_inv
                
                term1 = residuals @ curr_R_inv
                mahalanobis = (term1 * residuals).sum(dim=1)
                log_likelihoods = -0.5 * mahalanobis
                
                # Update vah v log prostoru (Log-Sum-Exp trik)
                # log(w_new) = log(w_old) + log(likelihood)
                # w_old je uniformní po resamplingu, takže jen přičteme likelihood
                # Pokud nebyl resampling, musíme přičíst log(current_weights)
                
                log_weights_unnorm = torch.log(current_weights + 1e-30) + log_likelihoods
                max_log = torch.max(log_weights_unnorm)
                
                # Zpět do lineárního prostoru bez underflow
                weights_unnorm = torch.exp(log_weights_unnorm - max_log)
                
                current_weights = self._normalize_weights(weights_unnorm)
                
                # 3. Resampling
                effective_N = 1.0 / torch.sum(current_weights**2)
                
                if effective_N < n_threshold:
                    current_particles = self._vectorized_systematic_resample(current_particles, current_weights)
                    current_weights.fill_(1.0 / self.N) # In-place fill je rychlejší
                
                # Ukládání
                # particles_history.append(current_particles.cpu().numpy()) 
                
                x_est, P_est = self.get_estimate(current_particles, current_weights)
                x_filtered_history[k] = x_est.squeeze()
                P_filtered_history[k] = P_est

        return {
            'x_filtered': x_filtered_history,
            'P_filtered': P_filtered_history,
            # 'particles_history': particles_history
        }