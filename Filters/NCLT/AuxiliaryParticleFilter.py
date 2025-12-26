import torch
import numpy as np
from scipy.stats import multivariate_normal

class AuxiliaryParticleFilterNCLT:
    """
    Vektorizovaný Auxiliary Particle Filter (APF) s podporou NaN a Control Input.
    Vhodný pro NCLT dataset.
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
        Vrátí log(p(z|x)) pro všechny částice najednou.
        """
        mask = ~np.isnan(z_np)
        if not mask.any():
            # Pokud je měření NaN, likelihood je uniformní (log(1) = 0, ale relativně je to jedno)
            # Vracíme nuly, což znamená, že nepřidáváme žádnou informaci.
            return np.zeros(self.N)

        z_valid = z_np[mask]
        R_valid = R_np[np.ix_(mask, mask)]

        # Predikce měření (h)
        particles_torch = torch.from_numpy(particles).float().to(self.device)
        expected_measurements = self.h(particles_torch).detach().cpu().numpy()
        expected_valid = expected_measurements[:, mask]

        # Výpočet Log-PDF
        try:
            log_lik = multivariate_normal.logpdf(expected_valid, mean=z_valid, cov=R_valid)
        except np.linalg.LinAlgError:
            return np.full(self.N, -np.inf)
            
        return log_lik

    def _normalize_log_weights(self, log_weights):
        """Stabilní softmax: exp(log_w - max) / sum(...)"""
        max_log = np.max(log_weights)
        if np.isinf(max_log):
            return np.full(self.N, 1.0 / self.N)
            
        weights_unnorm = np.exp(log_weights - max_log)
        weights = weights_unnorm / np.sum(weights_unnorm)
        return weights

    def _systematic_resample(self, weights):
        """Vrací indexy rodičů."""
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
        # 1. Klasický výpočet mean
        mean = np.average(particles, weights=weights, axis=0)
        
        # 2. Výpočet kovariance
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        
        # === OPRAVA ANEES: Regularizace Kovariance ===
        # Problém: U PF se stává, že všechny částice jsou identické (sample impoverishment),
        # pak je cov = 0. ANEES dělí nulou -> vybuchne.
        # Řešení: Přičteme malou "safety" diagonálu. 
        # Hodnota 1e-3 znamená, že si jsme jistí maximálně na milimetry, ne více.
        
        min_cov_val = 1e-3 
        cov = cov + np.eye(self.state_dim) * min_cov_val
        
        # Převod na torch
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
        # Poznámka: APF v kroku k potřebuje váhy z kroku k-1.
        current_particles = np.random.multivariate_normal(Ex0_np, P0_np, size=self.N)
        # Váhy držíme v lineárním prostoru, protože resampling je potřebuje lineární
        current_weights = np.full(self.N, 1.0 / self.N)
        
        # Uložení počátečního stavu
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est

        for k in range(1, seq_len):
            u_k = u_seq_np[k] if u_seq_np is not None else None
            y_k = y_seq_np[k]
            is_measurement_valid = not np.all(np.isnan(y_k))

            # --- APF FÁZE 1: Lookahead & Auxiliary Weights ---
            
            # 1. Deterministická predikce průměru (mu) pro všechny částice
            # mu_k^i = f(x_{k-1}^i, u_k)
            parts_torch = torch.from_numpy(current_particles).float().to(self.device)
            if u_k is not None:
                u_torch = torch.from_numpy(u_k).float().to(self.device).reshape(1, -1).repeat(self.N, 1)
                mu_particles_torch = self.f(parts_torch, u_torch)
            else:
                mu_particles_torch = self.f(parts_torch)
            
            mu_particles = mu_particles_torch.detach().cpu().numpy()

            # 2. Výpočet pomocných vah
            # Pokud máme měření: log_w_aux = log(w_{k-1}) + log p(y_k | mu_k)
            # Pokud nemáme měření: log_w_aux = log(w_{k-1})  (žádná nová info)
            
            log_weights_prev = np.log(current_weights + 1e-300)
            
            if is_measurement_valid:
                log_lik_aux = self._compute_log_likelihood_vectorized(mu_particles, y_k, self.R_np)
                log_weights_aux = log_weights_prev + log_lik_aux
            else:
                # Pokud není měření, pomocná váha je jen původní váha
                log_weights_aux = log_weights_prev
                
            # Normalizace pro resampling
            weights_aux = self._normalize_log_weights(log_weights_aux)

            # --- APF FÁZE 2: Resampling ---
            # Vybereme indexy "nadějných" částic
            parent_indices = self._systematic_resample(weights_aux)
            
            # Vybereme rodičovské částice
            parents = current_particles[parent_indices]
            
            # --- APF FÁZE 3: Finální propagace ---
            # x_k^j ~ N(f(parents^j, u_k), Q)
            # Musíme znovu prohnat modelem f() ty VYBRANÉ rodiče (vektorizovaně)
            
            parents_torch = torch.from_numpy(parents).float().to(self.device)
            if u_k is not None:
                # u_torch už máme, jen zkontrolujeme shape
                u_torch = torch.from_numpy(u_k).float().to(self.device).reshape(1, -1).repeat(self.N, 1)
                pred_parents_torch = self.f(parents_torch, u_torch)
            else:
                pred_parents_torch = self.f(parents_torch)
                
            pred_parents = pred_parents_torch.detach().cpu().numpy()
            
            # Přidání šumu Q
            noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.Q_np, size=self.N)
            current_particles = pred_parents + noise

            # --- APF FÁZE 4: Korekce vah ---
            # Váha w_k^j = p(y_k | x_k^j) / p(y_k | mu_{parent})
            # V logaritmech: log_w_new = log_lik_final - log_lik_aux_parent
            
            if is_measurement_valid:
                # 1. Likelihood finálních částic
                log_lik_final = self._compute_log_likelihood_vectorized(current_particles, y_k, self.R_np)
                
                # 2. Likelihood rodičů (ten už máme z Fáze 1, jen musíme vzít správné indexy)
                log_lik_aux_parents = log_lik_aux[parent_indices]
                
                # 3. Výpočet váhy (korekce biasu zavedeného resamplingem v kroku 2)
                log_weights_new = log_lik_final - log_lik_aux_parents
                
                # Ošetření: Vzhledem k šumu Q se může stát, že x_k je 'mimo', zatímco mu_k bylo 'ok'
                # nebo naopak. APF váhy by měly být kolem 1.0 (log 0).
            else:
                # Pokud nemáme měření, váhy jsou uniformní (1/N)
                # Protože jsme provedli resampling podle w_{k-1}, částice jsou již distribuovány
                # podle posterioru z minula. Nyní jen difundují šumem Q.
                log_weights_new = np.zeros(self.N) # log(1) = 0, po normalizaci to bude 1/N

            current_weights = self._normalize_log_weights(log_weights_new)
            
            # Odhad a uložení
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est
            
        return {'x_filtered': x_filtered_history, 'P_filtered': P_filtered_history}