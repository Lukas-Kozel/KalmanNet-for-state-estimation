import torch
import torch.distributions as dist
import numpy as np

class AuxiliaryParticleFilterNCLT:
    """
    Adaptive Auxiliary Particle Filter (AAPF) s optimalizací pro ANEES a MSE.
    
    Klíčové vlastnosti:
    1. Adaptive Resampling: Resampluje jen při nízkém N_eff (zabraňuje kolapsu).
    2. Direct Roughening (Jitter): Udržuje diverzitu částic i při výpadku GPS.
    3. R-Inflation: Zvyšuje robustnost vah vůči příliš sebevědomým měřením.
    """
    
    def __init__(self, system_model, num_particles=2000, jitter_strength=2.0, r_inflation=20.0, resample_threshold=0.5):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h
        self.N = num_particles
        self.state_dim = system_model.Q.shape[0]
        self.obs_dim = system_model.R.shape[0]
        
        # Hyperparametry
        self.jitter_strength = jitter_strength      # Síla rozostření (proti ANEES)
        self.r_inflation = r_inflation              # Nafouknutí R (proti ANEES)
        self.resample_threshold = resample_threshold # Práh pro adaptivní resampling (např. 0.5 * N)
        
        self.Ex0 = system_model.Ex0
        self.P0 = system_model.P0
        self.Q = system_model.Q
        self.R = system_model.R
        
        # Distribuce pro procesní šum
        self.dist_process_noise = dist.MultivariateNormal(
            torch.zeros(self.state_dim, device=self.device), 
            covariance_matrix=self.Q
        )
        
        # Záchranná brzda pro kovarianci (velmi malá, jen pro stabilitu)
        self.min_cov_matrix = torch.diag(torch.diag(self.Q)) * 0.01

        print(f"Initialized Adaptive APF: N={self.N}, Jitter={self.jitter_strength}, R-Infl={self.r_inflation}, ResampleThr={self.resample_threshold}")

    def _compute_log_likelihood_vectorized(self, particles, z, R):
        """Vypočte log-likelihood s nafouknutým R."""
        pred_z = self.h(particles)
        
        # Použití R_inflation pro "zploštění" vah (více částic přežije)
        inflated_R = R * self.r_inflation
        
        meas_dist = dist.MultivariateNormal(loc=pred_z, covariance_matrix=inflated_R)
        log_lik = meas_dist.log_prob(z.view(1, -1))
        return log_lik

    def _normalize_log_weights(self, log_weights):
        """Stabilní softmax."""
        max_log = torch.max(log_weights)
        if torch.isinf(max_log):
            return torch.full((self.N,), 1.0 / self.N, device=self.device)
        weights_unnorm = torch.exp(log_weights - max_log)
        weights = weights_unnorm / torch.sum(weights_unnorm)
        return weights

    def _systematic_resample(self, weights):
        """Systematic Resampling (GPU optimized)."""
        cumsum = torch.cumsum(weights, dim=0)
        cumsum[-1] = 1.0 + 1e-6 
        random_offset = torch.rand(1, device=self.device)
        positions = (torch.arange(self.N, device=self.device, dtype=torch.float32) + random_offset) / self.N
        indexes = torch.searchsorted(cumsum, positions)
        indexes = torch.clamp(indexes, 0, self.N - 1)
        return indexes

    def get_estimate(self, particles, weights):
        """Vrátí vážený průměr a kovarianci."""
        mean = torch.sum(particles * weights.unsqueeze(1), dim=0)
        diff = particles - mean
        cov = (diff.T * weights) @ diff
        
        # Přičtení malé záchranné podlahy (jen pro numerickou stabilitu)
        cov = cov + self.min_cov_matrix
        return mean.reshape(self.state_dim, 1), cov

    def apply_jitter(self, particles):
        """
        Direct Roughening: Aplikuje extra šum pro udržení diverzity.
        Používá se v každém kroku (Outage, SIS, APF).
        """
        if self.jitter_strength > 0:
            # Šum odvozený od procesního šumu Q
            noise = torch.randn_like(particles) @ torch.linalg.cholesky(self.Q).T
            return particles + noise * self.jitter_strength
        return particles

    def process_sequence(self, y_seq, u_sequence=None, Ex0=None, P0=None):
        # --- 1. Konverze dat na GPU ---
        if not isinstance(y_seq, torch.Tensor):
            y_seq = torch.from_numpy(y_seq).float().to(self.device)
        else:
            y_seq = y_seq.float().to(self.device)
            
        if u_sequence is not None:
            if not isinstance(u_sequence, torch.Tensor):
                u_seq_torch = torch.from_numpy(u_sequence).float().to(self.device)
            else:
                u_seq_torch = u_sequence.float().to(self.device)
        else:
            u_seq_torch = None
        
        seq_len = y_seq.shape[0]
        
        # --- 2. Inicializace ---
        if Ex0 is not None:
            start_mean = Ex0 if isinstance(Ex0, torch.Tensor) else torch.from_numpy(Ex0).float().to(self.device)
        else:
            start_mean = self.Ex0
        if P0 is not None:
            start_cov = P0 if isinstance(P0, torch.Tensor) else torch.from_numpy(P0).float().to(self.device)
        else:
            start_cov = self.P0
            
        # Alokace historie
        x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        # Inicializace částic
        init_dist = dist.MultivariateNormal(start_mean.squeeze(), start_cov)
        current_particles = init_dist.sample((self.N,))
        current_weights = torch.full((self.N,), 1.0 / self.N, device=self.device)
        
        # Uložení počátečního stavu
        x_est, P_est = self.get_estimate(current_particles, current_weights)
        x_filtered_history[0] = x_est.squeeze()
        P_filtered_history[0] = P_est

        # --- 3. Hlavní smyčka ---
        for k in range(1, seq_len):
            u_k = u_seq_torch[k] if u_seq_torch is not None else None
            y_k = y_seq[k]
            
            # Detekce platného měření (není NaN)
            is_measurement_valid = not torch.isnan(y_k).any()

            if not is_measurement_valid:
                # ==============================
                # A. OUTAGE MODE (Výpadek GPS)
                # ==============================
                # Pouze propagace dynamiky + difuze šumu
                
                # 1. Propagace f(x, u)
                if u_k is not None:
                    u_k_batch = u_k.unsqueeze(0).expand(self.N, -1)
                    pred_particles = self.f(current_particles, u_k_batch)
                else:
                    pred_particles = self.f(current_particles)
                
                # 2. Add Process Noise (Q)
                noise = self.dist_process_noise.sample((self.N,))
                current_particles = pred_particles + noise
                
                # 3. Direct Roughening (Důležité pro růst P během výpadku)
                current_particles = self.apply_jitter(current_particles)
                
                # Váhy se nemění (neseme si informaci z minula)
                
            else:
                # ==============================
                # B. MEASUREMENT MODE
                # ==============================
                
                # Spočítat N_eff (Effective Sample Size)
                # N_eff = 1 / sum(w^2)
                n_eff = 1.0 / torch.sum(current_weights ** 2)
                
                # Adaptivní rozhodnutí: Resamplovat?
                if n_eff < (self.N * self.resample_threshold):
                    # ----------------------------------------
                    # B1. APF UPDATE (Resampling nutný)
                    # ----------------------------------------
                    # Používáme Auxiliary PF pro lepší nasměrování částic
                    
                    # 1. Lookahead (kam by částice dopluly?)
                    if u_k is not None:
                        u_k_batch = u_k.unsqueeze(0).expand(self.N, -1)
                        mu_particles = self.f(current_particles, u_k_batch)
                    else:
                        mu_particles = self.f(current_particles)
                    
                    # 2. Auxiliary Weights
                    log_weights_prev = torch.log(current_weights + 1e-30)
                    log_lik_aux = self._compute_log_likelihood_vectorized(mu_particles, y_k, self.R)
                    log_weights_aux = log_weights_prev + log_lik_aux
                    weights_aux = self._normalize_log_weights(log_weights_aux)
                    
                    # 3. Resampling
                    parent_indices = self._systematic_resample(weights_aux)
                    parents = current_particles[parent_indices]
                    
                    # 4. Propagace rodičů
                    if u_k is not None:
                        pred_parents = self.f(parents, u_k_batch)
                    else:
                        pred_parents = self.f(parents)
                        
                    noise = self.dist_process_noise.sample((self.N,))
                    current_particles = pred_parents + noise
                    
                    # 5. Jitter (po resamplingu kritické proti ochuzení)
                    current_particles = self.apply_jitter(current_particles)

                    # 6. Korekce vah (APF Weight Update)
                    # w_new = p(y|x) / p(y|mu_parent)
                    log_lik_final = self._compute_log_likelihood_vectorized(current_particles, y_k, self.R)
                    log_lik_aux_parents = log_lik_aux[parent_indices]
                    
                    log_weights_new = log_lik_final - log_lik_aux_parents
                    current_weights = self._normalize_log_weights(log_weights_new)
                    
                else:
                    # ----------------------------------------
                    # B2. SIS UPDATE (Resampling není nutný)
                    # ----------------------------------------
                    # Klasický sekvenční update (jen propagace a vážení)
                    # Udržuje historii a diverzitu částic
                    
                    # 1. Propagace
                    if u_k is not None:
                        u_k_batch = u_k.unsqueeze(0).expand(self.N, -1)
                        pred_particles = self.f(current_particles, u_k_batch)
                    else:
                        pred_particles = self.f(current_particles)
                        
                    noise = self.dist_process_noise.sample((self.N,))
                    current_particles = pred_particles + noise
                    
                    # 2. Jitter (i zde pomáhá)
                    current_particles = self.apply_jitter(current_particles)
                    
                    # 3. Standardní update vah
                    # w_new = w_prev * p(y|x)
                    log_weights_prev = torch.log(current_weights + 1e-30)
                    log_lik = self._compute_log_likelihood_vectorized(current_particles, y_k, self.R)
                    
                    log_weights_new = log_weights_prev + log_lik
                    current_weights = self._normalize_log_weights(log_weights_new)

            # Odhad a uložení
            x_est, P_est = self.get_estimate(current_particles, current_weights)
            x_filtered_history[k] = x_est.squeeze()
            P_filtered_history[k] = P_est
            
        return {'x_filtered': x_filtered_history, 'P_filtered': P_filtered_history}