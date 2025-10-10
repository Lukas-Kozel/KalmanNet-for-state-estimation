import torch
import numpy as np
from scipy.stats import multivariate_normal
import copy

class ParticleFilterSIR:
    """
    Implementace SIR (Sequential Importance Resampling) Particle Filtru.
    Tato třída se co nejvěrněji drží logiky a struktury z tutoriálu
    "Particle Filters: A Hands-On Tutorial" (Elfring et al.).
    
    Cílem je přesně replikovat algoritmus popsaný v článku, zejména
    Algorithm 2: "Particle filter with resampling" (strana 14).
    """
    
    def __init__(self, system_model, num_particles=1000):
        """
        Inicializace SIR filtru.
        Zde nastavíme všechny potřebné parametry, které filtr bude používat.
        """
        self.device = system_model.Q.device
        
        # Obecné funkce modelu, které budeme volat.
        # V článku jsou tyto funkce označeny jako f_k a h_k v rovnicích (1) a (2) (strana 4).
        self.f = system_model.f  # Process model (model pohybu)
        self.h = system_model.h  # Measurement model (model měření)
        
        # Parametry filtru
        self.N = num_particles         # Počet částic, v článku označován jako N_s
        self.state_dim = system_model.Q.shape[0]
        
        # Uložíme si počáteční podmínky (prior) a matice šumů z modelu.
        # Tyto hodnoty budeme používat pro inicializaci a v průběhu filtrace.
        self.Ex0_np = system_model.Ex0.cpu().numpy().flatten()
        self.P0_np = system_model.P0.cpu().numpy()
        self.Q_np_default = system_model.Q.cpu().numpy() # Kovariance procesního šumu 'v' z rovnice (1)
        self.R_np_default = system_model.R.cpu().numpy() # Kovariance měřicího šumu 'n' z rovnice (2)
        
        # Interní stav filtru, který se bude měnit v každém kroku.
        # Odpovídá množině {x_k^i, w_k^i} z pseudokódu v článku.
        self.particles = None  # NumPy pole (N, state_dim), stavy všech částic x_k^i
        self.weights = None    # NumPy pole (N,), váhy všech částic w_k^i

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
        """
        Vypočítá pravděpodobnost (likelihood) p(z_k|x_k^i) pro JEDNU částici.
        Tato hodnota říká, jak moc je stav částice 'particle' v souladu se skutečným měřením 'z_np'.
        Je to klíčová komponenta pro aktualizaci vah (rovnice 16 a 17, str. 9).
        """
        # Převedeme částici na Torch tenzor pro použití modelové funkce h
        particle_torch = torch.from_numpy(particle).float().to(self.device).reshape(1, -1)
        
        # Spočítáme očekávané měření pro danou částici: h(x_k^i)
        expected_measurement_torch = self.h(particle_torch)
        expected_measurement = expected_measurement_torch.cpu().numpy().flatten()
        
        # Vypočítáme hodnotu PDF Gaussovy distribuce měřicího šumu.
        # Ptáme se: "Jaká je pravděpodobnost, že bychom naměřili 'z_np', pokud by
        # skutečný stav byl 'particle' a náš senzor měl šum s kovariancí R?"
        likelihood = multivariate_normal.pdf(z_np, mean=expected_measurement, cov=R_np) # zde je pdf protože chci konkrétní realizaci v daném bodě
        return likelihood

    def _normalize_weights(self, weights):
        """Normalizuje váhy tak, aby jejich součet byl 1. Ekvivalent 'normalize_weights' z reference."""
        sum_weights = np.sum(weights)
        if sum_weights < 1e-15:
            # Ochrana proti numerickému kolapsu, kdy jsou všechny váhy prakticky nulové.
            # V takovém případě resetujeme váhy na uniformní rozdělení.
            print("Varování: Součet vah je téměř nulový. Resetuji na uniformní rozdělení.")
            weights.fill(1.0 / self.N)
        else:
            weights /= sum_weights
        return weights

    def _systematic_resample(self, particles, weights):
        """
        Implementace Systematic Resampling. Tento algoritmus bojuje proti degeneraci
        (Challenge I, sekce 4.2.1) tím, že vytváří novou populaci částic, kde částice
        s vysokou vahou jsou naklonovány a ty s nízkou jsou odstraněny.
        Logika je přesnou replikou referenčního kódu.
        """
        N = len(particles)
        
        # Vytvoříme kumulativní součet vah (CDF). Představte si to jako "koláč",
        # kde velikost dílku pro každou částici odpovídá její váze.
        Q = np.cumsum(weights)

        # Vygenerujeme pouze JEDEN náhodný startovací bod v prvním segmentu [1e-10, 1/N).
        u0 = np.random.uniform(1e-10, 1.0 / N)
        
        new_particles = np.empty_like(particles)
        n = 0  # Index pro nové (resamplované) částice
        m = 0  # Index pro staré částice (pro procházení CDF)
        
        while n < N:
            # Vypočítáme pozici "ukazatele" pro aktuální novou částici.
            # Všechny ukazatele jsou od sebe vzdáleny přesně 1/N.
            u = u0 + float(n)/ N
            
            # Najdeme první částici v CDF, jejíž "dílek koláče" zasahuje za náš ukazatel.
            while u > Q[m]:
                m += 1
            
            # Zkopírujeme tuto vybranou částici do nové sady.
            new_particles[n] = particles[m]
            
            n += 1
            
        return new_particles

    def get_estimate(self, particles, weights):
        """
        Vypočítá finální odhad stavu jako vážený průměr všech částic.
        Článek zmiňuje, že mrak částic reprezentuje posterior (rovnice 14),
        a toto je způsob, jak z něj získat jeden bodový odhad.
        """
        mean = np.average(particles, weights=weights, axis=0) # vážený průměr přes particles a váhy
        # Vážená kovariance pro odhad nejistoty (potřebné pro ANEES)
        diff = particles - mean
        cov = (diff * weights[:, np.newaxis]).T @ diff
        
        # Převedení na Torch tenzory
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
        
        # Pojistka
        assert current_particles.shape == (self.N, self.state_dim), "Chyba inicializace!"
        
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
            
            # Vážené částice PŘED resampligem
            propagated_particles = temp_propagated_particles
            propagated_weights = self._normalize_weights(temp_new_weights)
            
            # debug
            assert propagated_particles.shape == (self.N, self.state_dim), f"Chyba po propagaci v kroku {k}!"

            effective_N = 1. / np.sum(propagated_weights**2)
            if effective_N < n_threshold:
                # 2. Provedeme resampling
                resampled_particles = self._systematic_resample(propagated_particles, propagated_weights)
                
                # Finální částice pro tento krok
                current_particles = resampled_particles
                current_weights = np.full(self.N, 1.0 / self.N)
            else:
                # Pokud neresamplujeme, přeneseme vážené částice do dalšího kroku
                current_particles = propagated_particles
                current_weights = propagated_weights
                
            # debug
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