import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, pinv, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

class KalmanFilter:
    """
    Kalmanův filtr pro t-invaritantní systém s lineární dynamikou.
    """
    def __init__(self,model):
        self.device = model.Q.device
        self.model = model
        self.Ex0 = model.Ex0  # očekávaná hodnota počátečního stavu
        self.P0 = model.P0  # Počáteční kovarianční matice
        self.F = model.F
        self.H = model.H
        self.Q = model.Q
        self.R = model.R
        self.state_dim = self.F.shape[0]
        self.obs_dim = self.H.shape[0]

        # Interní stav pro online použití: predikce pro aktuální krok
        self.x_predict_current = None
        self.P_predict_current = None

        self.reset(model.Ex0, model.P0)

    def reset(self, Ex0, P0):
        """
        Inicializuje nebo resetuje interní stav filtru pro online použití.
        """
        self.x_predict_current = Ex0.clone().detach().reshape(self.state_dim, 1)
        self.P_predict_current = P0.clone().detach()

    def predict_step(self, x_filtered, P_filtered):
        x_predict = self.F @ x_filtered
        P_predict = self.F @ P_filtered @ self.F.T + self.Q
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        y_t = y_t.reshape(self.obs_dim, 1)
        innovation = self.compute_innovation(y_t, x_predict)
        K = self.compute_kalman_gain(P_predict)
        x_filtered = x_predict + K @ innovation
        I = torch.eye(self.state_dim, device=self.device)
        P_filtered = (I - K @ self.H) @ P_predict @ (I - K @ self.H).T + K @ self.R @ K.T # Joseph form
        return x_filtered, P_filtered, K, innovation

    def compute_kalman_gain(self, P_predict):
        return P_predict @ self.H.T @ torch.linalg.inv(self.H @ P_predict @ self.H.T + self.R)
        
    
    def compute_innovation(self, y_t, x_predict):
        return y_t - self.H @ x_predict
    
    def step(self, y_t):
        """
        Provede jeden kompletní krok filtrace pro ONLINE použití.
        Vrací nejlepší odhad pro aktuální čas 't'.
        """
        x_filtered, P_filtered, _, _ = self.update_step(self.x_predict_current, y_t, self.P_predict_current)

        x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered)

        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next

        return x_filtered, P_filtered

    def process_sequence(self, y_seq, Ex0=None, P0=None):
            """
            Zpracuje celou sekvenci měření `y_seq` (offline) a vrátí detailní historii.
            """

            seq_len = y_seq.shape[0]

            x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
            P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
            kalman_gain_history = torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device)
            innovation_history = torch.zeros(seq_len, self.obs_dim, device=self.device)

            if Ex0 is None:
                Ex0 = self.Ex0
            if P0 is None:
                P0 = self.P0
            x_predict_k = Ex0.clone().detach().reshape(self.state_dim, 1)
            P_predict_k = P0.clone().detach()
            for k in range(seq_len):
                x_filtered, P_filtered, K, innovation = self.update_step(x_predict_k, y_seq[k], P_predict_k)

                x_predict_k_plus_1, P_predict_k_plus_1 = self.predict_step(x_filtered, P_filtered)

                x_predict_k = x_predict_k_plus_1
                P_predict_k = P_predict_k_plus_1

                x_filtered_history[k] = x_filtered.squeeze()
                P_filtered_history[k] = P_filtered
                kalman_gain_history[k] = K
                innovation_history[k] = innovation.squeeze()

            return {
                'x_filtered': x_filtered_history,
                'P_filtered': P_filtered_history,
                'Kalman_gain': kalman_gain_history,
                'innovation': innovation_history
            }
    
class AdaptiveKalmanFilter:
    def __init__(self, model, mdm_L=2, mdm_version=2):
        """
        Inicializuje adaptivní filtr.
        :param model: Objekt obsahující počáteční F, H, Q, R, Ex0, P0.
        :param mdm_L: Parametr L pro MDM metodu.
        :param mdm_version: Parametr version pro MDM metodu.
        """
        self.kf = KalmanFilter(model)
        self.device = model.Q.device
        self.mdm_L = mdm_L # delka posuvneho okna pro MDM
        self.mdm_version = mdm_version

    def _reconstruct_qr_from_alpha2(self, alpha_2, nw, nv):
        """
        Pomocná funkce pro rekonstrukci matic Q a R z vektoru alpha_2.
        """
        # Počet unikátních prvků v symetrických maticích Q a R
        q_elements_count = nw * (nw + 1) // 2
        
        # Extrahujeme prvky pro Q a R
        alpha_q = alpha_2[:q_elements_count]
        alpha_r = alpha_2[q_elements_count : q_elements_count + (nv * (nv + 1) // 2)]
        
        # Rekonstrukce Q
        Q_est = np.zeros((nw, nw))
        i, j = np.triu_indices(nw)
        Q_est[i, j] = alpha_q
        Q_est[j, i] = alpha_q
        
        # Rekonstrukce R
        R_est = np.zeros((nv, nv))
        i, j = np.triu_indices(nv)
        R_est[i, j] = alpha_r
        R_est[j, i] = alpha_r
        
        return Q_est, R_est

    def estimate_qr_from_data(self, y_seq, u_seq=None):
        """
        Provede odhad Q a R z balíku dat pomocí MDM.
        :param y_seq: Sekvence měření (PyTorch tensor).
        :param u_seq: Sekvence vstupů (PyTorch tensor), pokud existuje.
        """
        print("Spouštím MDM odhad Q a R...")
        
        # KROK 1: Konverze PyTorch -> NumPy
        F_np = self.kf.F.cpu().numpy() # převod F na numpy
        H_np = self.kf.H.cpu().numpy() # převod H na numpy
        z_np = y_seq.cpu().numpy().squeeze() # převod měření na numpy 

        # Ujistíme se, že z_np má správný tvar (N, obs_dim)
        if z_np.ndim == 1:
            z_np = z_np.reshape(-1, 1)

        # Pro G, E, D předpokládáme, že jsou součástí modelu
        G_np = np.zeros((F_np.shape[0], 1)) 
        E_np = np.eye(F_np.shape[0])
        D_np = np.eye(H_np.shape[0])
        
        if u_seq is not None:
            u_np = u_seq.cpu().numpy().squeeze()
            if u_np.ndim == 1:
                u_np = u_np.reshape(-1, 1)
        else:
            u_np = np.zeros((y_seq.shape[0], G_np.shape[1]))

        nw, nv = F_np.shape[0], H_np.shape[0]
        nz_np = np.array([nv])

        # KROK 2: Spuštění MDM algoritmu

        # r je realizace MDM statistky, což ve článku je označeno jako Z^~_k (rovnice 11). Není to jen reziduum jako y-hx, ale 
        # je to zkonstruovaný vektor tak, aby byl přímou lineární funkcí systémových šumů.

        # AWv je transformační matice ve článku označena jako A_k (rovnice 13). 
        # Popisuje přesný matematický vztah mezi šumy a statistikou r. Platí r[k] ≈ Awv @ [w[k]; v[k]].
        # Je to klíč, který nám později umožní z vlastností r odvodit vlastnosti w a v.
        r, Awv = MDM_nullO_LTI(self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, z_np, u_np, self.mdm_version)
        

        r0 = np.array(r).squeeze() # převod r na numpy pole
        Np = r0.shape[0]
        nr = r0.shape[1]
        
        # Ksi je selektorová matice.
        # Ksi slouží k tomu, abychom z "natažené" kovarianční matice vybrali jen unikátní prvky (např. ty na a nad diagonálou). 
        Ksi = Ksi_fun(nr) 

        #  kron2_vec udělá r[i] @ r[i].T a výsledek natáhne do vektoru. Ksi z něj pak vybere unikátní prvky
        r02 = np.array([Ksi @ kron2_vec(r0[i]) for i in range(Np)])

        # sestavení matice A. 
        w2b = [baseMatrix_fun(nw, 1)]
        v2b = [baseMatrix_fun(nv, 1)]
        Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)

        # popisuje jak se neznámé prvky Q a R promítají do druhého momentu statistiky r, rovnice (21) ve článku.
        Awv2u = Ksi @ kron2_mat(Awv) @ Upsilon_2

        # řešení sestavené rovnice 
        # Toto je finální řešení soustavy y = A * x. - řešení MNČ.
        # výsledný hledaný vektor, který obsahuje odhadnuté prvky matic Q a R, ve článku je to R̂_ε^U,m z rovnice (24).
        alpha_2 = pinv(Awv2u) @ np.mean(r02, axis=0) 
        
        # rekonstrukce matic Q a R z vektoru alpha_2
        Q_est_np, R_est_np = self._reconstruct_qr_from_alpha2(alpha_2, nw, nv)
        
        print("MDM odhad dokončen.")
        print("Odhadnuté Q:\n", Q_est_np)
        print("Odhadnuté R:\n", R_est_np)

        Q_est_torch = torch.from_numpy(Q_est_np).float().to(self.device)
        R_est_torch = torch.from_numpy(R_est_np).float().to(self.device)

        return Q_est_torch, R_est_torch

    def update_qr_matrices(self, new_Q, new_R):
        """Aktualizuje matice Q a R v interním Kalmanově filtru."""
        self.kf.Q = new_Q
        self.kf.R = new_R
        print("Matice Q a R v Kalmanově filtru byly aktualizovány.")

    def process_sequence_adaptively(self, y_seq, u_seq=None):
        """
        Kompletní adaptivní proces: odhad Q,R a následná filtrace.
        """
        # Fáze 1: Odhad
        Q_est, R_est = self.estimate_qr_from_data(y_seq, u_seq)

        # Fáze 2: Aktualizace a filtrace
        self.update_qr_matrices(Q_est, R_est)
        
        # Nyní spustíme filtraci s novými, odhadnutými maticemi
        results = self.kf.process_sequence(y_seq)
        return results, Q_est, R_est