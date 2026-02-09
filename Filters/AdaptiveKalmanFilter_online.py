import torch
import numpy as np
# Předpokládám, že MDM_functions jsou dostupné, jak je v zadání
from MDM.MDM_functions import MDM_nullO_LTI, pinv, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat
from tqdm import tqdm

class KalmanFilter:
    """
    Kalmanův filtr pro t-invaritantní systém s lineární dynamikou.
    (Původní implementace beze změn)
    """
    def __init__(self,model):
        self.device = model.Q.device
        self.model = model
        self.Ex0 = model.Ex0
        self.P0 = model.P0
        self.F = model.F
        self.H = model.H
        self.Q = model.Q
        self.R = model.R
        self.state_dim = self.F.shape[0]
        self.obs_dim = self.H.shape[0]

        # Interní stav pro online použití
        self.x_predict_current = None
        self.P_predict_current = None
        self.reset(model.Ex0, model.P0)

    def reset(self, Ex0, P0):
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
        P_filtered = (I - K @ self.H) @ P_predict @ (I - K @ self.H).T + K @ self.R @ K.T 
        return x_filtered, P_filtered, K, innovation

    def compute_kalman_gain(self, P_predict):
        return P_predict @ self.H.T @ torch.linalg.inv(self.H @ P_predict @ self.H.T + self.R)
    
    def compute_innovation(self, y_t, x_predict):
        return y_t - self.H @ x_predict
    
    def step(self, y_t):
        x_filtered, P_filtered, _, _ = self.update_step(self.x_predict_current, y_t, self.P_predict_current)
        x_predict_next, P_predict_next = self.predict_step(x_filtered, P_filtered)
        self.x_predict_current = x_predict_next
        self.P_predict_current = P_predict_next
        return x_filtered, P_filtered

    def process_sequence(self, y_seq, Ex0=None, P0=None):
            seq_len = y_seq.shape[0]
            x_filtered_history = torch.zeros(seq_len, self.state_dim, device=self.device)
            P_filtered_history = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
            kalman_gain_history = torch.zeros(seq_len, self.state_dim, self.obs_dim, device=self.device)
            innovation_history = torch.zeros(seq_len, self.obs_dim, device=self.device)

            if Ex0 is None: Ex0 = self.Ex0
            if P0 is None: P0 = self.P0
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

import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat
class AdaptiveKalmanFilter_online:
    def __init__(self, model, mdm_L=4, mdm_version=1):
        """
        Adaptivní KF (Semi-weighted Recursive verze - Sw-Re).
        Tato třída provádí online identifikaci matic Q a R.
        """
        self.kf = KalmanFilter(model)
        self.device = model.Q.device
        
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        
        # Získání dimenzí šumu
        self.nw = self.kf.F.shape[0] # Dimenze stavového šumu (W)
        self.nv = self.kf.H.shape[0] # Dimenze šumu měření (V)
        
        # --- Výpočet počtu parametrů k odhadu ---
        # Protože Q a R jsou kovarianční matice, MUSÍ být symetrické.
        # R_ij = R_ji. Nemusíme tedy odhadovat n*n prvků, ale jen unikátní prvky.
        # Počet unikátních prvků v matici n x n je n*(n+1)/2 (horní trojúhelník + diagonála).
        # Zde odhadujeme PLNÉ matice Q a R (včetně korelací mezi stavy), ne jen diagonálu.
        self.n_params = (self.nw * (self.nw + 1)) // 2 + (self.nv * (self.nv + 1)) // 2
        
        # 1. Inicializace odhadu (alpha) z modelu
        # Vektor 'alpha' bude obsahovat všechny unikátní prvky Q a R seřazené za sebou.
        Q0 = model.Q.cpu().numpy()
        R0 = model.R.cpu().numpy()
        
        # Získání indexů horního trojúhelníku (včetně diagonály)
        # np.triu_indices vrátí souřadnice (rows, cols) pro prvky nad hlavní diagonálou.
        i_q, j_q = np.triu_indices(self.nw)
        # alpha_q je plochý vektor unikátních prvků Q
        alpha_q = Q0[i_q, j_q]
        
        i_r, j_r = np.triu_indices(self.nv)
        # alpha_r je plochý vektor unikátních prvků R
        alpha_r = R0[i_r, j_r]
        
        # Spojíme je do jednoho vektoru parametrů, který bude RLS odhadovat.
        # Tento vektor odpovídá veličině (R_E^2)^U v článku MDM.
        self.alpha_est = np.concatenate([alpha_q, alpha_r])
        
        # 2. Inicializace kovariance RLS (Sigma)
        # Toto je "nejistota v parametrech Q a R". 
        # Nastavením na 10 * I říkáme: "Věříme počátečnímu Q0/R0, ale připouštíme změnu."
        # Odpovídá P_QR_SwRe_apri v MATLAB kódu.
        self.Sigma_RLS = np.eye(self.n_params) * 5.0 

        # Buffery pro uchování okna měření (Z_k^L)
        self.z_buffer = []
        self.u_buffer = []
        self.Upsilon_2 = None

    def _get_upsilon(self):
        """
        Pomocná funkce pro získání matice Upsilon.
        Tato matice je čistě strukturální (záleží jen na dimenzích) a slouží
        k manipulaci s Kroneckerovými součiny v rámci MDM teorie.
        Počítá se jen jednou.
        """
        if self.Upsilon_2 is None:
            w2b = [baseMatrix_fun(self.nw, 1)] 
            v2b = [baseMatrix_fun(self.nv, 1)] 
            self.Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        return self.Upsilon_2

    def _reconstruct_qr_from_alpha2(self, alpha_2):
        """
        KLÍČOVÁ FUNKCE: Převod vektoru parametrů zpět na matice.
        
        Vstup: alpha_2 (vektor délky n_params)
        Výstup: Matice Q a R
        
        Logika:
        1. Rozdělí vektor alpha na část pro Q a část pro R.
        2. Vytvoří prázdné matice.
        3. Vyplní horní trojúhelník (triu) hodnotami z vektoru.
        4. Zkopíruje hodnoty i do dolního trojúhelníku (symetrizace),
           protože kovarianční matice musí být symetrická (Q_ij = Q_ji).
        """
        # Kolik prvků patří matici Q?
        q_len = (self.nw * (self.nw + 1)) // 2
        
        # Rozdělení vektoru
        alpha_q = alpha_2[:q_len]
        alpha_r = alpha_2[q_len:]
        
        # Rekonstrukce Q
        Q_est = np.zeros((self.nw, self.nw))
        i, j = np.triu_indices(self.nw)
        # Vyplnění horního trojúhelníku
        Q_est[i, j] = alpha_q
        # Symetrizace: Q[j, i] = Q[i, j]
        # Tímto říkáme, že kovariance mezi stavem 1 a 2 je stejná jako mezi 2 a 1.
        Q_est[j, i] = alpha_q
        
        # Rekonstrukce R (stejný princip)
        R_est = np.zeros((self.nv, self.nv))
        i, j = np.triu_indices(self.nv)
        R_est[i, j] = alpha_r
        R_est[j, i] = alpha_r
        
        return Q_est, R_est

    def _project_to_psd(self, M, epsilon=1e-8):
        """
        Zajistí, že matice je Pozitivně Semidefinitní (PSD).
        RLS algoritmus je jen matematika - může mu vyjít matice Q, která má
        záporný rozptyl, což je fyzikální nesmysl.
        
        Tato funkce:
        1. Spočítá vlastní čísla (eigenvalues) matice.
        2. Všechna záporná nebo příliš malá vlastní čísla nahradí malým kladným epsilon.
        3. Zrekonstruuje matici.
        """
        # Pojistka symetrie (pro jistotu)
        M = (M + M.T) / 2.0
        
        eigvals, eigvecs = np.linalg.eigh(M)
        if np.any(eigvals < epsilon):
            # Nahrazení "špatných" vlastních čísel
            eigvals[eigvals < epsilon] = epsilon
            # Rekonstrukce: M = V * diag(lambda) * V.T
            M = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return M

    def step_adaptive(self, y_t, u_t=None):
        """
        Jeden krok adaptivního filtru.
        """
        # --- 1. Update Bufferu (Posuvné okno) ---
        y_np = y_t.cpu().numpy().squeeze()
        if y_np.ndim == 0: y_np = np.expand_dims(y_np, axis=0)
        
        if u_t is not None:
            u_np = u_t.cpu().numpy().squeeze()
            if u_np.ndim == 0: u_np = np.expand_dims(u_np, axis=0)
        else:
            u_np = np.zeros(1)

        self.z_buffer.append(y_np)
        self.u_buffer.append(u_np)
        
        # Udržování fixní délky okna L (FIFO fronta)
        if len(self.z_buffer) > self.mdm_L:
            self.z_buffer.pop(0)
            self.u_buffer.pop(0)

        # --- 2. Identifikace Q a R (RLS) ---
        # Spustí se pouze, pokud je buffer plný (máme dost dat pro MDM)
        if len(self.z_buffer) == self.mdm_L:
            try:
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                # Získání matic systému (předpokládáme LTI, ale bereme aktuální hodnoty)
                F_np = self.kf.F.cpu().numpy()
                H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1])) # matice pro vstup u 
                E_np = np.eye(self.nw) # matice pro šum w
                D_np = np.eye(self.nv) # matice pro šum v
                nz_np = np.array([self.nv])

                # A) MDM Jádro: Výpočet rezidua a regresoru
                # r_k: Vektor reziduí, který nese informaci o šumu v datech
                # Awv_matrix: Pomocná matice popisující dynamiku systému v okně
                r_list, Awv_matrix = MDM_nullO_LTI(
                    self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
                    z_window, u_window, self.mdm_version
                )
                r_k = r_list[0]
                
                # B) Příprava pro RLS (Vektorizace)
                nr = r_k.shape[0]
                Ksi = Ksi_fun(nr) # Unifikační matice (vybírá unikátní prvky ze symetrické matice)
                
                # Pozorování pro RLS (y_rls):
                # Vychází z vnějšího součinu rezidua (r_k * r_k^T), což je "okamžitá kovariance".
                # kron2_vec to převede na vektor a Ksi vybere unikátní prvky.
                y_rls = Ksi @ kron2_vec(r_k)
                
                # Regresní matice (H_rls):
                # Popisuje lineární vztah mezi parametry (Q, R) a pozorováním (kovariancí rezidua).
                Upsilon = self._get_upsilon()
                H_rls = Ksi @ kron2_mat(Awv_matrix) @ Upsilon

                # C) Semi-weighted Váhování (Omega)
                # Toto je specifikum "Semi-weighted" verze.
                # Místo Identity (Unweighted) se použije matice Omega = Ksi * Ksi^T.
                # Tato diagonální matice říká, že mimodiagonální prvky kovariance mají v LS
                # jinou váhu než diagonální prvky (kvůli symetrii se počítají 2x).
                Omega = Ksi @ Ksi.T

                # D) Výpočet RLS Gainu
                # Vzorec: K = P * H^T * (H * P * H^T + Omega)^-1
                # Zde: P = Sigma_RLS, H = H_rls
                S = H_rls @ self.Sigma_RLS @ H_rls.T + Omega
                K_gain_rls = self.Sigma_RLS @ H_rls.T @ np.linalg.pinv(S)
                
                # E) Update odhadu parametrů (alpha)
                # alpha_new = alpha_old + K * (y_measured - y_predicted)
                pred_error = y_rls - H_rls @ self.alpha_est
                self.alpha_est = self.alpha_est + K_gain_rls @ pred_error
                
                # F) Update kovariance parametrů (Sigma)
                # P_new = (I - K * H) * P_old
                I_p = np.eye(self.n_params)
                self.Sigma_RLS = (I_p - K_gain_rls @ H_rls) @ self.Sigma_RLS
                
                # G) Rekonstrukce a kontrola fyzikální smysluplnosti
                # Převedeme vektor alpha zpět na matice Q a R
                Q_new, R_new = self._reconstruct_qr_from_alpha2(self.alpha_est)
                
                # Opravíme případná záporná vlastní čísla (PSD projekce)
                Q_new = self._project_to_psd(Q_new, epsilon=1e-6)
                R_new = self._project_to_psd(R_new, epsilon=1e-3)
                
                # Nahrajeme nové matice do Kalmanova filtru pro tento krok
                self.kf.Q = torch.from_numpy(Q_new).float().to(self.device)
                self.kf.R = torch.from_numpy(R_new).float().to(self.device)
                
            except Exception as e:
                # Pokud nastane numerická chyba (např. singularita), přeskočíme update Q/R
                # a použijeme hodnoty z minulého kroku.
                pass

        # --- 3. Filtrace ---
        # Provedeme standardní krok Kalmanova filtru s aktuálními Q a R
        x_filt, P_filt = self.kf.step(y_t)
        return x_filt, P_filt, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Pomocná metoda pro simulaci běhu na celé sekvenci dat.
        """
        seq_len = y_seq.shape[0]
        
        # 1. Kompletní reset (zapomeneme vše z předchozích běhů)
        self.__init__(self.kf.model, self.mdm_L, self.mdm_version)

        # 2. Nastavení počátečních podmínek
        if Ex0 is None: Ex0 = self.kf.model.Ex0
        if P0 is None: P0 = self.kf.model.P0

        # 3. Reset vnitřního KF
        self.kf.reset(Ex0, P0)
        
        self.z_buffer = []
        self.u_buffer = []

        x_hist = torch.zeros(seq_len, self.kf.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.kf.state_dim, self.kf.state_dim, device=self.device)
        Q_hist = []
        R_hist = []

        for k in tqdm(range(1,seq_len), desc="Processing sequence adaptively"):
            y_t = y_seq[k]
            u_t = u_seq[k] if u_seq is not None else None
            
            x, P, Q_curr, R_curr = self.step_adaptive(y_t, u_t)
            
            x_hist[k] = x.squeeze()
            P_hist[k] = P
            Q_hist.append(Q_curr.clone().detach())
            R_hist.append(R_curr.clone().detach())
            
        results_dict = {
            'x_filtered': x_hist,
            'P_filtered': P_hist
        }
        
        return results_dict, Q_hist, R_hist