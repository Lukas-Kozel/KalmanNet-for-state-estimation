import torch
import numpy as np
from MDM.MDM_functions import MDM_nullO_LTI, Ksi_fun, baseMatrix_fun, Upsilon_2_fun, kron2_vec, kron2_mat

# Třída KalmanFilter zůstává beze změn (importuješ ji nebo definuješ výše)
# Zde definujeme novou třídu pro strukturovanou adaptaci:

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
        P_filtered = (I - K @ self.H) @ P_predict @ (I - K @ self.H).T + K @ self.R @ K.T # Joseph form
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

class StructuredAdaptiveKalmanFilter:
    def __init__(self, model, mdm_L=6, mdm_version=1, forgetting_factor=0.995, 
                 estimate_q_scale=True, diagonal_R=True):
        """
        Adaptivní KF se STRUKTUROVANÝM odhadem Q a R.
        
        Tato třída implementuje expertní radu: místo hledání plných matic Q a R
        hledáme jen jejich klíčové parametry (skalární škálování pro Q, diagonálu pro R).
        
        Args:
            model: Objekt dynamického systému (obsahuje F, H, Q_base, R_base).
            estimate_q_scale: Pokud True, hledá jeden skalár 'q', který škáluje model.Q.
            diagonal_R: Pokud True, hledá jen diagonální prvky R (ostatní jsou 0).
        """
        self.kf = KalmanFilter(model) # Předpokládá existenci třídy Filters.KalmanFilter
        self.device = model.Q.device
        self.mdm_L = mdm_L
        self.mdm_version = mdm_version
        self.lambda_rls = forgetting_factor
        
        # --- 1. Definice rozměrů problému ---
        self.nw = self.kf.F.shape[0] # Dimenze stavového šumu
        self.nv = self.kf.H.shape[0] # Dimenze šumu měření
        
        # Plný počet unikátních prvků v symetrických maticích (horní trojúhelník)
        # To je to, co "vyrábí" MDM funkce standardně.
        self.n_params_full_Q = (self.nw * (self.nw + 1)) // 2
        self.n_params_full_R = (self.nv * (self.nv + 1)) // 2
        
        # --- 2. Definice redukovaného (hledaného) prostoru ---
        self.estimate_q_scale = estimate_q_scale
        self.diagonal_R = diagonal_R
        
        # Q: Hledáme buď 1 skalár, nebo plnou matici
        if self.estimate_q_scale:
            self.n_params_red_Q = 1 
            # Uložíme si "bázovou" matici Q z modelu (např. kinematický model).
            # Budeme hledat koeficient 'q', aby platilo: Q_new = q * Q_base
            self.Q_base_np = model.Q.cpu().numpy() 
        else:
            self.n_params_red_Q = self.n_params_full_Q
            
        # R: Hledáme buď jen diagonálu (nv prvků), nebo plnou matici
        if self.diagonal_R:
            self.n_params_red_R = self.nv 
        else:
            self.n_params_red_R = self.n_params_full_R
            
        # Celkový počet hledaných parametrů (např. 1 + 4 = 5)
        self.n_params_red = self.n_params_red_Q + self.n_params_red_R

        # --- 3. Konstrukce Matice Zobrazení (Mapping Matrix M) ---
        # Toto je klíčový trik. Vztah: alpha_FULL = M * alpha_RED
        self.M = self._construct_mapping_matrix()
        
        # --- 4. Inicializace RLS algoritmu ---
        # Musíme inicializovat vektor alpha_est (hledané parametry).
        alpha_red_init = []
        
        # Init pro Q
        if self.estimate_q_scale:
            alpha_red_init.append(1.0) # Začínáme na hodnotě 1.0 (tedy Q = 1.0 * Model.Q)
        else:
            # Pokud bychom hledali plnou, vezmeme prvky z modelu
            Q0 = model.Q.cpu().numpy()
            i, j = np.triu_indices(self.nw)
            alpha_red_init.extend(Q0[i, j])
            
        # Init pro R
        R0 = model.R.cpu().numpy()
        if self.diagonal_R:
            alpha_red_init.extend(np.diag(R0)) # Vezmeme jen diagonálu
        else:
            i, j = np.triu_indices(self.nv)
            alpha_red_init.extend(R0[i, j]) # Vezmeme celý trojúhelník
            
        self.alpha_est = np.array(alpha_red_init)
        
        # Kovarianční matice RLS (nejistota odhadu parametrů)
        # Nyní má rozměr jen [5x5] místo [20x20] -> mnohem rychlejší konvergence!
        self.Sigma_RLS = np.eye(self.n_params_red) * 100.0

        # Buffery pro MDM okno
        self.z_buffer = []
        self.u_buffer = []
        self.Upsilon_2 = None

    def _construct_mapping_matrix(self):
        """
        Vytvoří matici M, která transformuje redukované parametry na plný vektor.
        Tato matice matematicky "vynucuje" strukturu Q a R.
        """
        # --- Část M pro Q ---
        if self.estimate_q_scale:
            # Pokud hledáme jen skalár q:
            # Plný vektor Q (vech Q) = q * vech(Q_base)
            # Takže sloupec matice M odpovídající Q je roven vech(Q_base).
            i, j = np.triu_indices(self.nw)
            vech_Q_base = self.Q_base_np[i, j]
            M_Q = vech_Q_base.reshape(-1, 1) # Sloupcový vektor [10, 1]
        else:
            M_Q = np.eye(self.n_params_full_Q) # Identita (žádná redukce)

        # --- Část M pro R ---
        if self.diagonal_R:
            # Pokud hledáme jen diagonálu:
            # M_R bude mít rozměr [n_full, n_diag].
            # Každý sloupec odpovídá jednomu diagonálnímu prvku a má "1" na řádku,
            # který v plném vektoru odpovídá pozici tohoto diagonálního prvku.
            M_R = np.zeros((self.n_params_full_R, self.nv))
            
            full_idx = 0 # Index v plném vektoru (vech R)
            diag_idx = 0 # Index v našem redukovaném vektoru (diagonála)
            
            # Procházíme horní trojúhelník (stejně jako np.triu_indices)
            for row in range(self.nv):
                for col in range(row, self.nv):
                    if row == col:
                        # Toto je diagonální prvek! 
                        M_R[full_idx, diag_idx] = 1.0
                        diag_idx += 1
                    # Jinak je to mimodiagonální prvek -> necháme 0 (vynucujeme nulu)
                    full_idx += 1
        else:
            M_R = np.eye(self.n_params_full_R)

        # --- Sestavení celkové M ---
        # M = [ M_Q   0  ]
        #     [  0   M_R ]
        dim_full_Q = M_Q.shape[0]
        dim_red_Q = M_Q.shape[1]
        dim_full_R = M_R.shape[0]
        dim_red_R = M_R.shape[1]
        
        M = np.zeros((dim_full_Q + dim_full_R, dim_red_Q + dim_red_R))
        M[:dim_full_Q, :dim_red_Q] = M_Q
        M[dim_full_Q:, dim_red_Q:] = M_R
        
        return M

    def _get_upsilon(self):
        # Lazy loading pro Upsilon matici (konstantní pro LTI)
        if self.Upsilon_2 is None:
            w2b = [baseMatrix_fun(self.nw, 1)] 
            v2b = [baseMatrix_fun(self.nv, 1)] 
            self.Upsilon_2 = Upsilon_2_fun(w2b, v2b, self.mdm_L)
        return self.Upsilon_2

    def _reconstruct_qr(self, alpha_red):
        """
        Převede redukované parametry zpět na matice Q a R pro Kalmanův filtr.
        """
        # 1. Rozdělení vektoru na část pro Q a část pro R
        if self.estimate_q_scale:
            q_scale = alpha_red[0] # První prvek je náš skalár
            
            # Rekonstrukce Q: Q_new = scale * Q_base
            Q_est = q_scale * self.Q_base_np
            
            # Zbytek vektoru patří R
            alpha_R = alpha_red[1:]
        else:
            # (Vynecháno pro stručnost - full Q logic)
            alpha_R = alpha_red[self.n_params_full_Q:]
            Q_est = np.zeros((self.nw, self.nw)) 

        # 2. Rekonstrukce R
        if self.diagonal_R:
            # alpha_R obsahuje přímo diagonální prvky
            # np.diag vytvoří matici s těmito prvky na diagonále a nulami jinde
            R_est = np.diag(alpha_R)
        else:
            # Full R logic...
            R_est = np.zeros((self.nv, self.nv))
            i, j = np.triu_indices(self.nv)
            R_est[i, j] = alpha_R
            R_est[j, i] = alpha_R

        return Q_est, R_est

    def step_adaptive(self, y_t, u_t=None):
        """
        Jeden krok online adaptace. Zde probíhá kouzlo strukturovaného odhadu.
        """
        # --- 1. Standardní update bufferu (posuvné okno) ---
        y_np = y_t.cpu().numpy().squeeze()
        if y_np.ndim == 0: y_np = np.expand_dims(y_np, axis=0)
        
        if u_t is not None:
            u_np = u_t.cpu().numpy().squeeze()
            if u_np.ndim == 0: u_np = np.expand_dims(u_np, axis=0)
        else:
            u_np = np.zeros(1)

        self.z_buffer.append(y_np)
        self.u_buffer.append(u_np)
        
        if len(self.z_buffer) > self.mdm_L:
            self.z_buffer.pop(0)
            self.u_buffer.pop(0)

        # --- 2. Adaptivní algoritmus (jen pokud máme plné okno) ---
        if len(self.z_buffer) == self.mdm_L:
            try:
                # Příprava dat pro MDM
                z_window = np.array(self.z_buffer)
                u_window = np.array(self.u_buffer)
                
                F_np = self.kf.F.cpu().numpy()
                H_np = self.kf.H.cpu().numpy()
                G_np = np.zeros((self.nw, u_window.shape[1])) 
                E_np = np.eye(self.nw)
                D_np = np.eye(self.nv)
                nz_np = np.array([self.nv])

                # >>> MDM VÝPOČET <<<
                # Získáme "raw" statistiku z dat. Toto je nezávislé na struktuře Q/R.
                # r_list[0] = vektor reziduí (data)
                # Awv_matrix = matice popisující vztah šumů a reziduí (model)
                r_list, Awv_matrix = MDM_nullO_LTI(
                    self.mdm_L, F_np, G_np, E_np, nz_np, H_np, D_np, 
                    z_window, u_window, self.mdm_version
                )
                r_k = r_list[0]
                
                # >>> PŘÍPRAVA RLS <<<
                nr = r_k.shape[0]
                Ksi = Ksi_fun(nr) # Selektor unikátních prvků kovariance
                
                # Pozorovaný vektor y_rls (levá strana rovnice y = H*x)
                # Toto je empirická kovariance reziduí z aktuálního okna
                y_rls = Ksi @ kron2_vec(r_k)
                
                # Matice Upsilon (předpočítaná)
                Upsilon = self._get_upsilon()
                
                # >>> APLIKACE STRUKTURY (To nejdůležitější!) <<<
                
                # 1. H_full: Regresní matice pro PLNÝ problém (hledáme všech 20 prvků)
                # Odpovídá rovnici v článku: A_2 = Ksi * (Awv (x) Awv) * Upsilon
                H_full = Ksi @ kron2_mat(Awv_matrix) @ Upsilon
                
                # 2. H_red: Projekce do STRUKTUROVANÉHO prostoru
                # Aplikujeme naši matici zobrazení M.
                # Rovnice: y = H_full * alpha_full
                # Substituce: alpha_full = M * alpha_red
                # Výsledek: y = (H_full * M) * alpha_red
                # Tímto krokem říkáme RLS algoritmu: "Hledej jen parametry alpha_red, 
                # a my už víme, jak z nich poskládat plné matice."
                H_red = H_full @ self.M 

                # >>> RLS UPDATE (Recursive Least Squares) <<<
                # Nyní pracujeme s výrazně menšími maticemi (např. 5x5 místo 20x20)
                
                # Inovační kovariance S
                S = H_red @ self.Sigma_RLS @ H_red.T + self.lambda_rls * np.eye(H_red.shape[0])
                
                # Kalman Gain pro parametry (zisky)
                # Používáme pinv pro numerickou stabilitu
                K_gain = self.Sigma_RLS @ H_red.T @ np.linalg.pinv(S)
                
                # Chyba predikce parametrů
                pred_error = y_rls - H_red @ self.alpha_est
                
                # Aktualizace odhadu parametrů
                self.alpha_est = self.alpha_est + K_gain @ pred_error
                
                # Aktualizace kovariance odhadu (Sigma)
                I_p = np.eye(self.n_params_red)
                self.Sigma_RLS = (1.0 / self.lambda_rls) * (I_p - K_gain @ H_red) @ self.Sigma_RLS
                
                # >>> REKONSTRUKCE A REGULARIZACE <<<
                Q_new, R_new = self._reconstruct_qr(self.alpha_est)
                
                # Ochrana proti fyzikálním nesmyslům (záporný rozptyl)
                if self.estimate_q_scale:
                     # Skalár q musí být kladný. Pokud RLS uletí do minusu, vrátíme ho na malou kladnou hodnotu.
                     if self.alpha_est[0] < 1e-9: 
                         self.alpha_est[0] = 1e-9
                         # Musíme znovu rekonstruovat, aby to platilo
                         Q_new, R_new = self._reconstruct_qr(self.alpha_est)
                
                if self.diagonal_R:
                    # Diagonální prvky R musí být kladné.
                    # Použijeme práh (např. 1e-3), což je ta "regularizace", o které jsme mluvili dříve.
                    # Zabraňuje to ANEES explozi.
                    np.fill_diagonal(R_new, np.maximum(np.diag(R_new), 1e-2))
                
                # Vložení do filtru
                self.kf.Q = torch.from_numpy(Q_new).float().to(self.device)
                self.kf.R = torch.from_numpy(R_new).float().to(self.device)

            except Exception as e:
                # Fallback: Pokud MDM v tomto kroku selže (např. numerika), 
                # neměníme Q a R a pokračujeme s minulými hodnotami.
                # print(f"MDM Fail: {e}")
                pass

        # --- 3. Krok Kalmanova Filtru ---
        x, P = self.kf.step(y_t)
        
        return x, P, self.kf.Q, self.kf.R

    def process_sequence_adaptively(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Simuluje online běh na celé sekvenci.
        """
        seq_len = y_seq.shape[0]
        
        # 1. Reset RLS (volá __init__ této třídy)
        # Předáváme všechny parametry struktury, aby se zachovaly
        self.__init__(self.kf.model, self.mdm_L, self.mdm_version, self.lambda_rls, 
                      self.estimate_q_scale, self.diagonal_R)
        
        # 2. Nastavení počátečních podmínek filtru
        if Ex0 is None: Ex0 = self.kf.model.Ex0
        if P0 is None: P0 = self.kf.model.P0
        self.kf.reset(Ex0, P0)
        
        # Reset bufferů
        self.z_buffer = []
        self.u_buffer = []

        # Logování historie
        x_hist = torch.zeros(seq_len, self.kf.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.kf.state_dim, self.kf.state_dim, device=self.device)
        Q_hist = []
        R_hist = []

        # Hlavní smyčka
        for k in range(seq_len):
            y_t = y_seq[k]
            u_t = u_seq[k] if u_seq is not None else None
            
            x, P, Q_curr, R_curr = self.step_adaptive(y_t, u_t)
            
            x_hist[k] = x.squeeze()
            P_hist[k] = P
            Q_hist.append(Q_curr.clone().detach())
            R_hist.append(R_curr.clone().detach())
            
        return {'x_filtered': x_hist, 'P_filtered': P_hist}, Q_hist, R_hist