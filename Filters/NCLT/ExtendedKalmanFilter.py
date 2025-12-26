import torch
from torch.func import jacrev

class ExtendedKalmanFilterNCLT:
    def __init__(self, system_model):
        self.device = system_model.Q.device
        self.f = system_model.f
        self.h = system_model.h

        # Detekce linearity (pro optimalizaci)
        self.is_linear_f = getattr(system_model, 'is_linear_f', False)
        self.is_linear_h = getattr(system_model, 'is_linear_h', False)
        if self.is_linear_f: self.F = system_model.F
        if self.is_linear_h: self.H = system_model.H

        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)

        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]

        # Interní stavy
        self.x_predict_current = None
        self.P_predict_current = None
        self.reset(system_model.Ex0, system_model.P0)

    def reset(self, Ex0, P0):
        self.x_predict_current = Ex0.clone().detach().reshape(self.state_dim, 1)
        self.P_predict_current = P0.clone().detach()

    def predict_step(self, x_filtered, P_filtered, u=None):
        """
        Krok predikce (Time Update).
        x_{k|k-1} = f(x_{k-1|k-1}, u_{k-1})
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        # Ošetření vstupu u
        if u is not None:
            # Zajistíme tvar [Input_dim, 1] pro výpočty
            u_in = u.view(-1, 1) 
            # Pro wrapper potřebujeme [1, Input_dim]
            u_in_flat = u.view(1, -1)
        else:
            u_in = None
            u_in_flat = None

        # 1. Výpočet Jakobiánu F (Linearizace v bodě x_filtered)
        if self.is_linear_f:
            F_t = self.F
        else:
            def f_wrapper(x_in):
                # x_in: [State_dim] -> unsqueeze -> [1, State_dim]
                # f očekává [Batch, State_dim] a [Batch, Input_dim]
                return self.f(x_in.unsqueeze(0), u_in_flat).squeeze(0)
            
            # jacrev vrátí [State_dim, State_dim]
            F_t = jacrev(f_wrapper)(x_filtered.squeeze())

        # 2. Predikce stavu x
        # x_filtered.T je [1, State], u_in_flat je [1, Input]
        x_predict = self.f(x_filtered.T, u_in_flat).T # Výsledek [State, 1]

        # 3. Predikce kovariance P
        P_predict = F_t @ P_filtered @ F_t.mT + self.Q
        
        return x_predict, P_predict

    def update_step(self, x_predict, y_t, P_predict):
        """
        Krok korekce (Measurement Update).
        Řeší NaN v y_t tak, že přeskočí update.
        """
        # === HANDLING NaN (Výpadek GPS) ===
        # Pokud je v měření jakékoliv NaN, nemůžeme provést update.
        if torch.any(torch.isnan(y_t)):
            # Vracíme predikci jako "vyfiltrovanou" hodnotu.
            # P se nesnižuje (protože nemáme novou informaci).
            # K a innovation vrátíme jako nuly (pro logování).
            zero_K = torch.zeros(self.state_dim, self.obs_dim, device=self.device)
            zero_innov = torch.zeros(self.obs_dim, 1, device=self.device)
            return x_predict, P_predict, zero_K, zero_innov

        # Příprava y_t
        y_t = y_t.reshape(self.obs_dim, 1)

        # 1. Výpočet Jakobiánu H (Linearizace v bodě predikce x_predict)
        if self.is_linear_h:
            H_t = self.H
        else:
            def h_wrapper(x_in):
                return self.h(x_in.unsqueeze(0)).squeeze(0)
            H_t = jacrev(h_wrapper)(x_predict.squeeze())

        # 2. Inovace (Residual)
        y_predict = self.h(x_predict.T).T
        innovation = y_t - y_predict
        
        # 3. Kalman Gain
        S = H_t @ P_predict @ H_t.mT + self.R
        K = P_predict @ H_t.mT @ torch.linalg.pinv(S)
        
        # 4. Update stavu a kovariance
        x_filtered = x_predict + K @ innovation
        I = torch.eye(self.state_dim, device=self.device)
        # Joseph form update pro numerickou stabilitu P = (I-KH)P(I-KH)' + KRK'
        P_filtered = (I - K @ H_t) @ P_predict @ (I - K @ H_t).mT + K @ self.R @ K.mT

        return x_filtered, P_filtered, K, innovation

    def process_sequence(self, y_seq, u_seq=None, Ex0=None, P0=None):
        """
        Dávkové zpracování celé trajektorie.
        Sekvenční volání Predict -> Update nebo Update -> Predict.
        Pro NCLT (kde máme x0 a chceme hned zpracovat y0) je lepší cyklus:
        Loop k:
           1. Update (koriguj x_{k|k-1} pomocí y_k) -> získáš x_{k|k}
           2. Ulož x_{k|k}
           3. Predict (posuň x_{k|k} pomocí u_k na x_{k+1|k})
        """
        seq_len = y_seq.shape[0]
        
        # Inicializace historie
        x_hist = torch.zeros(seq_len, self.state_dim, device=self.device)
        P_hist = torch.zeros(seq_len, self.state_dim, self.state_dim, device=self.device)
        
        # Inicializace (Předpokládáme, že Ex0 je x_{0|-1} nebo x_{0|0} - záleží na konvenci)
        # Zde předpokládáme: Ex0 je PRIOR pro čas t=0 (nebo startovní bod, který se má zkusit opravit měřením)
        x_curr = Ex0.clone().detach().reshape(self.state_dim, 1)
        P_curr = P0.clone().detach()
        
        for k in range(seq_len):
            # --- 1. UPDATE STEP (Korekce) ---
            # Zkusíme opravit aktuální odhad měřením (pokud existuje)
            # Pokud je y_seq[k] NaN, update_step vrátí x_curr beze změny (skip)
            x_upd, P_upd, _, _ = self.update_step(x_curr, y_seq[k], P_curr)
            
            # Uložíme VÝSLEDEK (Posterior) pro tento časový krok
            x_hist[k] = x_upd.squeeze()
            P_hist[k] = P_upd

            # --- 2. PREDICT STEP (Posun do budoucna) ---
            # Připravíme prior pro k+1
            if k < seq_len - 1:
                # Zpracování vstupu u
                if u_seq is not None:
                    u_k = u_seq[k] # Zde předpokládáme, že u[k] posouvá stav k -> k+1
                else:
                    u_k = None
                
                # Predikce x_{k+1|k}
                x_curr, P_curr = self.predict_step(x_upd, P_upd, u_k)

        return {'x_filtered': x_hist, 'P_filtered': P_hist}