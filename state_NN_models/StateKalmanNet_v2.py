import torch
import torch.nn as nn
import torch.nn.functional as F
# Zajistíme import nové DNN třídy
from .DNN_KalmanNet_v2 import DNN_KalmanNet_v2 

class StateKalmanNet_v2(nn.Module):
    """
    Rozšířená verze StateKalmanNet, která používá DNN_KalmanNet_v2 
    s referenčními dimenzemi a 4 vstupními příznaky.
    """
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1):
        super(StateKalmanNet_v2, self).__init__()
        self.returns_covariance = False # Tento model vrací jen stav
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        # Instance nové DNN
        self.dnn = DNN_KalmanNet_v2(system_model,hidden_size_multiplier, output_layer_multiplier, num_gru_layers,).to(device)

        # Inicializace stavů pro ukládání mezi kroky
        # Tyto se nastaví správně v self.reset()
        self.x_filtered_prev = None
        self.delta_x_prev = None
        self.y_prev = None
        self.h_prev = None

    def reset(self, batch_size=1, initial_state=None):
        """
        Resetuje vnitřní stav filtru (včetně stavů pro DNN vstupy).
        """
        if initial_state is not None:
            # Použijeme poskytnutý počáteční stav
            self.x_filtered_prev = initial_state.clone().to(self.device)
        else:
            # Pokud není poskytnut, použijeme Ex0 z modelu (nebo nuly)
            # Důležité: zajistit správný batch_size
            Ex0_batch = self.system_model.get_deterministic_initial_state().unsqueeze(0).expand(batch_size, -1)
            self.x_filtered_prev = Ex0_batch.clone().to(self.device)
            # Alternativně: self.x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)

        # delta_x_{t-1}: Rozdíl x_{t-1|t-1} - x_{t-1|t-2} (na začátku nula)
        self.delta_x_prev = torch.zeros_like(self.x_filtered_prev)

        # y_{t-1}: Předchozí měření (na začátku nula)
        self.y_prev = torch.zeros(batch_size, self.obs_dim, device=self.device)
        
        # h_{t-1}: Skrytý stav GRU (na začátku nula)
        # Získání správné dimenze z instance DNN
        self.h_prev = torch.zeros(self.dnn.gru.num_layers, batch_size, self.dnn.gru_hidden_dim, device=self.device)

    def step(self, y_t):
        """
        Provede jeden kompletní krok filtrace pro měření y_t.
        """
        # Zajistíme, že y_t je na správném zařízení
        y_t = y_t.to(self.device)
        batch_size = y_t.shape[0]

        # --- PREDIKCE ---
        # x_{t|t-1} = f(x_{t-1|t-1})
        x_predicted = self.system_model.f(self.x_filtered_prev)
        
        # y_{t|t-1} = h(x_{t|t-1})
        y_predicted = self.system_model.h(x_predicted)

        # --- VÝPOČET VSTUPŮ PRO DNN ---
        
        # 1. state_inno: Normalizovaný predikovaný stav x_{t|t-1}
        #    (Referenční kód nepoužívá stav, ale rozdíl x_{t|t-1} - x_{t-1|t-1}, 
        #     což je ekvivalentní x_predicted - self.x_filtered_prev. Zkusme to.)
        # state_diff_pred_prev = x_predicted - self.x_filtered_prev 
        # norm_state_inno = F.normalize(state_diff_pred_prev, p=2, dim=1, eps=1e-12)
        # JEDNODUŠŠÍ VARIANTA: Použijeme jen normalizovaný predikovaný stav
        norm_state_inno = F.normalize(x_predicted, p=2, dim=1, eps=1e-12)

        # 2. inovation (F2): Normalizovaný rozdíl y_t - y_{t|t-1}
        innovation = y_t - y_predicted
        norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)

        # 3. diff_state (F4): Normalizovaný rozdíl x_{t-1|t-1} - x_{t-1|t-2}
        norm_delta_x = F.normalize(self.delta_x_prev, p=2, dim=1, eps=1e-12)

        # 4. diff_obs (F1): Normalizovaný rozdíl y_t - y_{t-1}
        obs_diff = y_t - self.y_prev
        norm_obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)

        # --- VOLÁNÍ DNN ---
        K_vec, h_new = self.dnn(
            norm_state_inno, 
            norm_innovation, 
            norm_delta_x, 
            norm_obs_diff, 
            self.h_prev
        )

        # --- KOREKCE ---
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        x_filtered_unclamped = x_predicted + correction

        # --- OMEZENÍ STAVU (CLAMPING) ---
        # (Stejné jako v předchozí verzi)
        px_clamped = x_filtered_unclamped[:, 0].clamp(self.system_model.min_x, self.system_model.max_x)
        py_clamped = x_filtered_unclamped[:, 1].clamp(self.system_model.min_y, self.system_model.max_y)
        max_vel = 200.0 # Ponecháme jako pojistku
        vx_clamped = x_filtered_unclamped[:, 2].clamp(-max_vel, max_vel)
        vy_clamped = x_filtered_unclamped[:, 3].clamp(-max_vel, max_vel)
        
        # Logování omezení (volitelné)
        if torch.any((vx_clamped == max_vel) | (vx_clamped == -max_vel)):
            print(f"Varování: Došlo k omezení rychlosti v ose X (max_vel={max_vel}).")
        if torch.any((vy_clamped == max_vel) | (vy_clamped == -max_vel)):
            print(f"Varování: Došlo k omezení rychlosti v ose Y (max_vel={max_vel}).")

        x_filtered = torch.stack([px_clamped, py_clamped, vx_clamped, vy_clamped], dim=1)

        # --- AKTUALIZACE STAVŮ PRO PŘÍŠTÍ KROK ---
        # delta_x_t = x_{t|t} - x_{t|t-1}
        self.delta_x_prev = x_filtered - x_predicted # Použijeme finální, ořezaný stav
        
        # Uložíme finální stav jako předchozí pro další krok
        self.x_filtered_prev = x_filtered 
        
        # Uložíme aktuální měření jako předchozí pro další krok
        self.y_prev = y_t.clone() 
        
        # Uložíme nový skrytý stav GRU
        self.h_prev = h_new

        return x_filtered
