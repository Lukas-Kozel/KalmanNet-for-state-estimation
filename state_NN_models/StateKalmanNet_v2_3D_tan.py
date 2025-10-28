import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_KalmanNet_v2_3D_tan import DNN_KalmanNet_v2_3D_tan 

class StateKalmanNet_v2_3D_tan(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1):
        super(StateKalmanNet_v2_3D_tan, self).__init__()
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_KalmanNet_v2_3D_tan(system_model, hidden_size_multiplier=hidden_size_multiplier, output_layer_multiplier=output_layer_multiplier, num_gru_layers=num_gru_layers).to(device) # Odstraněny nepotřebné multiplikátory

        self.x_filtered_prev = None
        self.delta_x_prev = None # Toto je F4: x_{t-1|t-1} - x_{t-1|t-2}
        self.y_prev = None       # Pro F1: y_t - y_{t-1}
        self.h_prev = None
        # --- NOVÝ STAV ---
        self.x_pred_prev = None # Pro F3*: x_{t|t-1} - x_{t-1|t-2}
        # -----------------

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is not None:
            self.x_filtered_prev = initial_state.clone().to(self.device)
        else:
            Ex0_batch = self.system_model.get_deterministic_initial_state().unsqueeze(0).expand(batch_size, -1)
            self.x_filtered_prev = Ex0_batch.clone().to(self.device)

        # F4 (delta_x_prev) na začátku nula
        self.delta_x_prev = torch.zeros_like(self.x_filtered_prev)
        
        # y_prev na začátku nula
        self.y_prev = torch.zeros(batch_size, self.obs_dim, device=self.device)
        
        # h_prev na začátku nula
        self.h_prev = torch.zeros(self.dnn.gru.num_layers, batch_size, self.dnn.gru_hidden_dim, device=self.device)

        # --- NOVÝ STAV ---
        # Predikce pro krok t=0 (použijeme filtr. stav z t=-1, což je initial_state)
        # Pro jednoduchost můžeme na začátku nastavit jako filtrovaný stav
        self.x_pred_prev = self.x_filtered_prev.clone() 
        # -----------------

    def step(self, y_t):
        y_t = y_t.to(self.device)
        batch_size = y_t.shape[0]

        # --- PREDIKCE ---
        x_predicted = self.system_model.f(self.x_filtered_prev) # x_{t|t-1}
        y_predicted = self.system_model.h(x_predicted) # y_{t|t-1}

        # --- VÝPOČET VSTUPŮ PRO DNN ---
        
        # 1. Nový Feature (F3*): Rozdíl predikcí x_{t|t-1} - x_{t-1|t-2}
        pred_diff = x_predicted - self.x_pred_prev 
        norm_pred_diff = F.normalize(pred_diff, p=2, dim=1, eps=1e-12)

        # 2. Inovace (F2): y_t - y_{t|t-1}
        innovation = y_t - y_predicted
        norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)

        # 3. Rozdíl stavu (F4): x_{t-1|t-1} - x_{t-1|t-2}
        norm_delta_x = F.normalize(self.delta_x_prev, p=2, dim=1, eps=1e-12)

        # 4. Rozdíl pozorování (F1): y_t - y_{t-1}
        obs_diff = y_t - self.y_prev
        norm_obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)

        # --- VOLÁNÍ DNN se SPRÁVNÝMI vstupy ---
        K_vec, h_new = self.dnn(
            norm_pred_diff,    # Nahrazeno místo norm_state_inno
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
        px_clamped = x_filtered_unclamped[:, 0].clamp(self.system_model.min_x, self.system_model.max_x)
        py_clamped = x_filtered_unclamped[:, 1].clamp(self.system_model.min_y, self.system_model.max_y)
        x_filtered = torch.stack([px_clamped, py_clamped, x_filtered_unclamped[:,2]], dim=1)

        # --- AKTUALIZACE STAVŮ PRO PŘÍŠTÍ KROK ---
        self.delta_x_prev = x_filtered - x_predicted 
        self.x_filtered_prev = x_filtered 
        self.y_prev = y_t.clone() 
        self.h_prev = h_new
        # --- NOVÝ STAV ---
        self.x_pred_prev = x_predicted.clone() # Uložíme aktuální predikci pro další krok
        # -----------------

        return x_filtered
