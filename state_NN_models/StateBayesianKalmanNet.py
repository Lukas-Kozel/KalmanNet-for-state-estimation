from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class StateBayesianKalmanNet(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(StateBayesianKalmanNet, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_BayesianKalmanNet(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        self.h_init_master = torch.randn(
            self.dnn.gru.num_layers, 
            1,
            self.dnn.gru.hidden_size, 
            device=self.device
        ).detach()
        self.reset()


    def reset(self, batch_size=1, initial_state=None):
        if initial_state is not None:
            # Počáteční stav pro t=0
            self.x_filtered_t_minus_1 = initial_state.detach().clone()
        else:
            self.x_filtered_t_minus_1 = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # Pro t=1 budou rozdíly nulové, takže tyto stavy inicializujeme stejně
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.clone()
        
        # Predikovaný stav pro t-1 (x_{t-1|t-2})
        x_pred_t_minus_1 = self.system_model.f(self.x_filtered_t_minus_2)
        self.x_pred_t_minus_1 = x_pred_t_minus_1.clone().detach()
        
        # Měření pro t-1 (y_{t-1})
        self.y_t_minus_1 = self.system_model.h(x_pred_t_minus_1).clone().detach()

        # Skrytý stav pro GRU
        self.h_prev = self.h_init_master.expand(-1, batch_size, -1).clone()

    def step(self, y_t):
        # 1. PREDIKCE (pro aktuální čas t)
        # x_{t|t-1} = f(x_{t-1|t-1})
        x_predicted = self.system_model.f(self.x_filtered_t_minus_1)
        # y_{t|t-1} = h(x_{t|t-1})
        y_predicted = self.system_model.h(x_predicted)

        # Ošetření dimenzí
        if y_predicted.dim() == 1: y_predicted = y_predicted.unsqueeze(-1)
        if y_t.dim() == 1: y_t = y_t.unsqueeze(-1)

        # 2. VÝPOČET VSTUPŮ PRO DNN
        # x_{t-1|t-1} - x_{t-1|t-2}
        state_inno = self.x_filtered_t_minus_1 - self.x_pred_t_minus_1
        # y_t - y_{t|t-1}
        inovation = y_t - y_predicted
        # x_{t-1|t-1} - x_{t-2|t-2}
        diff_state = self.x_filtered_t_minus_1 - self.x_filtered_t_minus_2
        # y_t - y_{t-1}
        diff_obs = y_t - self.y_t_minus_1

        # 3. PRŮCHOD SÍTÍ A KOREKCE
        K_vec, h_new, regs = self.dnn(state_inno, inovation, diff_state, diff_obs, self.h_prev)
        
        K = K_vec.reshape(-1, self.state_dim, self.obs_dim)
        correction = torch.bmm(K, inovation.unsqueeze(-1)).squeeze(-1)
        x_filtered_unclamped = x_predicted + correction
        
        # --- BLOK PRO OMEZENÍ STAVU (CLAMPING) ---
        px_clamped = x_filtered_unclamped[:, 0].clamp(self.system_model.min_x, self.system_model.max_x)
        py_clamped = x_filtered_unclamped[:, 1].clamp(self.system_model.min_y, self.system_model.max_y)
        max_vel = 100.0 # Nebo jiná rozumná hodnota
        vx_clamped = x_filtered_unclamped[:, 2].clamp(-max_vel, max_vel)
        vy_clamped = x_filtered_unclamped[:, 3].clamp(-max_vel, max_vel)

        # Logování (volitelné)
        # if torch.any((vx_clamped == max_vel) | (vx_clamped == -max_vel)):
        #     print(f"Varování: Došlo k omezení rychlosti v ose X (max_vel={max_vel}).")
        # if torch.any((vy_clamped == max_vel) | (vy_clamped == -max_vel)):
        #     print(f"Varování: Došlo k omezení rychlosti v ose Y (max_vel={max_vel}).")

        # Sestavení nového, ořezaného tenzoru
        x_filtered = torch.stack([px_clamped, py_clamped, vx_clamped, vy_clamped], dim=1)
        
        # 4. AKTUALIZACE STAVŮ PRO PŘÍŠTÍ VOLÁNÍ (pro t+1)
        # Posuneme stavy v čase
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.detach()
        self.x_filtered_t_minus_1 = x_filtered.detach() # .clone() není nutné
        self.x_pred_t_minus_1 = x_predicted.detach()
        self.y_t_minus_1 = y_t.detach()
        self.h_prev = h_new.detach()
        
        total_regularization = regs
        
        return x_filtered, total_regularization