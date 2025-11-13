from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNet
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func

class StateBayesianKalmanNet_v2(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(StateBayesianKalmanNet_v2, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.returns_covariance = False

        self.dnn = DNN_BayesianKalmanNet(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        self.h_init_master = torch.randn(
            self.dnn.gru.num_layers, 
            1,
            self.dnn.gru.hidden_size, 
            device=self.device
        ).detach()

        self.h_prev = None                  
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.init_weights()


    def reset(self, batch_size=1, initial_state=None):
        """
        Inicializuje všechny stavy pro novou sekvenci (dávku).
        """
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        # Stavy jsou RAW
        self.x_filtered_prev = initial_state      # x_{t-1|t-1} (začíná jako x_0|0)
        self.x_filtered_prev_prev = initial_state.clone() # x_{t-2|t-2} (začíná jako x_0|0)
        
        with torch.no_grad():
            # Musíme spočítat y_prev (y_{t-1}) a x_pred_prev (x_{t-1|t-2})
            # Pro t=1 je x_{t-1|t-2} = x_{0|-1}
            # Pro jednoduchost můžeme předpokládat x_{0|-1} = x_0|0
            # a y_{t-1} = y_0 = h(x_0|0)
            self.x_pred_prev = initial_state.clone()        # x_{t-1|t-2} (začíná jako x_0|0)
            self.y_prev = self.system_model.h(self.x_filtered_prev) # y_{t-1} (začíná jako y_0)
        
        self.h_prev = torch.zeros(
            self.dnn.gru.num_layers, 
            batch_size, 
            self.dnn.gru.hidden_size, 
            device=self.device
        )

    def step(self, y_t):
        y_t_raw = y_t.to(self.device)
        # 1. PREDIKCE (pro aktuální čas t)
        # x_{t|t-1} = f(x_{t-1|t-1})
        x_pred_raw = self.system_model.f(self.x_filtered_prev)
        # y_{t|t-1} = h(x_{t|t-1})
        y_pred_raw = self.system_model.h(x_pred_raw)

        # Ošetření dimenzí
        if y_pred_raw.dim() == 1: y_pred_raw = y_pred_raw.unsqueeze(-1)
        if y_t_raw.dim() == 1: y_t_raw = y_t_raw.unsqueeze(-1)

        # 2. VÝPOČET VSTUPŮ PRO DNN
        # x_{t-1|t-1} - x_{t-1|t-2}
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev
        # y_t - y_{t|t-1}
        innovation_raw = y_t_raw - y_pred_raw
        # x_{t-1|t-1} - x_{t-2|t-2}
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev
        # y_t - y_{t-1}
        obs_diff = y_t_raw - self.y_prev

        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation_raw, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12) 
        if not torch.all(torch.isfinite(self.h_prev)):
            self._log_and_raise("self.h_prev (vstup GRU)", locals())
        if not torch.all(torch.isfinite(norm_obs_diff)):
            self._log_and_raise("norm_obs_diff (vstup GRU)", locals())
        if not torch.all(torch.isfinite(norm_innovation)):
            self._log_and_raise("norm_innovation (vstup GRU)", locals())
        
        # 3. PRŮCHOD SÍTÍ A KOREKCE
        K_vec, h_new, regs = self.dnn(norm_fw_update_diff, norm_innovation, norm_fw_evol_diff, norm_obs_diff, self.h_prev)
        
        K = K_vec.reshape(-1, self.state_dim, self.obs_dim)
        correction = (K @ norm_innovation.unsqueeze(-1)).squeeze(-1)
        if not torch.all(torch.isfinite(correction)):
            self._log_and_raise("correction (K @ innov)", locals())
        x_filtered = x_pred_raw + correction
        if not torch.all(torch.isfinite(x_filtered)):
            self._log_and_raise("x_filtered (finální stav)", locals())

        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered.clone()      
        self.y_prev = y_t_raw.clone()                    
        self.h_prev = h_new # Uložíme již oclampovaný stav!                     
        
        total_regularization = regs

        return x_filtered, total_regularization
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def _detach(self):
        """
        Odpojí všechny stavy přenášené mezi TBPTT okny.
        Toto je klíčová funkce pro `train_state_KalmanNet_sliding_window`.
        """
        # Clampujeme stavy PŘED odpojením
        self.h_prev = torch.clamp(self.h_prev.detach(), -100.0, 100.0)
        
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    def _log_and_raise(self, var_name, local_vars):
        """Pomocná funkce pro ladění NaN/Inf."""
        print(f"!!! CHYBA: Detekován NaN/Inf v '{var_name}' !!!")
        
        # Vypiš hodnoty všech rysů a stavů
        print("--- Stavy (t-1) ---")
        print(f"h_prev: {torch.isnan(local_vars['self'].h_prev).any()}, {torch.isinf(local_vars['self'].h_prev).any()}")
        print(f"y_prev: {torch.isnan(local_vars['self'].y_prev).any()}, {torch.isinf(local_vars['self'].y_prev).any()}")
        print(f"x_filtered_prev: {torch.isnan(local_vars['self'].x_filtered_prev).any()}, {torch.isinf(local_vars['self'].x_filtered_prev).any()}")
        print(f"x_filtered_prev_prev: {torch.isnan(local_vars['self'].x_filtered_prev_prev).any()}, {torch.isinf(local_vars['self'].x_filtered_prev_prev).any()}")
        print(f"x_pred_prev: {torch.isnan(local_vars['self'].x_pred_prev).any()}, {torch.isinf(local_vars['self'].x_pred_prev).any()}")
        
        print("--- Hodnoty (t) ---")
        print(f"y_t_raw: {torch.isnan(local_vars['y_t_raw']).any()}, {torch.isinf(local_vars['y_t_raw']).any()}")
        print(f"x_pred_raw: {torch.isnan(local_vars['x_pred_raw']).any()}, {torch.isinf(local_vars['x_pred_raw']).any()}")
        print(f"y_pred_raw: {torch.isnan(local_vars['y_pred_raw']).any()}, {torch.isinf(local_vars['y_pred_raw']).any()}")

        print("--- Rysy (RAW) ---")
        print(f"obs_diff: {torch.isnan(local_vars['obs_diff']).any()}, {torch.isinf(local_vars['obs_diff']).any()}")
        print(f"innovation_feat: {torch.isnan(local_vars['innovation_feat']).any()}, {torch.isinf(local_vars['innovation_feat']).any()}")
        print(f"fw_evol_diff: {torch.isnan(local_vars['fw_evol_diff']).any()}, {torch.isinf(local_vars['fw_evol_diff']).any()}")
        print(f"fw_update_diff: {torch.isnan(local_vars['fw_update_diff']).any()}, {torch.isinf(local_vars['fw_update_diff']).any()}")

        if 'K_vec' in local_vars:
             print(f"K_vec: {torch.isnan(local_vars['K_vec']).any()}, {torch.isinf(local_vars['K_vec']).any()}")
        
        raise RuntimeError(f"NaN/Inf detekován v: {var_name}")