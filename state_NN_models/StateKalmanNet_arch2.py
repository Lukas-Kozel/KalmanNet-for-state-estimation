import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
# Důležité: Importujte novou DNN třídu
from .DNN_KalmanNet_arch2 import DNN_KalmanNet_arch2 
from copy import deepcopy

class StateKalmanNet_arch2(nn.Module):
    
    def __init__(self, system_model, device, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4):
        
        super(StateKalmanNet_arch2, self).__init__()
        
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        # Instance nové DNN (Arch 2)
        self.dnn = DNN_KalmanNet_arch2(
            system_model, 
            hidden_size_multiplier=hidden_size_multiplier, 
            output_layer_multiplier=output_layer_multiplier
        ).to(device)

        # Předchozí stavy pro výpočet features
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.h_prev_Q = None
        self.h_prev_Sigma = None
        self.h_prev_S = None
        
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        """Inicializuje stavy pro novou sekvenci (dávku)."""
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_filtered_prev = initial_state
        self.x_filtered_prev_prev = initial_state.clone()
        self.x_pred_prev = initial_state.clone()
        
        self.y_prev = self.system_model.h(self.x_filtered_prev)

        self.h_prev_Q = torch.zeros(
            self.dnn.GRU_Q.num_layers, # 1
            batch_size, 
            self.dnn.d_hidden_Q,       # m*m
            device=self.device
        )
        self.h_prev_Sigma = torch.zeros(
            self.dnn.GRU_Sigma.num_layers, # 1
            batch_size, 
            self.dnn.d_hidden_Sigma,   # m*m
            device=self.device
        )
        self.h_prev_S = torch.zeros(
            self.dnn.GRU_S.num_layers, # 1
            batch_size, 
            self.dnn.d_hidden_S,       # n*n
            device=self.device
        )

    def step(self, y_t_raw):
        """
        Provede jeden kompletní krok (Architektura #2).
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        x_pred_raw = self.system_model.f(self.x_filtered_prev) # [B, m]
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]

        innovation_raw = y_t_raw - y_pred_raw
        
        # --- Výpočet 4 features ---
        obs_diff = y_t_raw - self.y_prev
        innovation = innovation_raw
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12) 

        if not torch.all(torch.isfinite(self.h_prev_Q)) or \
           not torch.all(torch.isfinite(self.h_prev_Sigma)) or \
           not torch.all(torch.isfinite(self.h_prev_S)):
            self._log_and_raise("Jeden ze skrytých stavů (h_prev_*) je NaN/Inf", locals())

        if not torch.all(torch.isfinite(norm_innovation)):
            self._log_and_raise("norm_innovation (vstup GRU)", locals())


        K_vec, h_new_Q, h_new_Sigma, h_new_S = self.dnn(
            norm_obs_diff,       
            norm_innovation,     
            norm_fw_evol_diff,   
            norm_fw_update_diff, 
            self.h_prev_Q,
            self.h_prev_Sigma,
            self.h_prev_S
        )
        
        if not torch.all(torch.isfinite(K_vec)):
            self._log_and_raise("K_vec (výstup DNN)", locals())
        

        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        if not torch.all(torch.isfinite(correction)):
            self._log_and_raise("correction (K @ innov)", locals())

        x_filtered_raw = x_pred_raw + correction 
        if not torch.all(torch.isfinite(x_filtered_raw)):
            self._log_and_raise("x_filtered_raw (finální stav)", locals())


        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        self.y_prev = y_t_raw.clone()                    
        

        self.h_prev_Q = h_new_Q
        self.h_prev_Sigma = h_new_Sigma
        self.h_prev_S = h_new_S
        
        return x_filtered_raw

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
        """Odpojí všechny stavy přenášené mezi TBPTT okny."""
        self.h_prev_Q = self.h_prev_Q.detach()
        self.h_prev_Sigma = self.h_prev_Sigma.detach()
        self.h_prev_S = self.h_prev_S.detach()
        
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    def _log_and_raise(self, failed_tensor_name, local_vars):
        """
        Helper metoda pro vypsání kompletního stavu při selhání.
        """
        print(f"\n{'!'*40} SELHÁNÍ DETEKOVÁNO {'!'*40}")
        print(f"Příčina: Tensor '{failed_tensor_name}' obsahuje NaN nebo Inf.")
        print(f"{'='*100}")
        
        max_elements = 4 
        
        try:
            failed_value = local_vars.get(failed_tensor_name, "NELZE ZÍSKAT HODNOTU")
            display_str = ""
            if isinstance(failed_value, torch.Tensor):
                batch_size = failed_value.shape[0] if failed_value.dim() > 0 else 1
                display_size = min(max_elements, batch_size)
                display_str = f"{failed_value[0:display_size]}"
                if batch_size > display_size:
                    display_str += f"\n... (zobrazeno prvních {display_size} z {batch_size} prvků)"
            else:
                display_str = str(failed_value)
            print(f"Hodnota selhaného tensoru [{failed_tensor_name}]:\n{display_str}\n")
        except Exception as e:
            print(f"Nepodařilo se vypsat hodnotu selhaného tensoru: {e}\n")

        print(f"{'-'*40} KOMPLETNÍ STAV V OKAMŽIKU SELHÁNÍ {'-'*40}")
        
        vars_to_log = [
            'y_t_raw', 
            'self.x_filtered_prev', 'self.y_prev', 'self.x_filtered_prev_prev', 
            'self.x_pred_prev', 'self.h_prev_Q', 'self.h_prev_Sigma', 'self.h_prev_S',
            'x_pred_raw', 'y_pred_raw', 'innovation_raw', 'obs_diff', 
            'fw_evol_diff', 'fw_update_diff',
            'norm_obs_diff', 'norm_innovation', 'norm_fw_evol_diff', 'norm_fw_update_diff',
            'K_vec', 'correction', 'x_filtered_raw'
        ]
        
        for var_name in vars_to_log:
            value = None
            source = ""
            if var_name in local_vars:
                value = local_vars[var_name]
                source = "(lokální)"
            elif hasattr(self, var_name.split('.')[-1]):
                value = getattr(self, var_name.split('.')[-1])
                source = "(ze self)"
            
            if value is not None:
                try:
                    print(f"--- {var_name} {source} ---")
                    display_str = ""
                    if isinstance(value, torch.Tensor):
                        batch_size = value.shape[0] if value.dim() > 0 else 1
                        display_size = min(max_elements, batch_size)
                        sliced_value = value[0:display_size]
                        display_str = f"{sliced_value}"
                        if batch_size > display_size:
                            display_str += f"\n... (zobrazeno prvních {display_size} z {batch_size} prvků)"
                    else:
                        display_str = str(value)
                    print(f"Hodnota: {display_str}")
                    print(f"Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    print(f"Obsahuje NaN: {torch.any(torch.isnan(value)) if isinstance(value, torch.Tensor) else 'N/A'}")
                    print(f"Obsahuje Inf: {torch.any(torch.isinf(value)) if isinstance(value, torch.Tensor) else 'N/A'}\n")
                except Exception as e:
                    print(f"--- {var_name} {source} ---")
                    print(f"Nelze vypsat hodnotu: {e}\n")
            
        print(f"{'='*100}")
        
        raise RuntimeError(f"Selhání: Tensor '{failed_tensor_name}' je NaN nebo Inf! (Viz log výše)")