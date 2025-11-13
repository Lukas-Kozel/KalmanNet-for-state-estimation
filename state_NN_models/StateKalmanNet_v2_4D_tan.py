
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
from .DNN_KalmanNet_v2 import DNN_KalmanNet_v2
from copy import deepcopy

class StateKalmanNet_v2_4D_tan(nn.Module): 
    def __init__(self, system_model, device, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4, 
                 num_gru_layers=1):

        super(StateKalmanNet_v2_4D_tan, self).__init__()

        self.returns_covariance = False
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim


        self.dnn = DNN_KalmanNet_v2(
            system_model, 
            hidden_size_multiplier=hidden_size_multiplier, 
            output_layer_multiplier=output_layer_multiplier, 
            num_gru_layers=num_gru_layers
        ).to(device)

        self.h_prev = None                  
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_filtered_prev = initial_state
        self.x_filtered_prev_prev = initial_state.clone()
        self.x_pred_prev = initial_state.clone()
        
        with torch.no_grad():
            self.y_prev = self.system_model.h(self.x_filtered_prev)
        
        self.h_prev = torch.zeros(
            self.dnn.gru.num_layers, 
            batch_size, 
            self.dnn.gru_hidden_dim, 
            device=self.device
        )
    def step(self, y_t_raw):

        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        x_pred_raw = self.system_model.f(self.x_filtered_prev) # [B, m]
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]

        y_pred_nan_mask = torch.isnan(y_pred_raw) | torch.isinf(y_pred_raw)
        y_t_nan_mask = torch.isnan(y_t_raw) | torch.isinf(y_t_raw)
        
        y_pred_safe = torch.where(y_pred_nan_mask, y_t_raw, y_pred_raw)
        y_t_safe = torch.where(y_t_nan_mask, y_pred_safe, y_t_raw)
        
        innovation_raw = y_t_safe - y_pred_safe
        
        innovation_safe = innovation_raw

        # F1: Rozdíl pozorování
        obs_diff = y_t_safe - self.y_prev

        # F2: Inovace
        innovation = innovation_safe

        # F3: Rozdíl posterior odhadů
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev

        # F4: Rozdíl (update)
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        if torch.any(torch.isnan(self.h_prev)):
            raise RuntimeError(f"Selhání: h_prev (vstup GRU) je NaN!")
        if torch.any(torch.isinf(self.h_prev)):
            raise RuntimeError(f"Selhání: h_prev (vstup GRU) je Inf!")
        
        K_vec, h_new = self.dnn(
            norm_obs_diff,       # F1 (L2-norm)
            norm_innovation,     # F2 (L2-norm)
            norm_fw_evol_diff,   # F3 (L2-norm)
            norm_fw_update_diff, # F4 (L2-norm)
            self.h_prev          # h_{t-1}
        )
        if torch.any(torch.isnan(K_vec)):
            raise RuntimeError("Selhání: K_vec (výstup DNN) je NaN!")
        if torch.any(torch.isinf(K_vec)):
            raise RuntimeError("Selhání: K_vec (výstup DNN) je Inf!")
        
        if torch.any(torch.isnan(innovation_safe)):
            raise RuntimeError("Selhání: innovation_safe je NaN!")
        if torch.any(torch.isinf(innovation_safe)):
            raise RuntimeError("Selhání: innovation_safe je Inf!")
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        correction = (K @ innovation_safe.unsqueeze(-1)).squeeze(-1)

        if torch.any(torch.isnan(correction)):
            raise RuntimeError("Selhání: 'correction' (K @ innov) je NaN!")
        if torch.any(torch.isinf(correction)):
            raise RuntimeError("Selhání: 'correction' (K @ innov) je Inf!")
        x_filtered_raw = x_pred_raw + correction 

        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        self.y_prev = y_t_safe.clone()                    
        self.h_prev = h_new                            
        
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
        self.h_prev = self.h_prev.detach()
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()