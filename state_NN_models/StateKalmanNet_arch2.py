import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
# Important: Import the new DNN class
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

        self.dnn = DNN_KalmanNet_arch2(
            system_model, 
            hidden_size_multiplier=hidden_size_multiplier, 
            output_layer_multiplier=output_layer_multiplier
        ).to(device)

        # Previous states for feature computation
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.h_prev_Q = None
        self.h_prev_Sigma = None
        self.h_prev_S = None
        
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        """Initialize states for a new sequence (batch)."""
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_filtered_prev = initial_state
        self.x_filtered_prev_prev = initial_state.clone()
        self.x_pred_prev = initial_state.clone()
        
        self.y_prev = self.system_model.h(self.x_filtered_prev)

        h_Q_init, h_Sigma_init, h_S_init = self.dnn.init_hidden_states(
            batch_size,
            prior_Q=self.system_model.Q,      # Matice Q z modelu
            prior_Sigma=self.system_model.P0, # Initial covariance P0
            prior_S=self.system_model.R       # Matice R z modelu
        )
        
        self.h_prev_Q = h_Q_init
        self.h_prev_Sigma = h_Sigma_init
        self.h_prev_S = h_S_init

    def step(self, y_t_raw, u_t_raw=None):
        """
        Perform one complete step (Architecture #2).
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        x_pred_raw = self.system_model.f(self.x_filtered_prev) # [B, m]

        if u_t_raw is not None:
            u_t_raw = u_t_raw.to(self.device)
            x_pred_raw = x_pred_raw + u_t_raw
        # mx = 1e+8
        # x_pred_raw = torch.clip(x_pred_raw, max=mx, min=-mx)
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]

        innovation_raw = y_t_raw - y_pred_raw
        
        # --- Compute 4 features ---
        obs_diff = y_t_raw - self.y_prev
        innovation = innovation_raw
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12) 
        # norm_obs_diff = obs_diff
        # norm_innovation = innovation
        # norm_fw_evol_diff = fw_evol_diff
        # norm_fw_update_diff = fw_update_diff

        if not torch.all(torch.isfinite(self.h_prev_Q)) or \
           not torch.all(torch.isfinite(self.h_prev_Sigma)) or \
           not torch.all(torch.isfinite(self.h_prev_S)):
            self._log_and_raise("One of the hidden states (h_prev_*) is NaN/Inf", locals())

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
            self._log_and_raise("K_vec (DNN output)", locals())
        

        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        correction = (K @ norm_innovation.unsqueeze(-1)).squeeze(-1)
        if not torch.all(torch.isfinite(correction)):
            self._log_and_raise("correction (K @ innov)", locals())

        x_filtered_raw = x_pred_raw + correction 
        if not torch.all(torch.isfinite(x_filtered_raw)):
            self._log_and_raise("x_filtered_raw (final state)", locals())


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
        """Detach all states carried across TBPTT windows."""
        self.h_prev_Q = self.h_prev_Q.detach()
        self.h_prev_Sigma = self.h_prev_Sigma.detach()
        self.h_prev_S = self.h_prev_S.detach()
        
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    def _log_and_raise(self, failed_tensor_name, local_vars):
        """
        Helper method to print the full state on failure.
        """
        print(f"\n{'!'*40} FAILURE DETECTED {'!'*40}")
        print(f"Cause: Tensor '{failed_tensor_name}' contains NaN or Inf.")
        print(f"{'='*100}")
        
        max_elements = 4 
        
        try:
            failed_value = local_vars.get(failed_tensor_name, "CANNOT RETRIEVE VALUE")
            display_str = ""
            if isinstance(failed_value, torch.Tensor):
                batch_size = failed_value.shape[0] if failed_value.dim() > 0 else 1
                display_size = min(max_elements, batch_size)
                display_str = f"{failed_value[0:display_size]}"
                if batch_size > display_size:
                    display_str += f"\n... (zobrazeno prvních {display_size} z {batch_size} prvků)"
            else:
                display_str = str(failed_value)
            print(f"Value of failed tensor [{failed_tensor_name}]:\n{display_str}\n")
        except Exception as e:
            print(f"Failed to print value of failed tensor: {e}\n")

        print(f"{'-'*40} COMPLETE STATE AT FAILURE {'-'*40}")
        
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
                source = "(local)"
            elif hasattr(self, var_name.split('.')[-1]):
                value = getattr(self, var_name.split('.')[-1])
                source = "(from self)"
            
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
                    print(f"Value: {display_str}")
                    print(f"Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    print(f"Contains NaN: {torch.any(torch.isnan(value)) if isinstance(value, torch.Tensor) else 'N/A'}")
                    print(f"Contains Inf: {torch.any(torch.isinf(value)) if isinstance(value, torch.Tensor) else 'N/A'}\n")
                except Exception as e:
                    print(f"--- {var_name} {source} ---")
                    print(f"Cannot print value: {e}\n")
            
        print(f"{'='*100}")
        
        raise RuntimeError(f"Failure: Tensor '{failed_tensor_name}' is NaN or Inf! (See log above)")