
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
from .DNN_KalmanNet import DNN_KalmanNet 
from copy import deepcopy

class StateKalmanNet(nn.Module):
    
    def __init__(self, system_model, device, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4, 
                 num_gru_layers=1,
                 gru_hidden_dim_multiplier=1):
        
        super(StateKalmanNet, self).__init__()
        
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim


        self.dnn = DNN_KalmanNet(
            system_model, 
            hidden_size_multiplier=hidden_size_multiplier, 
            output_layer_multiplier=output_layer_multiplier, 
            num_gru_layers=num_gru_layers,
            gru_hidden_dim_multiplier=gru_hidden_dim_multiplier
        ).to(device)

        self.h_prev = None                  
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        """Initializes states for a new sequence (batch)."""
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_filtered_prev = initial_state.detach().clone()
        self.x_filtered_prev_prev = initial_state.detach().clone()
        self.x_pred_prev = initial_state.detach().clone()
        
        with torch.no_grad():
            self.y_prev = self.system_model.h(self.x_filtered_prev)
        
        self.h_prev = torch.zeros(
            self.dnn.gru.num_layers, 
            batch_size, 
            self.dnn.gru_hidden_dim, 
            device=self.device
        )

    def step(self, y_t_raw):
        """
        Performs one complete KalmanNet step (prediction + correction) at time 't'.
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        x_pred_raw = self.system_model.f(self.x_filtered_prev) # [B, m]
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]
        

        # mx = 1e+8
        # x_pred_raw = torch.clip(x_pred_raw, max=mx, min=-mx)
        innovation = y_t_raw - y_pred_raw
        
        # F1: Observation difference
        obs_diff = y_t_raw - self.y_prev

        # F3: Difference of posterior estimates
        # fw_evol_diff = x_pred_raw - self.x_filtered_prev_prev
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev

        # F4: Rozdíl (update)
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        if not torch.all(torch.isfinite(self.h_prev)):
            self._log_and_raise("self.h_prev (GRU input)", locals())
        if not torch.all(torch.isfinite(obs_diff)):
            self._log_and_raise("obs_diff (GRU input)", locals())
        if not torch.all(torch.isfinite(innovation)):
            self._log_and_raise("innovation (GRU input)", locals())
        
        K_vec, h_new = self.dnn(
            norm_obs_diff,       # F1
            norm_innovation,     # F2
            norm_fw_evol_diff,   # F3
            norm_fw_update_diff, # F4
            self.h_prev          # h_{t-1}
        )
        if not torch.all(torch.isfinite(K_vec)):
            self._log_and_raise("K_vec (DNN output)", locals())
        
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        if not torch.all(torch.isfinite(correction)):
            self._log_and_raise("correction (K @ innov)", locals())

        x_filtered_raw = x_pred_raw + correction
         
        if not torch.all(torch.isfinite(x_filtered_raw)):
            self._log_and_raise("x_filtered_raw (final state)", locals())

        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        self.y_prev = y_t_raw.clone()                    
        self.h_prev = h_new                            
        
        return x_filtered_raw

    def init_weights(self) -> None:
        """
        Key method for stability:
        Initialize layers normally, BUT zero (or near-zero) the output layer for K.
        """
        for name, m in self.dnn.named_modules():
            if isinstance(m, nn.Linear):
                if "output_final_linear" in name: 
                    init.constant_(m.weight, 0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0.0)
                    print(f"DEBUG: Layer '{name}' initialized near zero (Start K=0).")
                else:
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        init.zeros_(m.bias)
            
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, 0)
                init.constant_(m.weight, 1.0)
            
            elif isinstance(m, nn.GRU):
                for param_name, param in m.named_parameters():
                    if 'weight_ih' in param_name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in param_name:
                        init.orthogonal_(param.data)
                    elif 'bias' in param_name:
                        param.data.fill_(0)

    def _detach(self):
        """Detach all states carried across TBPTT windows."""
        self.h_prev = self.h_prev.detach()
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    def _log_and_raise(self, failed_tensor_name, local_vars):
        """
        Helper to print complete state on failure.
        Prints only the first 'max_elements' items from the batch for readability.
        """
        print(f"\n{'!'*40} FAILURE DETECTED {'!'*40}")
        print(f"Cause: Tensor '{failed_tensor_name}' contains NaN or Inf.")
        print(f"{'='*100}")
        
        max_elements = 4
        try:
            failed_value = local_vars.get(failed_tensor_name, "CANNOT GET VALUE")
            
            display_str = ""
            if isinstance(failed_value, torch.Tensor):
                batch_size = failed_value.shape[0] if failed_value.dim() > 0 else 1
                display_size = min(max_elements, batch_size)
                display_str = f"{failed_value[0:display_size]}"
                if batch_size > display_size:
                    display_str += f"\n... (showing first {display_size} of {batch_size} items)"
            else:
                display_str = str(failed_value)
                
            print(f"Failed tensor value [{failed_tensor_name}]:\n{display_str}\n")
        except Exception as e:
            print(f"Failed to print tensor value: {e}\n")

        print(f"{'-'*40} COMPLETE STATE AT FAILURE {'-'*40}")
        
        vars_to_log = [
            'y_t_raw', 
            'self.x_filtered_prev', 'self.y_prev', 'self.x_filtered_prev_prev', 
            'self.x_pred_prev', 'self.h_prev',
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