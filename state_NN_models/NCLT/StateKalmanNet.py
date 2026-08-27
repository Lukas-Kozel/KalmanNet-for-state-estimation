import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
from .DNN_KalmanNet import DNN_KalmanNetNCLT
from copy import deepcopy

class StateKalmanNetNCLT(nn.Module):
    
    def __init__(self, system_model, device, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4, 
                 num_gru_layers=1,
                 gru_hidden_dim_multiplier=1):
        
        super(StateKalmanNetNCLT, self).__init__()
        
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim


        self.dnn = DNN_KalmanNetNCLT(
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
        
    def step(self, y_t_raw, u_t=None): # Added u_t for control support
        """
        Performs one complete KalmanNet step (prediction + correction) at time 't'.
        Handles missing measurements (NaNs).
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        # 1. PREDICTION (Time Update) - Always runs
        # Use f(x, u) - if we have input u_t, use it
        x_pred_raw = self.system_model.f(self.x_filtered_prev, u_t) # [B, m]
        
        # Measurement prediction (expected y)
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]

        # === UPDATE START: Handling Missing Data (NaN) ===
        
        # A. Create mask (1 = data present, 0 = missing/NaN)
        # It is enough to look at first column of y (if one is missing, usually whole measurement is missing)
        mask = ~torch.isnan(y_t_raw[:, 0]).view(-1, 1) # [B, 1]

        # B. Replace NaN with safe value (e.g., y_pred or 0)
        # So we can compute differences without errors, replace NaN in y_t_raw
        # If measurement is missing, assume we measured exactly what we expected (y_pred)
        # Therefore innovation = 0.
        y_t_safe = torch.where(mask, y_t_raw, y_pred_raw)
        
        # C. Innovation (Residual)
        innovation = y_t_safe - y_pred_raw
        # For safety, zero innovation where mask is 0 (redundant but safe)
        innovation = innovation * mask 

        # D. Measurement difference (for network input F1)
        # If current y is missing, use previous y (or y_pred)
        # Here: y_t_safe is already corrected.
        obs_diff = y_t_safe - self.y_prev
        obs_diff = obs_diff * mask # Zero diff where there is no new measurement
        
        # === UPDATE END ===

        # F3: Difference of posterior estimates
        fw_evol_diff = self.x_filtered
