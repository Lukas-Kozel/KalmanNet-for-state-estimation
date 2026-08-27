import torch.nn as nn
from .DNN_KalmanFormer import DNN_KalmanFormerNCLT
import torch.nn.functional as func
import torch.nn.init as init
import torch

class KalmanFormerNCLT(nn.Module):
    
    def __init__(self, system_model, device, 
                 d_model=64, nhead=4, num_encoder_layers=1, 
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1):
        
        super(KalmanFormerNCLT, self).__init__()
        
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_KalmanFormerNCLT(
            system_model,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)

        # Buffer for historical values
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

    def step(self, y_t_raw, u_t=None):
        """
        Robust step of KalmanFormer.
        Instead of 'if' branching uses masking to avoid breaking gradient graph.
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        # 1. PREDICTION (Always the same)
        x_pred_raw = self.system_model.f(self.x_filtered_prev, u_t) # [B, m]
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]
        
        # 2. MASKING NaN
        # Create mask: True where data IS valid, False where NaN
        # Assume that if NaN is in any channel, measurement is invalid
        is_nan_mask = torch.any(torch.isnan(y_t_raw), dim=1, keepdim=True) # [B, 1]
        valid_mask = ~is_nan_mask # [B, 1] (1 = valid, 0 = invalid)

        # Create "cleaned" input y_t
        # Where NaN is, insert y_pred (innovation will be 0, network won't go crazy)
        # This keeps graph continuity.
        y_t_clean = torch.where(is_nan_mask, y_pred_raw, y_t_raw)
        
        # 3. FEATURE COMPUTATION (always with clean data)
        
        # F1: Observation difference
        # If last time was NaN, y_prev was set smartly (see end of function)
        obs_diff = y_t_clean - self.y_prev
        
        # F2: Innovation
        # If NaN, y_t_clean == y_pred_raw, so innovation is 0 (which is correct)
        innovation = y_t_clean - y_pred_raw
        
        # F3 & F4: State diffs
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        # Normalization (L2)
        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12) 
        # norm_obs_diff = obs_diff
        # norm_innovation = innovation
        # norm_fw_evol_diff = fw_evol_diff
        # norm_fw_update_diff = fw_update_diff
        
        # 4. DNN PASS (Runs always, even for NaN data!)
        # This ensures that 'K_vec' is always part of computational graph
        K_vec = self.dnn(
            norm_obs_diff, norm_innovation, 
            norm_fw_evol_diff, norm_fw_update_diff
        )
        
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        # 5. CORRECTION WITH MASK
        # Compute correction (for NaN data K * 0 = 0, but we secure it)
        raw_correction = (K @ norm_innovation.unsqueeze(-1)).squeeze(-1)
        
        # Apply mask: 
        # Where valid_mask=1, correction passes. 
        # Where valid_mask=0 (NaN), correction is zeroed.
        # Important: This multiplication keeps graph connected (multiplying by zero is valid autograd operation).
        correction = raw_correction * valid_mask.float()

        x_filtered_raw = x_pred_raw + correction 

        # 6. UPDATE HISTORY
        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        
        # Update y_prev
        # If measurement was valid, store it.
        # If NaN, store 'y_pred_raw' (our prediction), so in next step
        # 'obs_diff' (y_t - y_prev) does not give nonsense.
        self.y_prev = torch.where(is_nan_mask, y_pred_raw.detach(), y_t_raw.detach())
        
        return x_filtered_raw

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.TransformerEncoderLayer) or isinstance(m, nn.TransformerDecoderLayer):
                 for name, param in m.named_parameters():
                    if 'weight' in name:
                        if param.dim() > 1:
                            init.xavier_uniform_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def _detach(self):
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    def _log_and_raise(self, failed_tensor_name, local_vars):
        # Logging can remain same if you want it
        pass
