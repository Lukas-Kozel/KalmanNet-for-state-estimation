from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNetNCLT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class StateBayesianKalmanNetNCLT(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1, init_min_dropout=0.5, init_max_dropout=0.8):
        super(StateBayesianKalmanNetNCLT, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_BayesianKalmanNetNCLT(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        # Internal states
        self.x_filtered_t_minus_1 = None
        self.x_filtered_t_minus_2 = None
        self.x_pred_t_minus_1 = None
        self.y_t_minus_1 = None
        self.h_prev = None
        
        self.init_weights()

    def init_weights(self) -> None:
        """Stable weight initialization."""
        print("INFO: Applying 'Start Zero' initialization for Kalman Gain.")
        for name, m in self.dnn.named_modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None: init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, 0)
                init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.GRU):
                for param_name, param in m.named_parameters():
                    if 'weight' in param_name: init.xavier_uniform_(param.data)
                    elif 'bias' in param_name: param.data.fill_(0)

        # Initialize output layer close to zero
        if hasattr(self.dnn, 'output_layer') and len(self.dnn.output_layer) > 0:
            last_layer = self.dnn.output_layer[-1] 
            if isinstance(last_layer, nn.Linear):
                init.uniform_(last_layer.weight, -1e-3, 1e-3)
                if last_layer.bias is not None:
                    init.zeros_(last_layer.bias)
                print("DEBUG: Output layer zeroed (Soft Start).")

    def reset(self, batch_size=1, initial_state=None):
        """
        Initializes states for a new sequence.
        Unified with deterministic version for maximum stability.
        """
        if initial_state is not None:
            # Detach is important to avoid backpropagating gradients from previous batch
            self.x_filtered_t_minus_1 = initial_state.detach().clone().to(self.device)
        else:
            self.x_filtered_t_minus_1 = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # 1. State history: At start all are same as initial_state
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.clone()
        
        # 2. Prediction history: No need to call f(), just copy init state
        self.x_pred_t_minus_1 = self.x_filtered_t_minus_1.clone()
        
        # 3. Measurement history: Compute h(x)
        with torch.no_grad():
            self.y_t_minus_1 = self.system_model.h(self.x_filtered_t_minus_1).detach()
        
        # 4. GRU Hidden State: MUST BE ZERO!
        # Random initialization (randn) causes a "shock" in the first step.
        self.h_prev = torch.zeros(
            self.dnn.gru.num_layers, 
            batch_size, 
            self.dnn.gru.hidden_size, 
            device=self.device
        )
        
    def _detach(self):
        self.x_filtered_t_minus_1 = self.x_filtered_t_minus_1.detach()
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_2.detach()
        self.x_pred_t_minus_1 = self.x_pred_t_minus_1.detach()
        self.y_t_minus_1 = self.y_t_minus_1.detach()
        self.h_prev = self.h_prev.detach()

    def _check_tensor(self, tensor, name, verbose=False):
        """Helper function for NaN checking."""
        if torch.isnan(tensor).any():
            msg = f"CRASH: NaN detected in '{name}'"
            print(f"\n!!! {msg} !!!")
            print(f"  Shape: {tensor.shape}")
            if verbose: print(f"  Value sample: {tensor[0]}")
            raise ValueError(msg)
        if torch.isinf(tensor).any():
            msg = f"CRASH: Inf detected in '{name}'"
            print(f"\n!!! {msg} !!!")
            raise ValueError(msg)

    def step(self, y_t, u_t=None):
        if y_t.dim() == 1: y_t = y_t.unsqueeze(0)
        if u_t is not None and u_t.dim() == 1: u_t = u_t.unsqueeze(0)

        # --- 0. INPUT CHECK ---
        # self._check_tensor(y_t, "Input y_t (Measurement)")
        self._check_tensor(u_t, "Input u_t (Control)")
        self._check_tensor(self.x_filtered_t_minus_1, "Previous State x_t-1")

        # 1. PREDICTION (Physics)
        x_predicted = self.system_model.f(self.x_filtered_t_minus_1, u_t)
        self._check_tensor(x_predicted, "Physics Prediction (f)")

        # Process Noise
        
        if self.state_dim == 6:
            # Position/Velocity: Increase to 0.5 - 0.8 (to spread ensemble)
            # Angle: Increase only cosmetically to 0.05 (to avoid rotating robot)
            scale_vec = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2, 0.2], device=self.device)
            # scale_vec = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], device=self.device)
            
            
        elif self.state_dim == 3: 
            scale_vec = torch.tensor([0.6, 0.6, 0.05], device=self.device)
        else:
            scale_vec = torch.tensor([0.2] * self.state_dim, device=self.device)
            
        scale_vec = scale_vec.view(1, -1)
            
            # Noise application
        noise = torch.randn_like(x_predicted) * scale_vec
        x_predicted = x_predicted + noise
        self._check_tensor(x_predicted, "Prediction after Noise Injection")

        y_predicted = self.system_model.h(x_predicted)
        if y_predicted.dim() == 1: y_predicted = y_predicted.unsqueeze(-1)
        self._check_tensor(y_predicted, "Observation Model (h)")
        
        # --- NAN HANDLING ---
        # Here we check NaN in y_t but mask it. This is OK.
        is_nan_mask = torch.any(torch.isnan(y_t), dim=1, keepdim=True) 
        valid_mask = ~is_nan_mask 
        y_t_clean = torch.where(is_nan_mask, y_predicted.detach(), y_t)
        
        # 3. FEATURE COMPUTATION
        innovation = y_t_clean - y_predicted
        diff_obs = y_t_clean - self.y_t_minus_1
        diff_state = self.x_filtered_t_minus_1 - self.x_filtered_t_minus_2
        state_inno = self.x_filtered_t_minus_1 - self.x_pred_t_minus_1
        
        # Check features before network
        self._check_tensor(innovation, "Innovation Input")
        self._check_tensor(diff_obs, "Diff Obs Input")

        # --- NORMALIZATION ---
        # norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-6)
        # norm_diff_obs = F.normalize(diff_obs, p=2, dim=1, eps=1e-6)
        # norm_diff_state = F.normalize(diff_state, p=2, dim=1, eps=1e-6)
        # norm_state_inno = F.normalize(state_inno, p=2, dim=1, eps=1e-6)
        norm_innovation = innovation
        norm_diff_obs = diff_obs
        norm_diff_state = diff_state
        norm_state_inno = state_inno

        # 4. NETWORK FORWARD
        # Check hidden state before GRU input
        self._check_tensor(self.h_prev, "Hidden State Prev")
        
        K_vec, h_new, regs = self.dnn(norm_state_inno, norm_innovation, norm_diff_state, norm_diff_obs, self.h_prev)
        
        # Check network output
        self._check_tensor(K_vec, "DNN Output (K_vec)")
        self._check_tensor(h_new, "DNN Output (h_new)")

        K = K_vec.reshape(-1, self.state_dim, self.obs_dim)
        
        # 5. CORRECTION
        raw_correction = torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        self._check_tensor(raw_correction, "Raw Correction (K * Innovation)")
        
        correction = raw_correction * valid_mask.float() 
        # correction = torch.clamp(correction, min=-10.0, max=10.0)
        # WARNING: If you commented out clamp, explosion may occur here
        # I recommend keeping at least a soft clamp for debug
        # correction = torch.clamp(correction, min=-50.0, max=50.0) 

        x_filtered = x_predicted + correction
        
        # Final check
        self._check_tensor(x_filtered, "Final Filtered State (x_filtered)")
        
        # 6. UPDATE
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.clone() 
        self.x_filtered_t_minus_1 = x_filtered
        self.x_pred_t_minus_1 = x_predicted
        self.h_prev = h_new
        self.y_t_minus_1 = torch.where(is_nan_mask, y_predicted.detach(), y_t.detach())
        
        return x_filtered, regs
