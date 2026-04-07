from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNetTAN
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func

class StateBayesianKalmanNetTAN(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,gru_hidden_dim_multiplier=4,
                 init_min_dropout=0.5,init_max_dropout=0.8,use_log_modulus=False, use_terrain_grad=True): # <--- NEW ARGUMENT
        super(StateBayesianKalmanNetTAN, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.use_log_modulus = use_log_modulus
        self.use_terrain_grad = use_terrain_grad # <--- STORE FLAG

        # Pass the flag down to the Bayesian DNN
        self.dnn = DNN_BayesianKalmanNetTAN(
            system_model, 
            hidden_size_multiplier, 
            output_layer_multiplier, 
            num_gru_layers,
            gru_hidden_dim_multiplier, 
            init_min_dropout, 
            init_max_dropout,
            use_terrain_grad=self.use_terrain_grad # <--- PASSED TO DNN
        ).to(device)

        self.x_filtered_t_minus_1 = None
        self.x_filtered_t_minus_2 = None
        self.x_pred_t_minus_1 = None
        self.y_t_minus_1 = None
        self.h_init_master = torch.randn(
            self.dnn.gru.num_layers, 
            1,
            self.dnn.gru.hidden_size, 
            device=self.device
        ).detach()
        self.reset()
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is not None:
            self.x_filtered_t_minus_1 = initial_state.detach().clone()
        else:
            self.x_filtered_t_minus_1 = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.clone()
        
        x_pred_t_minus_1 = self.system_model.f(self.x_filtered_t_minus_2)
        self.x_pred_t_minus_1 = x_pred_t_minus_1.clone().detach()
        
        self.y_t_minus_1 = self.system_model.h(x_pred_t_minus_1).clone().detach()

        self.h_prev = self.h_init_master.expand(-1, batch_size, -1).clone()

    def step(self, y_t):
        # x_{t|t-1} = f(x_{t-1|t-1})
        x_predicted = self.system_model.f(self.x_filtered_t_minus_1)
        # y_{t|t-1} = h(x_{t|t-1})
        y_predicted = self.system_model.h(x_predicted)

        if y_predicted.dim() == 1: y_predicted = y_predicted.unsqueeze(-1)
        if y_t.dim() == 1: y_t = y_t.unsqueeze(-1)

        # === CONDITIONAL TERRAIN SLOPE EXTRACTION ===
        if self.use_terrain_grad:
            eps = 5.0 # 5-meter perturbation
            
            # Perturb X position
            x_pred_dx = x_predicted.clone()
            x_pred_dx[:, 0] += eps
            y_pred_dx = self.system_model.h(x_pred_dx)
            if y_pred_dx.dim() == 1: y_pred_dx = y_pred_dx.unsqueeze(-1)
            
            # Perturb Y position
            x_pred_dy = x_predicted.clone()
            x_pred_dy[:, 1] += eps
            y_pred_dy = self.system_model.h(x_pred_dy)
            if y_pred_dy.dim() == 1: y_pred_dy = y_pred_dy.unsqueeze(-1)
            
            # Calculate slope (d_Alt / d_X and d_Alt / d_Y)
            slope_x = (y_pred_dx[:, 0] - y_predicted[:, 0]) / eps
            slope_y = (y_pred_dy[:, 0] - y_predicted[:, 0]) / eps
            
            terrain_slope = torch.stack([slope_x, slope_y], dim=1)
        # ============================================

        # x_{t-1|t-1} - x_{t-1|t-2}
        state_inno = self.x_filtered_t_minus_1 - self.x_pred_t_minus_1
        inovation = y_t - y_predicted
        diff_state = self.x_filtered_t_minus_1 - self.x_filtered_t_minus_2
        diff_obs = y_t - self.y_t_minus_1

        # Normalization
        if self.use_log_modulus:
            norm_state_inno = self.log_modulus(state_inno)
            norm_innovation = self.log_modulus(inovation)
            norm_diff_state = self.log_modulus(diff_state)
            norm_diff_obs = self.log_modulus(diff_obs)
            if self.use_terrain_grad:
                norm_terrain_slope = self.log_modulus(terrain_slope)
        else:
            norm_state_inno = func.normalize(state_inno, p=2, dim=1, eps=1e-12)
            norm_innovation = func.normalize(inovation, p=2, dim=1, eps=1e-12)
            norm_diff_state = func.normalize(diff_state, p=2, dim=1, eps=1e-12)
            norm_diff_obs = func.normalize(diff_obs, p=2, dim=1, eps=1e-12)
            if self.use_terrain_grad:
                norm_terrain_slope = func.normalize(terrain_slope, p=2, dim=1, eps=1e-12)

        # Conditionally pass terrain_slope to the Bayesian DNN
        if self.use_terrain_grad:
            K_vec, h_new, regs = self.dnn(
                norm_state_inno, 
                norm_innovation, 
                norm_diff_state, 
                norm_diff_obs, 
                norm_terrain_slope, # <--- NEW FEATURE
                self.h_prev
            )
        else:
            K_vec, h_new, regs = self.dnn(
                norm_state_inno, 
                norm_innovation, 
                norm_diff_state, 
                norm_diff_obs, 
                self.h_prev
            )
        
        K = K_vec.reshape(-1, self.state_dim, self.obs_dim)
        correction = torch.bmm(K, inovation.unsqueeze(-1)).squeeze(-1)
        
        # x_{t|t} = x_{t|t-1} + K * inovation
        x_filtered = x_predicted + correction
        
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.detach()
        self.x_filtered_t_minus_1 = x_filtered.detach()
        self.x_pred_t_minus_1 = x_predicted.detach()
        self.y_t_minus_1 = y_t.detach()
        self.h_prev = h_new.detach()
        
        return x_filtered, regs
    
    def log_modulus(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    def init_weights(self) -> None:
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

            if hasattr(self.dnn, 'output_layer') and len(self.dnn.output_layer) > 0:
                last_layer = self.dnn.output_layer[-1] 
                if isinstance(last_layer, nn.Linear):
                    init.uniform_(last_layer.weight, -1e-1, 1e-1) 
                    if last_layer.bias is not None:
                        init.zeros_(last_layer.bias)
                    
    def detach_hidden(self):
        if self.x_filtered_t_minus_1 is not None:
            self.x_filtered_t_minus_1 = self.x_filtered_t_minus_1.detach()
        if self.x_filtered_t_minus_2 is not None:
            self.x_filtered_t_minus_2 = self.x_filtered_t_minus_2.detach()
        if self.x_pred_t_minus_1 is not None:
            self.x_pred_t_minus_1 = self.x_pred_t_minus_1.detach()
        if self.y_t_minus_1 is not None:
            self.y_t_minus_1 = self.y_t_minus_1.detach()
        if self.h_prev is not None:
            self.h_prev = self.h_prev.detach()