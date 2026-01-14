from .DNN_BayesianKalmanNet_test import DNN_BayesianKalmanNetNCLT_test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
import numpy as np

class StateBayesianKalmanNetNCLT_Diagnostic(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,
                  init_min_dropout=0.5, init_max_dropout=0.8, process_jitter=[0.1, 0.1, 0.1, 0.1, 0.01, 0.01]):
        super(StateBayesianKalmanNetNCLT_Diagnostic, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        
        # Nový hyperparametr: Process Noise Std (Jitter)
        if isinstance(process_jitter, (list, tuple)):
            self.process_jitter = torch.tensor(process_jitter, device=device).float()
        else:
            self.process_jitter = process_jitter.to(device) # Pokud už je tensor
            
        # Reshape na [1, Dim] pro broadcasting (aby to šlo přičíst k batchi)
        self.process_jitter = self.process_jitter.view(1, -1)
        self.dnn = DNN_BayesianKalmanNetNCLT_test(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        # Interní stavy
        self.x_filtered_t_minus_1 = None
        self.x_filtered_t_minus_2 = None
        self.x_pred_t_minus_1 = None
        self.y_t_minus_1 = None
        self.h_prev = None

        self.h_init_master = torch.randn(
            self.dnn.gru.num_layers, 
            1,
            self.dnn.gru.hidden_size, 
            device=self.device
        ).detach()
        
        # --- DIAGNOSTIKA ---
        self.diagnostics_enabled = False
        self.history = defaultdict(list)

        self.init_weights()

    def init_weights(self) -> None:
        """Stabilní inicializace vah."""
        print("INFO: Aplikuji 'Start Zero' inicializaci pro Kalman Gain.")
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

        # Inicializace výstupní vrstvy blízko nule
        if hasattr(self.dnn, 'output_layer') and len(self.dnn.output_layer) > 0:
            last_layer = self.dnn.output_layer[-1] 
            if isinstance(last_layer, nn.Linear):
                init.uniform_(last_layer.weight, -1e-3, 1e-3)
                if last_layer.bias is not None:
                    init.zeros_(last_layer.bias)
                print("DEBUG: Výstupní vrstva vynulována (Soft Start).")

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is not None:
            initial_state = initial_state.clone().to(self.device)
        else:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        self.x_filtered_t_minus_1 = initial_state.detach().clone()
        self.x_filtered_t_minus_2 = initial_state.detach().clone()
        self.x_pred_t_minus_1 = initial_state.detach().clone()
        
        with torch.no_grad():
            self.y_t_minus_1 = self.system_model.h(self.x_filtered_t_minus_1)
        
        self.h_prev = torch.zeros(
            self.dnn.gru.num_layers, 
            batch_size, 
            self.dnn.gru.hidden_size, 
            device=self.device
        )
        
        if self.diagnostics_enabled:
            self.history.clear()
        
    def _detach(self):
        self.x_filtered_t_minus_1 = self.x_filtered_t_minus_1.detach()
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_2.detach()
        self.x_pred_t_minus_1 = self.x_pred_t_minus_1.detach()
        self.y_t_minus_1 = self.y_t_minus_1.detach()
        self.h_prev = self.h_prev.detach()

    def _check_tensor(self, tensor, name, verbose=False):
        if torch.isnan(tensor).any():
            msg = f"CRASH: NaN detected in '{name}'"
            print(f"\n!!! {msg} !!!")
            raise ValueError(msg)
        if torch.isinf(tensor).any():
            msg = f"CRASH: Inf detected in '{name}'"
            print(f"\n!!! {msg} !!!")
            raise ValueError(msg)

    def set_diagnostics(self, enabled: bool):
        self.diagnostics_enabled = enabled
        if not enabled:
            self.history.clear()

    def step(self, y_t, u_t=None):
        if y_t.dim() == 1: y_t = y_t.unsqueeze(0)
        if u_t is not None and u_t.dim() == 1: u_t = u_t.unsqueeze(0)

        # --- 0. KONTROLA VSTUPŮ ---
        self._check_tensor(u_t, "Input u_t (Control)")
        self._check_tensor(self.x_filtered_t_minus_1, "Previous State x_t-1")

        # 1. PREDIKCE (Fyzika)
        # Deterministická část
        x_predicted_det = self.system_model.f(self.x_filtered_t_minus_1, u_t)
        self._check_tensor(x_predicted_det, "Physics Prediction (f)")
        
        noise = torch.randn_like(x_predicted_det) * self.process_jitter
        
        x_predicted = x_predicted_det + noise

        y_predicted = self.system_model.h(x_predicted)
        if y_predicted.dim() == 1: y_predicted = y_predicted.unsqueeze(-1)
        self._check_tensor(y_predicted, "Observation Model (h)")
        
        # --- NAN HANDLING ---
        mask = ~torch.isnan(y_t[:, 0]).view(-1, 1)
        y_t_safe = torch.where(mask, y_t, y_predicted)
        innovation = (y_t_safe - y_predicted) * mask 

        # D. Rozdíl měření
        obs_diff = (y_t_safe - self.y_t_minus_1) * mask
        
        # === UPDATE END ===
        diff_state = self.x_filtered_t_minus_1 - self.x_filtered_t_minus_2
        state_inno = self.x_filtered_t_minus_1 - self.x_pred_t_minus_1
        
        # Normalizace
        norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_diff_obs = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_diff_state = F.normalize(diff_state, p=2, dim=1, eps=1e-12)
        norm_state_inno = F.normalize(state_inno, p=2, dim=1, eps=1e-12)

        # 4. NETWORK FORWARD
        K_vec, h_new, regs = self.dnn(norm_state_inno, norm_innovation, norm_diff_state, norm_diff_obs, self.h_prev)
        
        K = K_vec.reshape(-1, self.state_dim, self.obs_dim)
        mask_expanded = mask.unsqueeze(2) 
        
        K_masked = K * mask_expanded
        
        # 5. KOREKCE
        raw_correction = torch.bmm(K_masked, innovation.unsqueeze(-1)).squeeze(-1)
        x_filtered = x_predicted + raw_correction
        
        # --- DIAGNOSTIKA ---
        if self.diagnostics_enabled:
            with torch.no_grad():
                self.history['K_raw'].append(K.detach().cpu().numpy())       
                self.history['K_masked'].append(K_masked.detach().cpu().numpy()) 
                self.history['innovation'].append(innovation.detach().cpu().numpy()) 
                self.history['correction'].append(raw_correction.detach().cpu().numpy()) 
                self.history['mask'].append(mask.detach().cpu().numpy())

        # 6. AKTUALIZACE
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.clone() 
        self.x_filtered_t_minus_1 = x_filtered.clone()
        self.x_pred_t_minus_1 = x_predicted.clone()
        self.h_prev = h_new
        self.y_t_minus_1 = y_t_safe.clone()
        
        return x_filtered, regs