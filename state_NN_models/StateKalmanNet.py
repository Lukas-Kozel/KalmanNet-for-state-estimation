import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_KalmanNet import DNN_KalmanNet
import torch.nn.functional as func

class StateKalmanNet(nn.Module):
    def __init__(self, system_model, device, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4, 
                 num_gru_layers=1,
                 gru_hidden_dim_multiplier=1,returns_covariance=False):
        super(StateKalmanNet, self).__init__()

        self.returns_covariance = returns_covariance
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
        
    def step(self,y_t):
        """
        Provede jeden kompletní krok filtrace pro jedno měření y_t s dimenzí batch_size.
        """

        batch_size = y_t.shape[0]
        y_t = y_t.to(self.device)


        # 1. PREDIKCE (Time Update) - Vždy proběhne
        # Použijeme f(x, u) - pokud máme vstup u_t, použijeme ho
        x_pred_raw = self.system_model.f(self.x_filtered_prev) # [B, m]
        
        # Predikce měření (očekávané y)
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]
        # `y_predicted` Tvar: [batch_size, obs_dim]

        # Inovace: Rozdíl mezi skutečným a predikovaným měřením.
        # Tvar: [batch_size, obs_dim]
        innovation = y_t - y_pred_raw
        obs_diff = y_t - self.y_prev

        # F3: Difference of posterior estimates
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev

        # F4: Rozdíl (update)
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        # Normalizace (pokud používáš log, musíš ošetřit nuly/záporná čísla)
        # Zde necháváme identitu dle tvého kódu
        # norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        # norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        # norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        # norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)
        norm_obs_diff = obs_diff
        norm_innovation = innovation
        norm_fw_evol_diff = fw_evol_diff
        norm_fw_update_diff = fw_update_diff
        # normalizovaný rozdíl z minulého kroku
        K_vec, h_new = self.dnn(
            norm_obs_diff,       # F1
            norm_innovation,     # F2
            norm_fw_evol_diff,   # F3
            norm_fw_update_diff, # F4
            self.h_prev          # h_{t-1}
        )
        
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        # `K` Tvar: [batch_size, state_dim, obs_dim]

        # `innovation.unsqueeze(-1)` -> [batch_size, obs_dim, 1]
        # `K @ ...` -> [batch_size, state_dim, obs_dim] @ [batch_size, obs_dim, 1] -> [batch_size, state_dim, 1]
        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        x_filtered_raw = x_pred_raw + correction
        # `x_filtered` Tvar: [batch_size, state_dim]



        if self.returns_covariance:
            P_filtered_list = []
            with torch.no_grad(): # pro jistotu vypnutí gradienty
                R = self.system_model.R
                for i in range(batch_size):
                    K_i = K[i]  # Kalmanův zisk pro i-tý prvek v batche
                    
                    x_filtered_i = x_filtered_raw[i]

                    try:
                        if self.system_model.is_linear_h:
                            H_i = self.system_model.H
                        else:
                            H_i = torch.autograd.functional.jacobian(self.system_model.h, x_filtered_i).reshape(self.obs_dim, self.state_dim)
                        I = torch.eye(self.state_dim, device=self.device)
                        if self.state_dim == 1 or self.obs_dim == 1:
                            Htilde_i= 1/(H_i**2)
                            I_KH_i = (1 - K_i * H_i)
                            P_predict_i = 1/ (I_KH_i) * K_i * R * H_i * Htilde_i
                            P_filtered_i = I_KH_i * P_predict_i * I_KH_i + K_i * R * K_i
                            P_filtered_list.append(P_filtered_i)
                        else:    
                            Htilde_i = torch.linalg.inv(H_i.T @ H_i)
                            I_KH_i = (torch.eye(self.state_dim, device=self.device) - K_i @ H_i)
                            P_predict_i = torch.linalg.inv(I_KH_i) @ K_i @ R @ H_i @ Htilde_i
                            P_filtered_i = I_KH_i @ P_predict_i @ I_KH_i.T + K_i @ R @ K_i.T
                            P_filtered_list.append(P_filtered_i)

                    except:
                        print("Failed at Uncertainty Analysis of KalmanNet V1")
            P_filtered = torch.stack(P_filtered_list)

        # Update historie
        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        
        # Update y_prev: Pokud chybí data, držíme poslední známé (nebo y_pred)
        self.y_prev = y_t.clone()                    
        self.h_prev = h_new                            

        if self.returns_covariance:
            return x_filtered_raw, P_filtered
        else:
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