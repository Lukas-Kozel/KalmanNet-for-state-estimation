import torch
import torch.nn as nn

class KalmanNet(nn.Module):
    """    KalmanNet Architektura pro nelineární Kalmanův filtr.
    """
    def __init__(self, system_model,  hidden_size_multiplier=20):
        super(KalmanNet, self).__init__()
        
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim


        # heuristika o velikosti GRU z článku
        moments_dim = self.state_dim*self.state_dim + self.obs_dim*self.obs_dim
        self.hidden_dim = moments_dim * hidden_size_multiplier

        # Dimenze vstupu je stále stejná: dim(F4) + dim(F2)
        input_dim = self.state_dim + self.obs_dim

        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.state_dim * self.obs_dim)

        # Vstup pro f bude mít tvar (batch, state_dim), pro h také.
        self.f_vmap = torch.vmap(self.system_model.f, in_dims=(0,))
        self.h_vmap = torch.vmap(self.system_model.h, in_dims=(0,))


    def forward(self, y_seq):

        device = self.input_layer.weight.device
        batch_size, seq_len, _ = y_seq.shape
        
        # Posteriori odhad stavu z předchozího kroku (ˆx_{t-1|t-1})
        x_hat_prev_posterior = torch.zeros(batch_size, self.state_dim, device=device)
        
        # F4 z předchozího kroku (∆ˆx_{t-1})
        delta_x_hat_update_prev = torch.zeros(batch_size, self.state_dim, device=device)

        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        x_hat_list = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            # --- Predikce (nelineární) ---
            # 1. Predikce stavu pomocí vektorizované funkce f
            #    x_hat_priori odpovídá ˆx_{t|t-1} = f(ˆx_{t-1|t-1})
            x_hat_priori = self.f_vmap(x_hat_prev_posterior)
            
            # 2. Predikce měření pomocí vektorizované funkce h
            #    y_hat odpovídá ŷ_t = h(ˆx_{t|t-1})
            y_hat = self.h_vmap(x_hat_priori)
            
            # --- Výpočet vstupů pro síť (F2 a F4) ---
            innovation = y_t - y_hat
            
            nn_input = torch.cat([delta_x_hat_update_prev, innovation], dim=1)

            # --- Průchod sítí a výpočet Kalmanova zisku ---
            out_input_layer = self.input_layer(nn_input)
            out_gru, h = self.gru(out_input_layer.unsqueeze(0), h)
            out_output_layer = self.output_layer(out_gru.squeeze(0))
            
            K = out_output_layer.reshape(batch_size, self.state_dim, self.obs_dim)

            # --- Aktualizace odhadu stavu (Update step) ---
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            
            # Finální (posteriori) odhad stavu (ˆx_{t|t})
            x_hat_posterior = x_hat_priori + correction

            # delta_x_hat_update odpovídá ∆ˆx_t = ˆx_{t|t} - ˆx_{t|t-1}
            delta_x_hat_update = x_hat_posterior - x_hat_priori
            
            delta_x_hat_update_prev = delta_x_hat_update.clone()
            x_hat_prev_posterior = x_hat_posterior.clone()
            
            x_hat_list.append(x_hat_posterior)

        return torch.stack(x_hat_list, dim=1)