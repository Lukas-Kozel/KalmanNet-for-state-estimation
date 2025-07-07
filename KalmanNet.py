import torch
import torch.nn as nn

class KalmanNet(nn.Module):
    def __init__(self, system_model, hidden_dim=128):
        super(KalmanNet, self).__init__()
        
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.hidden_dim = hidden_dim

        # Dimenze vstupu = dim(F4) + dim(F2)
        input_dim = self.state_dim + self.obs_dim

        # Architektura sítě zůstává stejná
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, self.state_dim * self.obs_dim)

    def forward(self, y_seq):
        """
        Provede filtraci celé sekvence měření y_seq.
        Používá explicitní výpočet F4 (delta_x_hat_update) a F2 (innovation) jako vstupy.
        """
        device = self.input_layer.weight.device
        batch_size, seq_len, _ = y_seq.shape
        
        # x_hat_prev_posterior odpovídá ˆx_{t-1|t-1}
        x_hat_prev_posterior = torch.zeros(batch_size, self.state_dim, device=device)
        
        # delta_x_hat_update_prev odpovídá ∆ˆx_{t-1}
        delta_x_hat_update_prev = torch.zeros(batch_size, self.state_dim, device=device)


        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        x_hat_list = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            F = self.system_model.F.to(device)
            H = self.system_model.H.to(device)

            # --- Predikce ---
            # x_hat_priori odpovídá ˆx_{t|t-1}
            x_hat_priori = (F @ x_hat_prev_posterior.unsqueeze(-1)).squeeze(-1)
            
            # y_hat odpovídá ŷ_t
            y_hat = (H @ x_hat_priori.unsqueeze(-1)).squeeze(-1)
            
            # --- Výpočet vstupů pro síť (F2 a F4) ---
            # innovation odpovídá ∆y_t
            innovation = y_t - y_hat
            
            nn_input = torch.cat([delta_x_hat_update_prev, innovation], dim=1)

            # --- Průchod sítí a výpočet Kalmanova zisku ---
            out_input_layer = self.input_layer(nn_input)
            out_gru, h = self.gru(out_input_layer.unsqueeze(0), h)
            out_output_layer = self.output_layer(out_gru.squeeze(0))
            
            K = out_output_layer.reshape(batch_size, self.state_dim, self.obs_dim)

            # --- Aktualizace odhadu stavu (Update step) ---
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            
            # Finální (posteriori) odhad stavu pro čas t
            # x_hat_posterior odpovídá ˆx_{t|t}
            x_hat_posterior = x_hat_priori + correction
            
            # delta_x_hat_update odpovídá ∆ˆx_t = ˆx_{t|t} - ˆx_{t|t-1}
            delta_x_hat_update = x_hat_posterior - x_hat_priori
            
            delta_x_hat_update_prev = delta_x_hat_update.clone()
            x_hat_prev_posterior = x_hat_posterior.clone()
            
            x_hat_list.append(x_hat_posterior)

        return torch.stack(x_hat_list, dim=1)