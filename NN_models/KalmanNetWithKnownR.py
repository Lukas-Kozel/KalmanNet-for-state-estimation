import torch
import torch.nn as nn
import torch.nn.functional as F

class KalmanNetWithKnownR(nn.Module):
    """
    KalmanNet Architektura #1 s tím, že je známa kovarianční matice měření R.
    """
    def __init__(self, system_model, device, hidden_size_multiplier=10):
        super(KalmanNetWithKnownR, self).__init__()
        
        self.device = device
        
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        # Heuristika pro velikost skrytého stavu
        moments_dim = self.state_dim*self.state_dim + self.obs_dim*self.obs_dim
        self.hidden_dim = moments_dim * hidden_size_multiplier

        # Vstupní dimenze
        input_dim = self.state_dim + self.obs_dim

        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.state_dim * self.obs_dim)

        # vmap pro nelineární funkce
        self.f = system_model.f
        self.h = system_model.h

        self.R = system_model.R # TODO: upravit, aby model znal vzdy jen diagonalu matice R (aby to co nejlepe reflektovalo realne podminky)
    

    def forward(self, y_seq):
        batch_size, seq_len, _ = y_seq.shape
        
        x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        delta_x_hat_update_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        h = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

        x_hat_list = []
        P_predict_list = []
        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            # Predikce
            x_predicted_list = [self.f(x.unsqueeze(-1)) for x in x_filtered_prev]
            x_predicted = torch.stack(x_predicted_list).squeeze(-1)

            y_predicted_list = [self.h(x.unsqueeze(-1)) for x in x_predicted]
            y_predicted = torch.stack(y_predicted_list).squeeze(-1)

            # Výpočet vstupů
            innovation = y_t - y_predicted


            norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
            norm_delta_x = F.normalize(delta_x_hat_update_prev, p=2, dim=1, eps=1e-12)

            nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

            # Průchod sítí
            out_input_layer = self.input_layer(nn_input)
            activated_input = F.relu(out_input_layer) 
            out_gru, h = self.gru(activated_input.unsqueeze(0), h)
            out_output_layer = self.output_layer(out_gru.squeeze(0))
            
            K = out_output_layer.reshape(batch_size, self.state_dim, self.obs_dim)

            # Aktualizace stavu
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            x_hat_posterior = x_predicted + correction

            # Příprava pro další krok
            delta_x_hat_update = x_hat_posterior - x_predicted
            delta_x_hat_update_prev = delta_x_hat_update.clone()
            x_filtered_prev = x_hat_posterior.clone()
            
            x_hat_list.append(x_hat_posterior)

            if (self.system_model.H @ self.system_model.H.T).det() != 0:
                P_predict = K @ self.system_model.R @ (torch.eye(self.state_dim) - K @ self.system_model.H).inverse() @ self.system_model.H.T @ self.system_model.H
                P_predict_list.append(P_predict)
            

        return torch.stack(x_hat_list, dim=1)