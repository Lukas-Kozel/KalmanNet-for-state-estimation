import torch
import torch.nn as nn
import torch.nn.functional as F # Správný import pro relu

class KalmanNet(nn.Module):
    """
    KalmanNet Architektura #1 pro nelineární systémy.
    Tato verze je plně "device-aware" a přijímá zařízení jako
    parametr v konstruktoru.
    """
    def __init__(self, system_model, device, hidden_size_multiplier=10):
        super(KalmanNet, self).__init__()
        
        # Uložíme si zařízení jako atribut třídy
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
        self.f_vmap = torch.vmap(self.system_model.f, in_dims=(0,))
        self.h_vmap = torch.vmap(self.system_model.h, in_dims=(0,))


    def forward(self, y_seq):
        batch_size, seq_len, _ = y_seq.shape
        
        # Vytváříme tenzory na správném zařízení
        x_hat_prev_posterior = torch.zeros(batch_size, self.state_dim, device=self.device)
        delta_x_hat_update_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        h = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

        x_hat_list = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            # Predikce
            x_hat_priori = self.f_vmap(x_hat_prev_posterior)
            y_hat = self.h_vmap(x_hat_priori)
            
            # Výpočet vstupů
            innovation = y_t - y_hat


            norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
            norm_delta_x = F.normalize(delta_x_hat_update_prev, p=2, dim=1, eps=1e-12)

            nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

            # Průchod sítí
            out_input_layer = self.input_layer(nn_input)
            activated_input = F.relu(out_input_layer) # Používáme F, ne nn_func
            out_gru, h = self.gru(activated_input.unsqueeze(0), h)
            out_output_layer = self.output_layer(out_gru.squeeze(0))
            
            K = out_output_layer.reshape(batch_size, self.state_dim, self.obs_dim)

            # Aktualizace stavu
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            x_hat_posterior = x_hat_priori + correction

            # Příprava pro další krok
            delta_x_hat_update = x_hat_posterior - x_hat_priori
            delta_x_hat_update_prev = delta_x_hat_update.clone()
            x_hat_prev_posterior = x_hat_posterior.clone()
            
            x_hat_list.append(x_hat_posterior)

        return torch.stack(x_hat_list, dim=1)