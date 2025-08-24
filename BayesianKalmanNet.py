import torch
import torch.nn as nn
import torch.nn.functional as F # Správný import pro relu

class BayesianKalmanNet(nn.Module):
    """
    BayesianKalmanNet Architektura #1 pro nelineární systémy.
    Tato verze je plně "device-aware" a přijímá zařízení jako
    parametr v konstruktoru.
    """
    def __init__(self, system_model, device, hidden_size_multiplier=10, dropout_prob=0.2):
        super(BayesianKalmanNet, self).__init__()
        
        # Uložíme si zařízení jako atribut třídy
        self.device = device
        
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dropout_prob = dropout_prob

        # Heuristika pro velikost skrytého stavu
        moments_dim = self.state_dim*self.state_dim + self.obs_dim*self.obs_dim
        self.hidden_dim = moments_dim * hidden_size_multiplier

        # Vstupní dimenze
        input_dim = self.state_dim + self.obs_dim

        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.dropout_layer1 = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.state_dim * self.obs_dim)

        self.dropout_layer2 = nn.Dropout(self.dropout_prob)

        # vmap pro nelineární funkce
        self.f_vmap = torch.vmap(self.system_model.f, in_dims=(0,))
        self.h_vmap = torch.vmap(self.system_model.h, in_dims=(0,))


    def forward(self, y_seq, num_samples=1):
        if num_samples > 1:
            self.train()
        batch_size, seq_len, _ = y_seq.shape

        # Původní tvar: (B, T, D_y) -> Nový tvar: (J, B, T, D_y)
        y_seq_expanded = y_seq.unsqueeze(0).expand(num_samples, -1, -1, -1)

        # (J, B, T, D_y) -> (J*B, T, D_y)    
        # Vytváříme tenzory na správném zařízení
        flat_batch_size = num_samples * batch_size
        
        y_seq_flat = y_seq_expanded.reshape(flat_batch_size, seq_len, self.obs_dim)

        x_hat_prev_posterior = torch.zeros(flat_batch_size, self.state_dim, device=self.device)
        delta_x_hat_update_prev = torch.zeros(flat_batch_size, self.state_dim, device=self.device)
        h = torch.zeros(1, flat_batch_size, self.hidden_dim, device=self.device)

        x_hat_list = []

        for t in range(seq_len):
            y_t = y_seq_flat[:, t, :]

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
            out_input_dropout_layer = self.dropout_layer1(out_input_layer)
            activated_input = F.relu(out_input_dropout_layer)

            out_gru, h = self.gru(activated_input.unsqueeze(0), h)
            
            
            out_output_layer = self.output_layer(out_gru.squeeze(0))
            out_output_dropout_layer = self.dropout_layer2(out_output_layer)
            K = out_output_dropout_layer.reshape(flat_batch_size, self.state_dim, self.obs_dim)

            # Aktualizace stavu
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            x_hat_posterior = x_hat_priori + correction

            # Příprava pro další krok
            delta_x_hat_update = x_hat_posterior - x_hat_priori
            delta_x_hat_update_prev = delta_x_hat_update.clone()
            x_hat_prev_posterior = x_hat_posterior.clone()
            
            x_hat_list.append(x_hat_posterior)
        all_x_hat_flat = torch.stack(x_hat_list, dim=1)

        return all_x_hat_flat.view(num_samples, batch_size, seq_len, self.state_dim)
    # Uvnitř třídy BayesianKalmanNet

    def predict_with_uncertainty(self, y_seq, num_samples=20):
        """
        Provádí odhad stavu a jeho nejistoty pomocí Monte Carlo Dropoutu.

        Args:
            y_seq (torch.Tensor): Sekvence měření o tvaru (batch_size, seq_len, obs_dim).
            num_samples (int): Počet Monte Carlo vzorků (dopředných průchodů).

        Returns:
            torch.Tensor: Finální odhad stavu (průměr), tvar (batch, seq, dim).
            torch.Tensor: Odhad kovariance (výběrová kovariance), tvar (batch, seq, dim, dim).
        """
        self.eval() # model je v eval módu
        
        with torch.no_grad():
            # Získáme soubor (ensemble) J odhadů
            # Tvar: (num_samples, batch_size, seq_len, state_dim)
            ensemble_x_hat = self.forward(y_seq, num_samples=num_samples)

            # 1. Vypočítáme finální odhad stavu jako průměr přes vzorky (dim=0)
            x_hat_final = ensemble_x_hat.mean(dim=0)
            
            # 2. Vypočítáme finální odhad kovariance
            # Musíme nejprve odečíst průměr od každého vzorku
            # ensemble_x_hat má tvar (J, B, T, D)
            # x_hat_final má tvar (B, T, D)
            # potřebujeme, aby byly broadcast-kompatibilní
            diff = ensemble_x_hat - x_hat_final.unsqueeze(0) # (J, B, T, D)

            # Výpočet výběrové kovariance: (1/(J-1)) * Σ(diff * diff^T)
            # diff.unsqueeze(-1) -> (J, B, T, D, 1)
            # diff.unsqueeze(-2) -> (J, B, T, 1, D)
            # (J, B, T, D, 1) @ (J, B, T, 1, D) -> (J, B, T, D, D)
            cov_sum = (diff.unsqueeze(-1) @ diff.unsqueeze(-2)).sum(dim=0)
            
            covariance_final = cov_sum / (num_samples - 1)
            
        return x_hat_final, covariance_final