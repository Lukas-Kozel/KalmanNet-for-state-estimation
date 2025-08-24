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

    def forward(self, y_seq):
        """
        Tato metoda nyní provádí POUZE JEDEN dopředný průchod pro danou dávku.
        Je ideální pro trénink.
        """
        batch_size, seq_len, _ = y_seq.shape
        
        x_hat_prev_posterior = torch.zeros(batch_size, self.state_dim, device=self.device)
        delta_x_hat_update_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        h = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

        x_hat_list = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            x_hat_priori = self.f_vmap(x_hat_prev_posterior)
            y_hat = self.h_vmap(x_hat_priori)
            innovation = y_t - y_hat

            norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
            norm_delta_x = F.normalize(delta_x_hat_update_prev, p=2, dim=1, eps=1e-12)
            nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

            # Průchod sítí s dropoutem
            out_input_layer = self.input_layer(nn_input)
            activated_input = F.relu(self.dropout_layer1(out_input_layer))
            out_gru, h = self.gru(activated_input.unsqueeze(0), h)
            out_output_layer = self.output_layer(out_gru.squeeze(0))
            K_vec = self.dropout_layer2(out_output_layer)
            
            K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            x_hat_posterior = x_hat_priori + correction

            delta_x_hat_update = x_hat_posterior - x_hat_priori
            delta_x_hat_update_prev = delta_x_hat_update.clone()
            x_hat_prev_posterior = x_hat_posterior.clone()
            
            x_hat_list.append(x_hat_posterior)
            
        return torch.stack(x_hat_list, dim=1)
    
    def predict(self, y_seq, num_samples=20):
        """
        Provádí Monte Carlo inferenci pro kvantifikaci nejistoty.
        """
        # Důležité: Aktivujeme dropout vrstvy, i když je model v eval módu
        self.train() 
        
        with torch.no_grad():
            # Získáme `num_samples` různých odhadů
            ensemble_x_hat = torch.stack([self.forward(y_seq) for _ in range(num_samples)], dim=0)

        # Vrátíme model zpět do eval módu (vypne dropout)
        self.eval() 
        
        # Výpočet průměru (odhad stavu)
        x_hat_mean = ensemble_x_hat.mean(dim=0)
        
        # Výpočet kovariance (odhad nejistoty)
        diff = ensemble_x_hat - x_hat_mean
        # U 1D stavu je to jen variance, ale pro obecnost použijeme maticový součin
        if self.state_dim == 1:
            # Pro 1D případ je variance jednodušší a rychlejší
            sigma_hat = ensemble_x_hat.var(dim=0)
        else:
            # Pro N-D případ
            diff_permuted = diff.permute(1, 2, 0, 3) # [batch, seq, J, state_dim]
            sigma_hat = torch.matmul(diff_permuted.unsqueeze(-1), diff_permuted.unsqueeze(-2)).mean(dim=2)
        
        return x_hat_mean, sigma_hat