# KalmanNet.py
import torch
import torch.nn as nn

class KalmanNet(nn.Module):
    def __init__(self, system_model, hidden_dim=128):
        super(KalmanNet, self).__init__()
        
        # Uložení parametrů a modelu
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.hidden_dim = hidden_dim

        # Dimenze spojeného vstupního vektoru [∆x̂_{t-1}, ∆y_t]
        input_dim = self.state_dim + self.obs_dim

        # --- Architektura 1 sítě ---
        # 1. Vstupní plně propojená vrstva (Fully connected linear input layer)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # 2. Rekurentní jádro (GRU)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        
        # 3. Výstupní plně propojená vrstva (Fully connected linear output layer)
        self.output_layer = nn.Linear(hidden_dim, self.state_dim * self.obs_dim)

    def forward(self, y_seq):
        """
        Provede filtraci celé sekvence měření y_seq.
        """
        device = self.input_layer.weight.device  # Získání zařízení, na kterém jsou váhy

        batch_size, seq_len, _ = y_seq.shape
        
        # --- Inicializace proměnných ---
        # Aktuální odhad stavu (pro t-1)
        x_hat = torch.zeros(batch_size, self.state_dim, device=device)
        # Předchozí odhad stavu (pro t-2)
        x_hat_previous = torch.zeros(batch_size, self.state_dim, device=device)

        # Skrytý stav GRU
        h = torch.zeros(1, batch_size, self.hidden_dim,device=device)

        # Seznam pro ukládání výsledných odhadů
        x_hat_list = []

        # Smyčka přes všechny časové kroky v sekvenci
        for t in range(seq_len):
            y_t = y_seq[:, t, :]  # Aktuální měření y_t
            
            # Přesun parametrů modelu na správné zařízení
            F = self.system_model.F.to(device)
            H = self.system_model.H.to(device)

            #### Predikce #### 
            x_hat_priori = (F @ x_hat.unsqueeze(-1)).squeeze(-1)
            y_hat = (H @ x_hat_priori.unsqueeze(-1)).squeeze(-1)
            
            #### Výpočet vstupů pro síť #### 
            # Inovace měření: ∆y_t
            innovation = y_t - y_hat
            # Změna odhadu stavu: ∆x̂_{t-1}
            delta_x_hat = x_hat - x_hat_previous

            # Spojení vstupů do jednoho vektoru
            nn_input = torch.cat([delta_x_hat, innovation], dim=1)

            ##### Průchod sítí pro výpočet Kalmanova zisku K_t ##### 
            out_input_layer = self.input_layer(nn_input)

            # GRU očekává vstup (seq_len, batch_size, input_size)
            out_gru, h = self.gru(out_input_layer.unsqueeze(0), h)

            out_output_layer = self.output_layer(out_gru.squeeze(0))
            
            # Přetvarování výstupu na matici Kalmanova zisku
            K = out_output_layer.reshape(batch_size, self.state_dim, self.obs_dim)

            #### Aktualizace odhadu stavu ####
            correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
            
            # Aktualizace proměnných pro další krok
            x_hat_previous = x_hat.clone()  # Uložíme si starý odhad (t-1)
            x_hat = x_hat_priori + correction  # Vypočítáme nový odhad (t)
            
            x_hat_list.append(x_hat)

        # Spojení seznamu odhadů do jednoho tenzoru
        return torch.stack(x_hat_list, dim=1)