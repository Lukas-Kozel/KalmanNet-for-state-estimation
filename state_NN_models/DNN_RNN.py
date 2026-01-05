import torch
import torch.nn as nn
import torch.nn.init as init

class DNN_RNN(nn.Module):
    def __init__(self, system_model, n_gru_layers=3, in_mult=3,out_mult=2):
        super().__init__()
        
        self.sys_state_dim = system_model.state_dim  # m
        self.sys_obs_dim = system_model.obs_dim      # n

        # Vstup do sítě: [Minulý_Stav, Měření, Ovládání]
        # Aby to bylo fér vůči EKF/KNet, musíme RNN dát i 'u' (IMU/Odo), 
        # jinak by nemělo šanci odhadnout pohyb bez GPS.
        self.raw_input_dim = self.sys_state_dim + self.sys_obs_dim
        
        # B) Expanded Input (Výstup první FC vrstvy -> Vstup do GRU)
        # Autor: (m + n) * in_mult
        self.gru_input_dim = self.raw_input_dim * in_mult
        
        # C) Hidden Dimension (Vnitřní stav GRU)
        # Autor: (n*n + m*m) * out_mult
        # My: Přidáme i q*q, aby síť měla kapacitu i na zpracování vstupu
        self.hidden_dim = (
            (self.sys_state_dim**2) + 
            (self.sys_obs_dim**2)
        ) * out_mult
        
        self.n_layers = n_gru_layers
        
        # --- 2. Architektura ---
        
        # FC1: Roztáhne vstup (Raw -> Expanded)
        # Autor: Linear(m+n, (m+n)*in_mult) -> ReLU
        self.input_layer = nn.Sequential(
            nn.Linear(self.raw_input_dim, self.gru_input_dim),
            nn.ReLU()
        )
        
        # GRU: Zpracuje roztažený vstup
        # Autor: Input=(m+n)*in_mult, Hidden=Výpočet_s_mocninami
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True
        )
        
        # FC2: Výstupní hlava (Hidden -> State)
        # Autor: Linear(Hidden, m) -> ReLU (ALE POZOR!)
        # My: Zde musíme použít "Deep Head" bez ReLU na konci, jak jsme řešili minule.
        # Autoři měli ReLU, protože asi predikovali Gain (vždy kladný) nebo normalizovaná data.
        # My predikujeme Delta X (může být záporné).
        
        # Zachováme strukturu "projekce dolů", ale bez finální nelinearity.
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.sys_state_dim)
        )

    def forward(self, x_prev, y_t, h_prev):
        """
        x_prev: [Batch, state_dim]
        y_t:    [Batch, obs_dim]
        h_prev: [n_layers, Batch, hidden_dim]
        """
        # Konkatenace vstupů
        dnn_input = torch.cat((x_prev, y_t), dim=1) # [B, m+n]
        
        # Input Layer
        feat = self.input_layer(dnn_input) # [B, Hidden]
        
        # GRU vyžaduje [Batch, Seq, Feature]
        feat = feat.unsqueeze(1) # [B, 1, Hidden]
        
        # GRU Step
        gru_out, h_new = self.gru(feat, h_prev)
        
        # Output Layer
        gru_out = gru_out.squeeze(1) # [B, Hidden]
        x_new = self.output_layer(gru_out) # [B, m]
        
        return x_new, h_new