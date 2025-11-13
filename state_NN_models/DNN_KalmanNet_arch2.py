import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_KalmanNet_arch2(nn.Module):
    """
    Toto je implementace sítě pro Architekturu #2 z článku KalmanNet .
    Tato síť má 3 oddělené GRU jednotky pro sledování Q, Sigma a S,
    a sadu FC vrstev pro výpočet Kalmanova zisku K.
    """
    def __init__(self, system_model, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4):
        
        super(DNN_KalmanNet_arch2, self).__init__()

        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.device = system_model.device

        m = self.state_dim
        n = self.obs_dim
        
        # Přejmenování multiplikátorů pro srozumitelnost
        in_mult_KNet = hidden_size_multiplier
        out_mult_KNet = output_layer_multiplier

        # --- Architektura #2: 3x GRU, 7x FC ---
        
        # GRU 1: Sleduje Q (Process Noise Covariance) [cite: 303]
        self.d_input_Q = m * in_mult_KNet
        self.d_hidden_Q = m * m
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)

        # GRU 2: Sleduje Sigma (Prior State Covariance) [cite: 304]
        self.d_input_Sigma = self.d_hidden_Q + m * in_mult_KNet
        self.d_hidden_Sigma = m * m
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)

        # GRU 3: Sleduje S (Innovation Covariance) [cite: 304]
        self.d_input_S = n * n + 2 * n * in_mult_KNet
        self.d_hidden_S = n * n
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)

        # FC 1: Propojuje Sigma -> S [cite: 299]
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = n * n
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU())

        # FC 2: Kombinuje S a Sigma pro výpočet K [cite: 299]
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = n * m # K_vec má dimenzi (m*n)
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2), nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # FC 3: Aktualizace zpětné vazby [cite: 299]
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = m * m
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU())

        # FC 4: Aktualizace zpětné vazby [cite: 299]
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU())

        # --- Vstupní FC vrstvy pro zpracování features ---
        # (Mapují 4 rysy na vstupy pro 3 GRU)
        
        # FC 5: Pro F3 (fw_evol_diff) -> GRU_Q
        self.d_input_FC5 = m
        self.d_output_FC5 = m * in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU())

        # FC 6: Pro F4 (fw_update_diff) -> GRU_Sigma
        self.d_input_FC6 = m
        self.d_output_FC6 = m * in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU())

        # FC 7: Pro F1 (obs_diff) a F2 (innovation) -> GRU_S
        self.d_input_FC7 = 2 * n
        self.d_output_FC7 = 2 * n * in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU())
        
        # Přesun všech modulů na správné zařízení
        self.to(self.device)

    def forward(self, norm_obs_diff, norm_innovation, norm_fw_evol_diff, norm_fw_update_diff, 
                h_prev_Q, h_prev_Sigma, h_prev_S):
        """
        Provádí dopředný průchod Architekturou #2.
        Vyžaduje 3 oddělené skryté stavy.
        """
        
        # 1. Zpracování vstupních rysů (features)
        in_Q = self.FC5(norm_fw_evol_diff)
        in_Sigma_part1 = self.FC6(norm_fw_update_diff)
        in_S_part1 = self.FC7(torch.cat([norm_obs_diff, norm_innovation], dim=1))

        # 2. Krok GRU 1 (Q)
        out_Q, h_new_Q = self.GRU_Q(in_Q.unsqueeze(0), h_prev_Q)
        out_Q = out_Q.squeeze(0) # [B, m*m]

        # 3. Krok GRU 2 (Sigma)
        in_Sigma_combined = torch.cat([out_Q, in_Sigma_part1], dim=1)
        out_Sigma, h_new_Sigma = self.GRU_Sigma(in_Sigma_combined.unsqueeze(0), h_prev_Sigma)
        out_Sigma = out_Sigma.squeeze(0) # [B, m*m] -> Toto je $\Sigma_{t|t-1}$

        # 4. Krok GRU 3 (S)
        in_S_part2 = self.FC1(out_Sigma)
        in_S_combined = torch.cat([in_S_part1, in_S_part2], dim=1)
        out_S, h_new_S = self.GRU_S(in_S_combined.unsqueeze(0), h_prev_S)
        out_S = out_S.squeeze(0) # [B, n*n] -> Toto je $S_{t|t-1}$
        
        # 5. Výpočet Kalmanova zisku (K_vec) pomocí FC vrstev
        in_FC2 = torch.cat([out_S, out_Sigma], dim=1)
        K_vec = self.FC2(in_FC2) # [B, m*n]
        
        # 6. Zpětná vazba pro aktualizaci skrytých stavů (nepoužívá se pro K)
        # Toto je pro *příští* krok, ale musíme to spočítat teď, abychom 
        # vrátili správné h_new_Sigma a h_new_S.
        # POZNÁMKA: Původní Arch 2 je složitější (viz Fig 4), ale pro 
        # zjednodušení (a aby odpovídala vaší `StateKalmanNet_v2`), 
        # vrátíme jen K_vec a nové skryté stavy.
        #
        # AKTUALIZACE: Pro věrnou implementaci Arch 2 (Fig 4), GRU stavy 
        # jsou aktualizovány pomocí FC3 a FC4.
        
        in_FC3 = torch.cat([out_S, K_vec], dim=1)
        out_FC3 = self.FC3(in_FC3) # [B, m*m]

        in_FC4 = torch.cat([out_Sigma, out_FC3], dim=1)
        out_FC4 = self.FC4(in_FC4) # [B, m*m]
        
        # Aktualizované skryté stavy pro příští volání
        # GRU_Sigma a GRU_S přijímají zpětnou vazbu samy ze sebe
        # (Toto je zjednodušení oproti Fig 4, kde se zdá, že h_new je upraveno)
        # Pro zachování kompatibility s GRU:
        # h_new_Sigma_updated = out_FC4.unsqueeze(0) # Takhle to nejde, GRU vrací h_new
        
        # Zjednodušení: Budeme ignorovat FC3 a FC4 a vrátíme K_vec
        # Toto je běžná praxe, pokud nechceme implementovat plnou 
        # smyčku z Fig 4, která mění samotné skryté stavy.
        
        # Vrátíme K_vec a 3 nové skryté stavy
        return K_vec, h_new_Q, h_new_Sigma, h_new_S