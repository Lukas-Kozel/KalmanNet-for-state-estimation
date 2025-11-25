import torch
import torch.nn as nn

class DNN_KalmanNet_arch2(nn.Module):
    def __init__(self, system_model, 
                 hidden_size_multiplier=10, 
                 output_layer_multiplier=4):
        
        super(DNN_KalmanNet_arch2, self).__init__()

        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.device = system_model.device

        m = self.state_dim
        n = self.obs_dim

        in_mult_KNet = hidden_size_multiplier
        out_mult_KNet = output_layer_multiplier

        # --- GRU VRSTVY ---
        # 1. GRU pro Q (Process Noise)
        self.d_input_Q = m * in_mult_KNet
        self.d_hidden_Q = m * m
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)

        # 2. GRU pro Sigma (Covariance)
        self.d_input_Sigma = self.d_hidden_Q + m * in_mult_KNet
        self.d_hidden_Sigma = m * m
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)

        # 3. GRU pro S (Innovation Covariance)
        self.d_input_S = n * n + 2 * n * in_mult_KNet
        self.d_hidden_S = n * n
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)

        # FC1: Sigma -> S
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = n * n
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU())

        # FC2: Výpočet Kalmanova zisku K (z Sigma a S)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = n * m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2), nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # FC3: Backward Flow (S + K)
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = m * m
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU())

        # FC4: Backward Flow (Sigma + FC3 -> Posterior Sigma)
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU())

        # FC5: Feature Encoding (Evol Diff -> Q input)
        self.d_input_FC5 = m
        self.d_output_FC5 = m * in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU())

        # FC6: Feature Encoding (Update Diff -> Sigma input)
        self.d_input_FC6 = m
        self.d_output_FC6 = m * in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU())

        # FC7: Feature Encoding (Obs Diff + Innov -> S input)
        self.d_input_FC7 = 2 * n
        self.d_output_FC7 = 2 * n * in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU())
        
        self.to(self.device)
    
    def forward(self, obs_diff, innovation, fw_evol_diff, fw_update_diff, 
                h_prev_Q, h_prev_Sigma, h_prev_S):
        

        # 1. Q-GRU Větev
        in_Q = self.FC5(fw_evol_diff) # Zpracování F3
        out_Q, h_new_Q = self.GRU_Q(in_Q.unsqueeze(0), h_prev_Q)
        out_Q = out_Q.squeeze(0) # [B, m*m]

        # 2. Sigma-GRU Větev (Prior Covariance Estimation)
        in_FC6 = self.FC6(fw_update_diff) # Zpracování F4
        in_Sigma = torch.cat([out_Q, in_FC6], dim=1)
        
        # Standardní GRU krok, ALE pozor: h_new_Sigma z tohoto volání ignorujeme!
        # Autoři používají výstup z FC4 jako hidden state pro další krok.
        out_Sigma, _ = self.GRU_Sigma(in_Sigma.unsqueeze(0), h_prev_Sigma)
        out_Sigma = out_Sigma.squeeze(0) # [B, m*m]

        # 3. S-GRU Větev (Innovation Covariance Estimation)
        in_FC7 = self.FC7(torch.cat([obs_diff, innovation], dim=1)) # F1 + F2
        in_FC1 = self.FC1(out_Sigma)
        in_S = torch.cat([in_FC1, in_FC7], dim=1)
        
        out_S, h_new_S = self.GRU_S(in_S.unsqueeze(0), h_prev_S)
        out_S = out_S.squeeze(0) # [B, n*n]

        # 4. Výpočet Kalmanova Zisku (K)
        in_FC2 = torch.cat([out_Sigma, out_S], dim=1)
        K_vec = self.FC2(in_FC2) # [B, m*n]

        in_FC3 = torch.cat([out_S, K_vec], dim=1)
        out_FC3 = self.FC3(in_FC3)

        # FC4: Kombinace Sigma (Prior) a korekce -> Sigma (Posterior)
        in_FC4 = torch.cat([out_Sigma, out_FC3], dim=1)
        out_FC4 = self.FC4(in_FC4) # [B, m*m]
        # Pro Sigma bereme VÝSTUP z FC4 (Posterior Covariance) jako hidden state pro další krok.
        h_new_Sigma = out_FC4.unsqueeze(0) # [1, B, m*m]

        # K a nové stavy pro další krok TBPTT
        return K_vec, h_new_Q, h_new_Sigma, h_new_S


    def init_hidden_states(self, batch_size, prior_Q=None, prior_Sigma=None, prior_S=None):
        """
        Inicializuje skryté stavy GRU na základě fyzikálních priorů (Q, P0, R).
        Toto umožňuje síti začít s dobrou představou o neurčitostech.
        
        Args:
            batch_size (int): Velikost dávky.
            prior_Q (Tensor, volitelný): Matice šumu procesu Q [m, m].
            prior_Sigma (Tensor, volitelný): Počáteční kovariance P0 [m, m].
            prior_S (Tensor, volitelný): Matice šumu měření R [n, n] (jako odhad S).
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Inicializované hidden states pro Q, Sigma, S.
        """
        if prior_Q is None: 
            prior_Q = torch.eye(self.state_dim, device=self.device)
        if prior_Sigma is None: 
            prior_Sigma = torch.eye(self.state_dim, device=self.device)
        if prior_S is None: 
            prior_S = torch.eye(self.obs_dim, device=self.device)

        # 1. Q
        h_Q = prior_Q.flatten().reshape(1, 1, -1).repeat(1, batch_size, 1)
        
        # 2. Sigma
        h_Sigma = prior_Sigma.flatten().reshape(1, 1, -1).repeat(1, batch_size, 1)
        
        # 3. S
        h_S = prior_S.flatten().reshape(1, 1, -1).repeat(1, batch_size, 1)

        return h_Q, h_Sigma, h_S