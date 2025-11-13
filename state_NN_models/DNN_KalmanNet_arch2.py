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

        in_mult_KNet = hidden_size_multiplier
        out_mult_KNet = output_layer_multiplier


        self.d_input_Q = m * in_mult_KNet
        self.d_hidden_Q = m * m
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)

        self.d_input_Sigma = self.d_hidden_Q + m * in_mult_KNet
        self.d_hidden_Sigma = m * m
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)

        self.d_input_S = n * n + 2 * n * in_mult_KNet
        self.d_hidden_S = n * n
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)


        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = n * n
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU())

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = n * m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2), nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = m * m
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU())


        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU())

        self.d_input_FC5 = m
        self.d_output_FC5 = m * in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU())

        self.d_input_FC6 = m
        self.d_output_FC6 = m * in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU())


        self.d_input_FC7 = 2 * n
        self.d_output_FC7 = 2 * n * in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU())
        

        self.to(self.device)

    def forward(self, norm_obs_diff, norm_innovation, norm_fw_evol_diff, norm_fw_update_diff, 
                h_prev_Q, h_prev_Sigma, h_prev_S):
        """
        Provádí dopředný průchod Architekturou #2.
        Vyžaduje 3 oddělené skryté stavy.
        """
        
        in_Q = self.FC5(norm_fw_evol_diff)
        in_Sigma_part1 = self.FC6(norm_fw_update_diff)
        in_S_part1 = self.FC7(torch.cat([norm_obs_diff, norm_innovation], dim=1))


        out_Q, h_new_Q = self.GRU_Q(in_Q.unsqueeze(0), h_prev_Q)
        out_Q = out_Q.squeeze(0) # [B, m*m]


        in_Sigma_combined = torch.cat([out_Q, in_Sigma_part1], dim=1)
        out_Sigma, h_new_Sigma = self.GRU_Sigma(in_Sigma_combined.unsqueeze(0), h_prev_Sigma)
        out_Sigma = out_Sigma.squeeze(0)


        in_S_part2 = self.FC1(out_Sigma)
        in_S_combined = torch.cat([in_S_part1, in_S_part2], dim=1)
        out_S, h_new_S = self.GRU_S(in_S_combined.unsqueeze(0), h_prev_S)
        out_S = out_S.squeeze(0) 
        
        in_FC2 = torch.cat([out_S, out_Sigma], dim=1)
        K_vec = self.FC2(in_FC2) # [B, m*n]
        
        ## TODO: tyhle 2 vrstvy nejsou použity -> analyzovat
        in_FC3 = torch.cat([out_S, K_vec], dim=1)
        out_FC3 = self.FC3(in_FC3) # [B, m*m]

        in_FC4 = torch.cat([out_Sigma, out_FC3], dim=1)
        out_FC4 = self.FC4(in_FC4) # [B, m*m]
        
        return K_vec, h_new_Q, h_new_Sigma, h_new_S