import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_KalmanNet(nn.Module):
    def __init__(self, system_model, hidden_size_multiplier=10):
        super(DNN_KalmanNet, self).__init__()
        
        
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        hidden_dim = (self.state_dim**2 + self.obs_dim**2) * hidden_size_multiplier
        input_dim = self.state_dim + self.obs_dim
        output_dim = self.state_dim * self.obs_dim

        # Definice vrstev
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, nn_input, h_prev):
        
        activated_input = F.relu(self.input_layer(nn_input))
        out_gru,h_new = self.gru(activated_input.unsqueeze(0), h_prev)
        out_output_layer = self.output_layer(out_gru.squeeze(0))
        
        return out_output_layer, h_new