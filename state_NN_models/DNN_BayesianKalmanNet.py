import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ConcreteDropout import ConcreteDropout

class DNN_BayesianKalmanNet(nn.Module):
    def __init__(self, system_model, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(DNN_BayesianKalmanNet, self).__init__()
        
        
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.device = system_model.device

        self.H1 = (self.state_dim + self.obs_dim) * hidden_size_multiplier
        self.H2 = (self.state_dim * self.obs_dim) * output_layer_multiplier
        self.input_dim = self.state_dim + self.obs_dim
        self.output_dim = self.state_dim * self.obs_dim

        # Definice vrstev
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.H1),
            nn.ReLU()
        )
        self.concrete_dropout1 = ConcreteDropout(device=self.device, init_min=init_min_dropout, init_max=init_max_dropout)
        self.gru = nn.GRU(self.H1, self.H1, num_layers=num_gru_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(self.H1, self.H2),
            nn.ReLU(),
            nn.Linear(self.H2, self.output_dim)
        )
        self.concrete_dropout2 = ConcreteDropout(device=self.device, init_min=init_min_dropout, init_max=init_max_dropout)

    def forward(self, nn_input, h_prev):
        
        regularization = []
        activated_input, regularization[0] = self.concrete_dropout1(nn_input, self.input_layer)
                
        out_gru,h_new = self.gru(activated_input.unsqueeze(0), h_prev)

        out_gru_squeezed = out_gru.squeeze(0)

        out_final, regularization[1] = self.concrete_dropout2(out_gru_squeezed, self.output_layer)

        return out_final, h_new, regularization