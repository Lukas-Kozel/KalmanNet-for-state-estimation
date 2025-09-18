import torch
import torch.nn as nn   
from ConcreteDropout import ConcreteDropout


class DNN_KalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim:int=2, y_dim:int=2, ngru=1, gru_scale_k:int=4):

        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        H1 = (x_dim + y_dim) * (10) * 8
        H2 = (x_dim * y_dim) * 1 * (4)

        self.input_dim = (self.x_dim * 2) + (self.y_dim * 2)
        self.output_dim = self.x_dim * self.y_dim

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim, H1),
            nn.ReLU()
        )

        self.l1_conc = ConcreteDropout()

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = gru_scale_k * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim))
        self.gru_n_layer = ngru
        self.batch_size = 1
        self.seq_len_input = 1

        self.hn = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn_init = self.hn.detach().clone()
        self.GRU = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> kalman gain
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim)
        )
        self.l2_conc = ConcreteDropout()
        

    def initialize_hidden(self):
        self.hn = self.hn_init.detach().clone()

    def forward(self, state_inno, observation_inno, diff_state, diff_obs):
        regularization = torch.empty(2)

        input = torch.cat((state_inno, observation_inno, diff_state, diff_obs), axis=0).reshape(-1)


        l1_out, regularization[0] = self.l1_conc(input, self.l1)

        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l1_out
        GRU_out, self.hn = self.GRU(GRU_in, self.hn)


        l2_out, regularization[1] = self.l2_conc(GRU_out, self.l2)

        kalman_gain = torch.reshape(l2_out, (self.x_dim, self.y_dim))

        return kalman_gain, regularization

