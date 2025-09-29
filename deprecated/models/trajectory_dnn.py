import torch
import torch.nn as nn
import torch.nn.functional as F
class trajectory_DNN_BayesianKalmanNet(nn.Module):
    def __init__(self, system_model, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(trajectory_DNN_BayesianKalmanNet, self).__init__()


        super().__init__()
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        H1 = (self. state_dim + self. obs_dim) * (hidden_size_multiplier) * 8
        H2 = (self. state_dim * self. obs_dim) * 1 * (output_layer_multiplier)

        self.input_dim = (self.self. state_dim * 2) + (self.self. obs_dim * 2)
        self.output_dim = self.self. state_dim * self.self. obs_dim

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim, H1),
            nn.ReLU()
        )
        self.input_norm = nn.LayerNorm(self.input_dim)

        self.l1_conc = ConcreteDropout(init_max=init_max_dropout,init_min=init_min_dropout)

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = 4 * ((self.self. state_dim * self.self. state_dim) + (self.self. obs_dim * self.self. obs_dim))
        self.gru_n_layer = num_gru_layers
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
        self.l2_conc = ConcreteDropout(init_max=init_max_dropout,init_min=init_min_dropout)
        

    def initialize_hidden(self):
        self.hn = self.hn_init.detach().clone()

    def forward(self, state_inno, observation_inno, diff_state, diff_obs):
        regularization = torch.empty(int(2))

        input = torch.cat((state_inno, observation_inno, diff_state, diff_obs), axis=0).reshape(-1)
        normalized_input = self.input_norm(input)

        l1_out, regularization[0] = self.l1_conc(normalized_input, self.l1)


        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l1_out
        GRU_out, self.hn = self.GRU(GRU_in, self.hn)

        l2_out, regularization[1] = self.l2_conc(GRU_out, self.l2)
        

        kalman_gain = torch.reshape(l2_out, (self.self. state_dim, self.self. obs_dim))

        return kalman_gain, regularization
        

from torch import nn
import torch

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-6, init_min=0.5, init_max=0.8,device=None):
        """
        weight_regularizer: váha regularizace vah - ve článku je to l^2
        dropout_regularizer: váha regularizace dropout pravděpodobnosti - ve článku je to K
        """
        super(ConcreteDropout, self).__init__()
        # self.device = device
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = torch.log(torch.tensor(init_min)) - torch.log(1. - torch.tensor(init_min))
        init_max = torch.log(torch.tensor(init_max)) - torch.log(1. - torch.tensor(init_max))

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max)) # dropout prob p je trenovatelny parametr,
        #uniform zajisti, ze je vzdy mezi 0 a 1
        # "logit" je transformace pravdepodobnosti p na realnou osu, tedy misto toho, aby se neuronka ucila, ze p musi byt mezi 0 a 1
        # , ucime se p_logit, ktery muze byt libovolne realne cislo

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit.to(x.device)) # p je mezi 0 a 1

        x_dropped = self._concrete_dropout(x, p)
        out = layer(x_dropped)

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_squares / (1. - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dim = x[0].numel()
        dropout_regularizer *= self.dropout_regularizer * input_dim

        regularization = weights_regularizer + dropout_regularizer

        return out, regularization

    def _concrete_dropout(self, x, p):
        """
        Applies Concrete Dropout to the input tensor x.
        """
        eps = 1e-7
        t = 0.1

        unif_noise = torch.rand_like(x) # ve clanku je to parametr u - U(0,1)

        drop_prob = (torch.log(p+eps) - torch.log(1. - p+eps) + torch.log(unif_noise+eps) - torch.log(1. - unif_noise+eps))
        drop_prob = torch.sigmoid(drop_prob / t) # pravdepodobnost vypnuti neuronu

        random_tensor = 1. - drop_prob
        retain_prob = 1. - p # pravdepodobnost, ze neuron zustane aktivni

        x= torch.mul(x, random_tensor) # vektor, ktery reprezentuje dropout masku, zpusobi ztlumeni neuronu, misto jejich uplneho vypnuti
        x = x / retain_prob # aplikace inverzniho skalovani

        return x
