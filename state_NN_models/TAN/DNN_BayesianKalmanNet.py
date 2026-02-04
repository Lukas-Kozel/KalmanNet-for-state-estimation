import torch
import torch.nn as nn
import torch.nn.functional as F
class DNN_BayesianKalmanNetTAN(nn.Module):
    def __init__(self, system_model, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(DNN_BayesianKalmanNetTAN, self).__init__()

        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.device = system_model.device

        self.input_dim = 2 * self.state_dim + 2 * self.obs_dim
        self.output_dim = self.state_dim * self.obs_dim

        self.H1 = (self.state_dim + self.obs_dim) * hidden_size_multiplier * 8
        self.H2 = (self.state_dim * self.obs_dim) * output_layer_multiplier * 1

        # Přidání LayerNorm pro stabilizaci vstupů
        self.input_norm = nn.LayerNorm(self.input_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.H1),
            nn.ReLU()
        )
        self.concrete_dropout1 = ConcreteDropout(device=self.device, init_min=init_min_dropout, init_max=init_max_dropout)

        gru_hidden_dim = 4 * ((self.state_dim * self.state_dim) + (self.obs_dim * self.obs_dim))

        self.gru = nn.GRU(self.H1, gru_hidden_dim, num_layers=num_gru_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim, self.H2),
            nn.ReLU(),
            nn.Linear(self.H2, self.output_dim)
        )
        self.concrete_dropout2 = ConcreteDropout(device=self.device, init_min=init_min_dropout, init_max=init_max_dropout)

    def forward(self, state_inno, inovation, diff_state, diff_obs, h_prev):

        nn_input = torch.cat([state_inno, inovation, diff_state, diff_obs], dim=1)

        # Aplikace normalizace
        normalized_input = self.input_norm(nn_input)

        # Dropout se aplikuje na normalizovaná data
        activated_input, reg1 = self.concrete_dropout1(normalized_input, self.input_layer)

        out_gru, h_new = self.gru(activated_input.unsqueeze(0), h_prev)
        out_gru_squeezed = out_gru.squeeze(0)

        out_final, reg2 = self.concrete_dropout2(out_gru_squeezed, self.output_layer)

        total_reg = reg1+reg2
        return out_final, h_new, total_reg



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