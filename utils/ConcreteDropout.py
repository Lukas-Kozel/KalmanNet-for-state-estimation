from torch import nn
import torch

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.5, init_max=0.8,device='cpu'):
        """
        weight_regularizer: váha regularizace vah - ve článku je to l^2
        dropout_regularizer: váha regularizace dropout pravděpodobnosti - ve článku je to K
        """
        super(ConcreteDropout, self).__init__()
        self.device = device
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = torch.log(torch.tensor(init_min)) - torch.log(1. - torch.tensor(init_min))
        init_max = torch.log(torch.tensor(init_max)) - torch.log(1. - torch.tensor(init_max))

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max)).to(self.device) # dropout prob p je trenovatelny parametr,
        #uniform zajisti, ze je vzdy mezi 0 a 1
        # "logit" je transformace pravdepodobnosti p na realnou osu, tedy misto toho, aby se neuronka ucila, ze p musi byt mezi 0 a 1
        # , ucime se p_logit, ktery muze byt libovolne realne cislo

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit) # p je mezi 0 a 1

        out = layer(self._concrete_dropout(x, p))

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

        unif_noise = torch.rand_like(x).to(self.device) # ve clanku je to parametr u - U(0,1)

        drop_prob = (torch.log(p+eps) - torch.log(1. - p+eps) + torch.log(unif_noise+eps) - torch.log(1. - unif_noise+eps))
        drop_prob = torch.sigmoid(drop_prob / t) # pravdepodobnost vypnuti neuronu

        random_tensor = 1. - drop_prob
        retain_prob = 1. - p # pravdepodobnost, ze neuron zustane aktivni

        x= torch.mul(x, random_tensor) # vektor, ktery reprezentuje dropout masku, zpusobi ztlumeni neuronu, misto jejich uplneho vypnuti
        x = x / retain_prob # aplikace inverzniho skalovani

        return x
