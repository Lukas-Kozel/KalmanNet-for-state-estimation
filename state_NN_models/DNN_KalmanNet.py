import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_KalmanNet(nn.Module):
    def __init__(self, system_model, hidden_size_multiplier=10):
        super(DNN_KalmanNet, self).__init__()
        
        
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        hidden_dim = (self.state_dim**2 + self.obs_dim**2) * hidden_size_multiplier #  heuristická volba velikosti skryté vrstvy
        input_dim = self.state_dim + self.obs_dim # vstup je dán jako vektor delta_x_prev a innovace
        output_dim = self.state_dim * self.obs_dim # Výstupní dimenze odpovídá velikosti Kalmanova zisku ale jako vektor o jednom řádku

        # Definice vrstev
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, nn_input, h_prev):
        # nn_input: Vstupní tenzor zřetězených příznaků.
        #           Tvar: [batch_size, state_dim+obs_dim]
        
        # h_prev: Skrytý stav GRU z předchozího časového kroku.
        #         Tvar: [Num_Layers, batch_size, Hidden_Dim] (např. [1, 8, 120])
        #         `Num_Layers` je počet GRU vrstev nad sebou, zde je 1.
        

        # Vstupní vrstva + aktivace
        # input_layer(nn_input) provede maticové násobení:
        # [batch_size, state_dim + obs_dim] @ [state_dim + obs_dim, hidden_dim] -> [batch_size, hidden_dim]
        activated_input = F.relu(self.input_layer(nn_input))
        # `activated_input` Tvar: [batch_size, hidden_dim]

        # GRU očekává vstup ve formátu [Seq_Len, Batch_Size, Input_Dim].
        # Proto musíme `activated_input` rozšířit o "sekvenční" dimenzi délky 1.
        # .unsqueeze(0) přidá dimenzi na začátek:
        # [batch_size, hidden_dim] -> [1, batch_size, hidden_dim]

        # h_new: Finální skrytý stav po zpracování celé sekvence.
        #        Tvar: [Num_Layers, batch_size, Hidden_Dim] (např. [1, 8, 120])
        out_gru,h_new = self.gru(activated_input.unsqueeze(0), h_prev)

        # Výstupní vrstva
        # `out_gru` má zbytečnou sekvenční dimenzi. Musíme ji odstranit,
        # aby odpovídala vstupu pro `output_layer`.
        # .squeeze(0) odstraní dimenzi 0:
        # [1, batch_size, hidden_dim] -> [batch_size, hidden_dim]
        squeezed_gru_out = out_gru.squeeze(0)
        
        # output_layer provede finální transformaci:
        # [batch_size, hidden_dim] @ [hidden_dim, state_dim * obs_dim] -> [batch_size, state_dim * obs_dim]
        out_output_layer = self.output_layer(squeezed_gru_out)
        # `out_output_layer` Tvar: [batch_size, state_dim * obs_dim] (např. [8, 4])
        # Toto je finální vektorizovaný Kalmanův zisk pro každý prvek v dávce.

        # `h_new` Tvar: [Num_Layers, batch_size, Hidden_Dim] (např. [1, 8, 120])
        # Tento tenzor se "schová" a použije se jako `h_prev` v příštím volání.
        return out_output_layer, h_new