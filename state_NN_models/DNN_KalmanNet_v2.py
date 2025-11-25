import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNorm(nn.Module):
     def __init__(self, dim=-1):
         super().__init__()
         self.dim = dim
     def forward(self, x):
         x = x - x.mean(dim=self.dim, keepdims=True)
         n = torch.sqrt(x.pow(2).sum(dim=self.dim, keepdims=True) + 1e-10)
         x = x / n
         return x

class DNN_KalmanNet_v2(nn.Module):
     def __init__(self, system_model, hidden_size_multiplier=10,
output_layer_multiplier=4, num_gru_layers=1, gru_hidden_dim_multiplier=1):
         super(DNN_KalmanNet_v2, self).__init__()

         self.state_dim = system_model.state_dim
         self.obs_dim = system_model.obs_dim
         self.device = system_model.device

         m = self.state_dim
         n = self.obs_dim
         # 4 vstupy: state_inno, inovation, diff_state, diff_obs
         # (Rozměry: m, n, m, n)
         self.input_dim = 2 * m + 2 * n

         # 1. skrytá vrstva (vstupní)
         # H1_KNet = (m + n) * 10 * 8
         self.H1 = (m + n) * hidden_size_multiplier * 8

         # Dimenze skrytého stavu GRU
         # hidden_dim = (m^2 + n^2) * 10
         self.gru_hidden_dim = (m*m + n*n) * gru_hidden_dim_multiplier

         # 2. skrytá vrstva (výstupní)
         # H2_KNet = (m * n) * 1 * 4
         self.H2 = (m * n) * output_layer_multiplier

         # Výstupní vrstva (vektorizovaný zisk)
         self.output_dim = m * n

         self.input_layer = nn.Sequential(
             nn.Linear(self.input_dim, self.H1),
             CustomNorm(),
             #nn.Dropout(0.1),
             nn.ReLU(),
         ).to(self.device)

         #self.input_repository = Repository(self.input_dim, self.H1)

         #self.scale = Scale(self.input_dim)

         self.gru = nn.GRU(
             self.H1,                   # Vstupní dimenze z předchozí vrstvy
             self.gru_hidden_dim,       # Dimenze skrytého stavu
             num_layers=num_gru_layers, # Počet vrstev
             dropout=0.0,
         ).to(self.device)

         #self.passGRU = nn.Linear(self.H1,
        # self.gru_hidden_dim).to(self.device)

         self.output_hidden_layer = nn.Sequential(
             #CustomNorm(),
             #nn.Linear(self.gru_hidden_dim, self.output_dim),
             nn.Linear(self.gru_hidden_dim, self.H2),
             #nn.Dropout(0.1),
             #CustomNorm(),
             #nn.LayerNorm(self.H2),
             nn.ReLU(),
         ).to(self.device)

         self.n_repository = 32

         self.output_final_linear = nn.Sequential(
             nn.Linear(self.H2, self.output_dim)
             #nn.Linear(self.H2, self.n_repository),
             #nn.Softmax(dim=-1),
             #nn.Linear(self.n_repository, self.output_dim),
         ).to(self.device)


     def forward(self, state_inno, inovation, diff_state, diff_obs, h_prev):
         """
         Provádí dopředný průchod sítí.

         Vstupy (všechny [batch_size, dim]):
         - state_inno: Normalizovaný predikovaný stav (např. x_{t|t-1})
         - inovation: Normalizovaná inovace (F2: y_t - h(x_{t|t-1}))
         - diff_state: Normalizovaný rozdíl stavu (F4: x_{t-1|t-1} -x_{t-1|t-2})
         - diff_obs: Normalizovaný rozdíl pozorování (F1: y_t - y_{t-1})
         - h_prev: Skrytý stav GRU z minulého kroku
         """

         #plt.plot(state_inno.detach().numpy(), label="state_inno")
         #plt.plot(inovation.detach().numpy(), label="inovation")
         #plt.plot(diff_state.detach().numpy(), label="diff_state")
         #plt.plot(diff_obs.detach().numpy(), label="diff_obs")
         #print(h_prev.detach().numpy())
         #plt.legend()
         #plt.show()
         #raise Exception("stop")

         nn_input = torch.cat([state_inno, inovation, diff_state,diff_obs], dim=1)

         if False:
             mx = 10.0
             nn_input = torch.clip(nn_input, min=-mx, max=mx)

         if False:
             nn_input = torch.sign(nn_input) * torch.log(nn_input.abs() + 1.0)

         #nn_input = self.input_repository(nn_input)

         #nn_input = self.scale(nn_input)

         activated_input = self.input_layer(nn_input)

         #print(activated_input)

         #activated_input = 1e-10 * activated_input

         #activated_input = activated_input.tanh()

         out_gru, h_new = self.gru(activated_input.unsqueeze(0), h_prev)
         out_gru_squeezed = out_gru.squeeze(0)
         h_new_squeezed = h_new.squeeze(0)

         #h_new = h_prev
         #out_gru_squeezed = self.passGRU(activated_input)

         #print(out_gru_squeezed)
         #print(h_new_squeezed)
         #raise Exception("debug")

         out_hidden = self.output_hidden_layer(out_gru_squeezed)
         #out_hidden = self.output_hidden_layer(h_new_squeezed)
         #out_hidden =self.output_hidden_layer(torch.cat([out_gru_squeezed, h_new_squeezed],dim=-1))

         K_vec_raw = self.output_final_linear(out_hidden)

         #K_vec_raw = torch.matmul(K_vec_raw, self.repository)

         #K_vec_raw = self.foo.tile(K_vec_raw.size(0), 1)

         #mx = 0.1
         #K_vec_raw = torch.clip(K_vec_raw, max=mx, min=-mx)

         return K_vec_raw, h_new
