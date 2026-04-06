import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_KalmanNetTAN(nn.Module):
     def __init__(self, system_model, hidden_size_multiplier=10,
                  output_layer_multiplier=4, num_gru_layers=1, gru_hidden_dim_multiplier=1,
                  use_terrain_grad=True): # <--- NEW FLAG
         super(DNN_KalmanNetTAN, self).__init__()

         self.state_dim = system_model.state_dim
         self.obs_dim = system_model.obs_dim
         self.device = system_model.device
         self.use_terrain_grad = use_terrain_grad # <--- STORE FLAG

         m = self.state_dim
         n = self.obs_dim
         
         # Dynamically calculate input dimension based on the flag
         if self.use_terrain_grad:
             self.input_dim = 2 * m + 2 * n + 2 
             print("DNN initialized WITH terrain slope (+2 features)")
         else:
             self.input_dim = 2 * m + 2 * n
             print("DNN initialized WITHOUT terrain slope")

         # 1st hidden layer (input)
         self.H1 = (m + n) * hidden_size_multiplier * 8

         # GRU hidden state dimension
         self.gru_hidden_dim = (m*m + n*n) * gru_hidden_dim_multiplier

         # 2nd hidden layer (output)
         self.H2 = (m * n) * output_layer_multiplier

         # Output layer (vectorized gain)
         self.output_dim = m * n

         self.input_layer = nn.Sequential(
             nn.Linear(self.input_dim, self.H1),
             nn.ReLU(),
         ).to(self.device)

         self.gru = nn.GRU(
             self.H1,                   # Input dimension from previous layer
             self.gru_hidden_dim,       # Hidden state dimension
             num_layers=num_gru_layers, # Number of layers
             dropout=0.0,
         ).to(self.device)

         self.output_hidden_layer = nn.Sequential(
             nn.Linear(self.gru_hidden_dim, self.H2),
             nn.ReLU(),
         ).to(self.device)

         self.output_final_linear = nn.Sequential(
             nn.Linear(self.H2, self.output_dim)
         ).to(self.device)


     # CHANGED: Using *args to handle flexible input lengths conditionally
     def forward(self, state_inno, inovation, diff_state, diff_obs, *args):
         """
         Performs the forward pass of the network.

         Inputs (all [batch_size, dim]):
         - state_inno: Normalized predicted state (e.g., x_{t|t-1})
         - inovation: Normalized innovation (F2: y_t - h(x_{t|t-1}))
         - diff_state: Normalized state difference (F4: x_{t-1|t-1} - x_{t-1|t-2})
         - diff_obs: Normalized observation difference (F1: y_t - y_{t-1})
         
         *args unpacking:
         - If use_terrain_grad == True: args[0] = terrain_slope, args[1] = h_prev
         - If use_terrain_grad == False: args[0] = h_prev
         """

         # Conditionally unpack the remaining arguments and build the input tensor
         if self.use_terrain_grad:
             terrain_slope = args[0]
             h_prev = args[1]
             nn_input = torch.cat([state_inno, inovation, diff_state, diff_obs, terrain_slope], dim=1)
         else:
             h_prev = args[0]
             nn_input = torch.cat([state_inno, inovation, diff_state, diff_obs], dim=1)

         activated_input = self.input_layer(nn_input)

         out_gru, h_new = self.gru(activated_input.unsqueeze(0), h_prev)
         out_gru_squeezed = out_gru.squeeze(0)

         out_hidden = self.output_hidden_layer(out_gru_squeezed)

         K_vec_raw = self.output_final_linear(out_hidden)

         return K_vec_raw, h_new