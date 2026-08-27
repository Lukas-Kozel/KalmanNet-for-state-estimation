import torch
import torch.nn as nn
import torch.nn.init as init

class DNN_RNN(nn.Module):
    def __init__(self, system_model, n_gru_layers=3, in_mult=3,out_mult=2):
        super().__init__()
        
        self.sys_state_dim = system_model.state_dim  # m
        self.sys_obs_dim = system_model.obs_dim      # n

        # Network input: [Previous_State, Measurement, Control]
        # To be fair to EKF/KNet, the RNN must also receive 'u' (IMU/Odo),
        # otherwise it would have no chance to estimate motion without GPS.
        self.raw_input_dim = self.sys_state_dim + self.sys_obs_dim
        
        # B) Expanded Input (Output of first FC layer -> GRU Input)
        # Author: (m + n) * in_mult
        self.gru_input_dim = self.raw_input_dim * in_mult
        
        # C) Hidden Dimension (GRU internal state)
        # Author: (n*n + m*m) * out_mult
        # Note: Add q*q as well to give the network capacity to process input
        self.hidden_dim = (
            (self.sys_state_dim**2) + 
            (self.sys_obs_dim**2)
        ) * out_mult
        
        self.n_layers = n_gru_layers
        
        # --- 2. Architecture ---
        
        # FC1: Expands input (Raw -> Expanded)
        # Author: Linear(m+n, (m+n)*in_mult) -> ReLU
        self.input_layer = nn.Sequential(
            nn.Linear(self.raw_input_dim, self.gru_input_dim),
            nn.ReLU()
        )
        
        # GRU: Processes expanded input
        # Author: Input=(m+n)*in_mult, Hidden=Computation_with_powers
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True
        )
        
        # FC2: Output head (Hidden -> State)
        # Author: Linear(Hidden, m) -> ReLU (BUT NOTE!)
        # Note: Here we must use a "Deep Head" without ReLU at the end, as discussed previously.
        # Authors used ReLU because they probably predicted Gain (always positive) or normalized data.
        # We predict Delta X (can be negative).
        
        # Keep the "down-projection" structure but without final non-linearity.
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.sys_state_dim)
        )

    def forward(self, x_prev, y_t, h_prev):
        """
        x_prev: [Batch, state_dim]
        y_t:    [Batch, obs_dim]
        h_prev: [n_layers, Batch, hidden_dim]
        """
        # Concatenate inputs
        dnn_input = torch.cat((x_prev, y_t), dim=1) # [B, m+n]
        
        # Input Layer
        feat = self.input_layer(dnn_input) # [B, Hidden]
        
        # GRU requires [Batch, Seq, Feature]
        feat = feat.unsqueeze(1) # [B, 1, Hidden]
        
        # GRU Step
        gru_out, h_new = self.gru(feat, h_prev)
        
        # Output Layer
        gru_out = gru_out.squeeze(1) # [B, Hidden]
        x_new = self.output_layer(gru_out) # [B, m]
        
        return x_new, h_new
