import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1) # Tvar [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # Tvar [d_model/2]
        
        pe = torch.zeros(1, max_len, d_model)
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DNN_KalmanFormer(nn.Module):
    
    def __init__(self, system_model, 
                 d_model=64, nhead=4, num_encoder_layers=1, 
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1):
        
        super(DNN_KalmanFormer, self).__init__()
        
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        
        self.encoder_input_dim = self.obs_dim * 2
        self.decoder_input_dim = self.state_dim * 2
        
        self.d_model = d_model

        self.encoder_input_norm = nn.LayerNorm(self.encoder_input_dim)
        self.decoder_input_norm = nn.LayerNorm(self.decoder_input_dim)
        
        self.encoder_input_layer = nn.Linear(self.encoder_input_dim, d_model)
        self.decoder_input_layer = nn.Linear(self.decoder_input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True 
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, self.state_dim * self.obs_dim)

    def forward(self, norm_obs_diff, norm_innovation, 
                norm_fw_evol_diff, norm_fw_update_diff):
        
        obs_features = torch.cat((norm_obs_diff, norm_innovation), dim=1)
        state_features = torch.cat((norm_fw_evol_diff, norm_fw_update_diff), dim=1)

        obs_features = self.encoder_input_norm(obs_features)
        state_features = self.decoder_input_norm(state_features)

        src = self.encoder_input_layer(obs_features)
        tgt = self.decoder_input_layer(state_features)

        src = src.unsqueeze(1) # [B, 1, d_model]
        tgt = tgt.unsqueeze(1) # [B, 1, d_model]

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        
        output = output.squeeze(1) # [B, d_model]


        K_vec = self.output_layer(output)
        
        return K_vec