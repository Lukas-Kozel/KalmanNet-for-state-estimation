import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Změna tvaru: [max_len, 1, d_model] -> [1, max_len, d_model]
        # aby odpovídal formátu (Batch, Seq, Feature)
        
        # --- ZDE BYLA OPRAVA ---
        position = torch.arange(max_len).unsqueeze(1) # Tvar [max_len, 1] (např. [5, 1])
        # Původně bylo: unsqueeze(0), což dávalo [1, 5]
        # --- KONEC OPRAVY ---
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # Tvar [d_model/2] (např. [32])
        
        pe = torch.zeros(1, max_len, d_model)
        
        # Broadcasting [5, 1] * [32] -> [5, 32]
        # pe[0, :, 0::2] má také tvar [5, 32]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Přičteme poziční kódování k sekvenční dimenzi (dim=1)
        # x.size(1) je nyní délka sekvence (což je 1)
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

        # --- PŘIDÁNO: Vrstvy LayerNorm (pro stabilitu) ---
        self.encoder_input_norm = nn.LayerNorm(self.encoder_input_dim)
        self.decoder_input_norm = nn.LayerNorm(self.decoder_input_dim)
        # --- KONEC PŘIDÁNÍ ---

        self.encoder_input_layer = nn.Linear(self.encoder_input_dim, d_model)
        self.decoder_input_layer = nn.Linear(self.decoder_input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # --- OPRAVA: batch_first=True (pro odstranění varování) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True # <--- ZMĚNA ZDE
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True # <--- ZMĚNA ZDE
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # --- KONEC OPRAVY ---
        
        self.output_layer = nn.Linear(d_model, self.state_dim * self.obs_dim)

    def forward(self, norm_obs_diff, norm_innovation, 
                norm_fw_evol_diff, norm_fw_update_diff):
        
        obs_features = torch.cat((norm_obs_diff, norm_innovation), dim=1)
        state_features = torch.cat((norm_fw_evol_diff, norm_fw_update_diff), dim=1)

        # --- PŘIDÁNO: Použití LayerNorm ---
        obs_features = self.encoder_input_norm(obs_features)
        state_features = self.decoder_input_norm(state_features)
        # --- KONEC PŘIDÁNÍ ---

        src = self.encoder_input_layer(obs_features)
        tgt = self.decoder_input_layer(state_features)

        # --- OPRAVA: unsqueeze(1) pro batch_first=True ---
        src = src.unsqueeze(1) # [B, 1, d_model]
        tgt = tgt.unsqueeze(1) # [B, 1, d_model]
        # --- KONEC OPRAVY ---

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        
        # --- OPRAVA: squeeze(1) pro batch_first=True ---
        output = output.squeeze(1) # [B, d_model]
        # --- KONEC OPRAVY ---

        K_vec = self.output_layer(output)
        
        return K_vec