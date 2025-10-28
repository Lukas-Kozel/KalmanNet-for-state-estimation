import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_KalmanNet_v2_3D_tan(nn.Module):
    """
    Rozšířená verze DNN pro KalmanNet, která se řídí principy 
    a dimenzemi z referenčního kódu (KalmanNetNN), ale 
    používá rozšířenou sadu 4 vstupních příznaků.
    """
    def __init__(self, system_model, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1):
        super(DNN_KalmanNet_v2_3D_tan, self).__init__()

        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.device = system_model.device
        
        m = self.state_dim
        n = self.obs_dim

        # ---- Dimenze podle tvého cíle (4 vstupy) ----
        # 4 vstupy: state_inno, inovation, diff_state, diff_obs
        # (Rozměry: m, n, m, n)
        self.input_dim = 2 * m + 2 * n

        # ---- Dimenze vrstev podle referenčního kódu KalmanNetNN ----
        
        # 1. skrytá vrstva (vstupní)
        # H1_KNet = (m + n) * 10 * 8
        self.H1 = (m + n) * hidden_size_multiplier * 8

        # Dimenze skrytého stavu GRU
        # hidden_dim = (m^2 + n^2) * 10
        self.gru_hidden_dim = (m*m + n*n) * 10
        
        # 2. skrytá vrstva (výstupní)
        # H2_KNet = (m * n) * 1 * 4
        self.H2 = (m * n) * output_layer_multiplier
        
        # Výstupní vrstva (vektorizovaný zisk)
        self.output_dim = m * n
        # ---------------------------------------------------------

        # Tvé vylepšení: LayerNorm pro stabilizaci vstupů
        self.input_norm = nn.LayerNorm(self.input_dim).to(self.device)

        # Vstupní vrstva (Linear + ReLU)
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.H1),
            nn.ReLU()
        ).to(self.device)

        # GRU vrstva
        self.gru = nn.GRU(
            self.H1,                  # Vstupní dimenze z předchozí vrstvy
            self.gru_hidden_dim,      # Dimenze skrytého stavu (podle reference)
            num_layers=num_gru_layers # Počet vrstev (tvůj hyperparametr)
        ).to(self.device)

        # Výstupní část (Linear + ReLU + Linear)
        self.output_layer = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, self.H2),
            nn.ReLU(),
            nn.Linear(self.H2, self.output_dim)
        ).to(self.device)

    def forward(self, state_inno, inovation, diff_state, diff_obs, h_prev):
        """
        Provádí dopředný průchod sítí.
        
        Vstupy (všechny [batch_size, dim]):
        - state_inno: Normalizovaný predikovaný stav (např. x_{t|t-1})
        - inovation: Normalizovaná inovace (F2: y_t - h(x_{t|t-1}))
        - diff_state: Normalizovaný rozdíl stavu (F4: x_{t-1|t-1} - x_{t-1|t-2})
        - diff_obs: Normalizovaný rozdíl pozorování (F1: y_t - y_{t-1})
        - h_prev: Skrytý stav GRU z minulého kroku
        """

        # Zřetězení všech 4 vstupních příznaků
        nn_input = torch.cat([state_inno, inovation, diff_state, diff_obs], dim=1)

        # Aplikace normalizace
        normalized_input = self.input_norm(nn_input)
        
        # Vstupní vrstva (už obsahuje ReLU)
        # OPRAVA: Odstraněno dvojité F.relu()
        activated_input = self.input_layer(normalized_input)

        # Průchod GRU
        # .unsqueeze(0) přidává dimenzi sekvence (Seq_Len=1)
        out_gru, h_new = self.gru(activated_input.unsqueeze(0), h_prev)
        
        # .squeeze(0) odstraňuje dimenzi sekvence
        out_gru_squeezed = out_gru.squeeze(0)

        # Průchod výstupními vrstvami
        out_output_layer = self.output_layer(out_gru_squeezed)
        
        return out_output_layer, h_new
