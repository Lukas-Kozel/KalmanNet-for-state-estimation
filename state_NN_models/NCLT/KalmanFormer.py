import torch.nn as nn
from .DNN_KalmanFormer import DNN_KalmanFormerNCLT
import torch.nn.functional as func
import torch.nn.init as init
import torch

class KalmanFormerNCLT(nn.Module):
    
    def __init__(self, system_model, device, 
                 d_model=64, nhead=4, num_encoder_layers=1, 
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1):
        
        super(KalmanFormerNCLT, self).__init__()
        
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_KalmanFormerNCLT(
            system_model,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)

        # Buffer pro historické hodnoty
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_filtered_prev = initial_state
        self.x_filtered_prev_prev = initial_state.clone()
        self.x_pred_prev = initial_state.clone()
        
        with torch.no_grad():
            self.y_prev = self.system_model.h(self.x_filtered_prev)

    def step(self, y_t_raw, u_t=None):
        """
        Robustní krok KalmanFormeru.
        Místo 'if' větvení používá maskování, aby se nepřetrhl graf gradientů.
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        # 1. PREDIKCE (Vždy stejná)
        x_pred_raw = self.system_model.f(self.x_filtered_prev, u_t) # [B, m]
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]
        
        # 2. MASKING NaN (Klíčová oprava)
        # Vytvoříme masku: True tam, kde JSOU data platná, False kde je NaN
        # Předpokládáme, že pokud je NaN v jakémkoliv kanálu, měření je neplatné
        is_nan_mask = torch.any(torch.isnan(y_t_raw), dim=1, keepdim=True) # [B, 1]
        valid_mask = ~is_nan_mask # [B, 1] (1 = valid, 0 = invalid)

        # Vytvoříme "vyčištěný" vstup y_t
        # Tam, kde je NaN, vložíme y_pred (inovace bude 0, síť se nezblázní)
        # Tím udržíme spojitost grafu.
        y_t_clean = torch.where(is_nan_mask, y_pred_raw, y_t_raw)
        
        # 3. VÝPOČET FEATURES (vždy s čistými daty)
        
        # F1: Observation difference
        # Pokud minule bylo NaN, y_prev jsme nastavili chytře (viz konec funkce)
        obs_diff = y_t_clean - self.y_prev
        
        # F2: Innovation
        # Pokud bylo NaN, y_t_clean == y_pred_raw, takže innovation je 0 (což je správně)
        innovation = y_t_clean - y_pred_raw
        
        # F3 & F4: State diffs
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        # Normalizace (L2)
        norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12) 
        
        # 4. DNN PASS (Běží vždy, i pro NaN data!)
        # Tím zajistíme, že 'K_vec' je vždy součástí výpočetního grafu
        K_vec = self.dnn(
            norm_obs_diff, norm_innovation, 
            norm_fw_evol_diff, norm_fw_update_diff
        )
        
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        # 5. KOREKCE S MASKOU
        # Spočítáme korekci (pro NaN data bude K * 0 = 0, ale raději to pojistíme)
        raw_correction = (K @ norm_innovation.unsqueeze(-1)).squeeze(-1)
        
        # Aplikujeme masku: 
        # Kde je valid_mask=1, projde korekce. 
        # Kde je valid_mask=0 (NaN), korekce se vynuluje.
        # Důležité: Tímto násobením zůstává graf spojený (vynásobení nulou je validní operace v autogradu).
        correction = raw_correction * valid_mask.float()

        x_filtered_raw = x_pred_raw + correction 

        # 6. UPDATE HISTORIE
        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        
        # Update y_prev
        # Pokud bylo měření platné, uložíme ho.
        # Pokud bylo NaN, uložíme 'y_pred_raw' (naši predikci), aby v dalším kroku
        # 'obs_diff' (y_t - y_prev) nedával nesmysly.
        self.y_prev = torch.where(is_nan_mask, y_pred_raw.detach(), y_t_raw.detach())
        
        return x_filtered_raw

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.TransformerEncoderLayer) or isinstance(m, nn.TransformerDecoderLayer):
                 for name, param in m.named_parameters():
                    if 'weight' in name:
                        if param.dim() > 1:
                            init.xavier_uniform_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def _detach(self):
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    def _log_and_raise(self, failed_tensor_name, local_vars):
        # (Logování může zůstat stejné, pokud ho tam chceš)
        pass