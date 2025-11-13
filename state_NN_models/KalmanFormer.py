import torch.nn as nn
from .DNN_KalmanFormer import DNN_KalmanFormer
import torch.nn.functional as func
import torch.nn.init as init
import torch

class KalmanFormer(nn.Module):
    
    def __init__(self, system_model, device, 
                 d_model=64, nhead=4, num_encoder_layers=1, 
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1):
        
        super(KalmanFormer, self).__init__()
        
        self.returns_covariance = False 
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        # --- ZMĚNA ZDE ---
        self.dnn = DNN_KalmanFormer(
            system_model,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)
        # --- KONEC ZMĚNY ---

        # self.h_prev byl ODSTRANĚN
        self.y_prev = None                  
        self.x_filtered_prev = None     
        self.x_filtered_prev_prev = None  
        self.x_pred_prev = None         
        
        self.init_weights() # Tato metoda je stále užitečná pro lineární vrstvy

    def reset(self, batch_size=1, initial_state=None):
        """Inicializuje stavy pro novou sekvenci (dávku)."""
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_filtered_prev = initial_state
        self.x_filtered_prev_prev = initial_state.clone()
        self.x_pred_prev = initial_state.clone()
        
        with torch.no_grad():
            self.y_prev = self.system_model.h(self.x_filtered_prev)
        
        # self.h_prev byl ODSTRANĚN

    def step(self, y_t_raw):
        """
        Provede jeden kompletní krok KalmanFormer filtru (predikce + korekce) v čase 't'.
        """
        y_t_raw = y_t_raw.to(self.device) # [B, n]
        batch_size = y_t_raw.shape[0]

        x_pred_raw = self.system_model.f(self.x_filtered_prev) # [B, m]
        y_pred_raw = self.system_model.h(x_pred_raw) # [B, n]
        
        innovation_raw = y_t_raw - y_pred_raw
        innovation_safe = innovation_raw # (vaše stabilizace zde)
        
        # F1: Rozdíl pozorování
        obs_diff = y_t_raw - self.y_prev
        # F2: Inovace 
        innovation = innovation_safe
        # F3: Rozdíl posterior odhadů
        fw_evol_diff = self.x_filtered_prev - self.x_filtered_prev_prev
        # F4: Rozdíl (update)
        fw_update_diff = self.x_filtered_prev - self.x_pred_prev

        # Normalizace rysů (stejné jako u vás)
        # norm_obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        # norm_innovation = func.normalize(innovation, p=2, dim=1, eps=1e-12)
        # norm_fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        # norm_fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12) 
        norm_obs_diff = obs_diff
        norm_innovation = innovation
        norm_fw_evol_diff = fw_evol_diff
        norm_fw_update_diff = fw_update_diff
        # --- ZMĚNA ZDE ---
        # Volání DNN již nepracuje se skrytým stavem h_prev
        K_vec = self.dnn(
            norm_obs_diff,       # F1 (L2-norm)
            norm_innovation,     # F2 (L2-norm)
            norm_fw_evol_diff,   # F3 (L2-norm)
            norm_fw_update_diff  # F4 (L2-norm)
        )
        # --- KONEC ZMĚNY ---
        
        if not torch.all(torch.isfinite(K_vec)):
            self._log_and_raise("K_vec (výstup DNN)", locals())
        
        # --- KOREKCE (zůstává stejná) ---
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        
        correction = (K @ norm_innovation.unsqueeze(-1)).squeeze(-1)
        if not torch.all(torch.isfinite(correction)):
            self._log_and_raise("correction (K @ innov)", locals())

        x_filtered_raw = x_pred_raw + correction 
        if not torch.all(torch.isfinite(x_filtered_raw)):
            self._log_and_raise("x_filtered_raw (finální stav)", locals())

        # --- AKTUALIZACE STAVŮ (zůstává stejná, jen bez h_prev) ---
        self.x_filtered_prev_prev = self.x_filtered_prev.clone() 
        self.x_pred_prev = x_pred_raw.clone()         
        self.x_filtered_prev = x_filtered_raw.clone()      
        self.y_prev = y_t_raw.clone()                    
        # self.h_prev = h_new byl ODSTRANĚN
        
        return x_filtered_raw

    def init_weights(self) -> None:
        # Inicializace pro lineární vrstvy a Transformer vrstvy
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
        """Odpojí všechny stavy přenášené mezi TBPTT okny."""
        # self.h_prev byl ODSTRANĚN
        self.y_prev = self.y_prev.detach()
        self.x_filtered_prev = self.x_filtered_prev.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev_prev.detach()
        self.x_pred_prev = self.x_pred_prev.detach()

    # Metoda _log_and_raise zůstává stejná
    def _log_and_raise(self, failed_tensor_name, local_vars):
        # (Sem zkopírujte vaši stávající implementaci _log_and_raise)
        print(f"\n{'!'*40} SELHÁNÍ DETEKOVÁNO {'!'*40}")
        print(f"Příčina: Tensor '{failed_tensor_name}' obsahuje NaN nebo Inf.")
        # ... zbytek vaší logovací funkce ...
        raise RuntimeError(f"Selhání: Tensor '{failed_tensor_name}' je NaN nebo Inf!")