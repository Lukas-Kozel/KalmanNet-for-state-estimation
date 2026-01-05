import torch
import torch.nn as nn
import torch.nn.init as init
from .DNN_RNN import DNN_RNN 

class RNN(nn.Module):
    def __init__(self, system_model, device, 
                  n_gru_layers=3, in_mult=3,out_mult=2):
        super().__init__()
        
        self.returns_covariance = False
        self.system_model = system_model
        self.device = device
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        
        # Inicializace DNN
        self.dnn = DNN_RNN(
            system_model, 
            n_gru_layers=n_gru_layers,
            in_mult=in_mult,
            out_mult=out_mult
        ).to(device)
        
        # Buffery pro stav
        self.x_prev = None      
        self.hn = None          
        
        # Buffery pro "Sample-and-Hold" strategii (paměť senzorů)
        self.last_valid_y = None
        self.last_valid_u = None
        
        self.init_weights()

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is None:
            initial_state = torch.zeros(batch_size, self.state_dim, device=self.device)
        else:
            initial_state = initial_state.clone().to(self.device)

        self.x_prev = initial_state
        
        self.hn = torch.zeros(
            self.dnn.n_layers, 
            batch_size, 
            self.dnn.hidden_dim, 
            device=self.device
        )
        
        # Reset paměti senzorů (na nuly, protože na začátku nic lepšího nemáme)
        self.last_valid_y = torch.zeros(batch_size, self.obs_dim, device=self.device)

    def step(self, y_t):
        y_t = y_t.to(self.device)
        
        # --- ROBUSTNÍ IMU/GPS IMPUTATION (Sample-and-Hold) ---
        
        # 1. Zjistíme, kde jsou NaN
        y_isnan = torch.isnan(y_t)
        
        # 2. Nahradíme NaN hodnotami z minulého kroku
        # Použijeme torch.where: Kde je NaN, vezmi last_valid, jinak vezmi aktuální y_t
        y_t_imputed = torch.where(y_isnan, self.last_valid_y, y_t)
        
        # 3. Aktualizujeme last_valid jen tam, kde data BYLA platná
        # (Pokud je teď NaN, last_valid se nemění. Pokud je číslo, přepíšeme ho.)
        self.last_valid_y = torch.where(y_isnan, self.last_valid_y, y_t)
        # Pojistka pro případ, že hned první vzorek je NaN (zůstane 0.0, to je OK)
        self.last_valid_y = torch.nan_to_num(self.last_valid_y, nan=0.0) 

        
        # Forward pass s "čistými" daty
        delta_x, h_new = self.dnn(self.x_prev, y_t_imputed, self.hn)
        x_new = self.x_prev + delta_x
        # Update historie
        self.x_prev = x_new
        self.hn = h_new
        
        return x_new

    def _detach(self):
        self.x_prev = self.x_prev.detach()
        if self.hn is not None:
            self.hn = self.hn.detach()
        # Buffery last_valid není třeba detachovat, protože nevstupují do grafu gradientů přímo,
        # ale pro jistotu (aby se neuvolnily z paměti divně) to ničemu nevadí:
        self.last_valid_y = self.last_valid_y.detach()
        self.last_valid_u = self.last_valid_u.detach()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)