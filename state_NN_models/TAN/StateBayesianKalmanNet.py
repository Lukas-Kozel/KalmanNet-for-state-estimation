from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNetTAN
import torch
import torch.nn as nn
import torch.nn.init as init

class StateBayesianKalmanNetTAN(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(StateBayesianKalmanNetTAN, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_BayesianKalmanNetTAN(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        # Interní stavy
        self.x_filtered_t_minus_1 = None
        self.x_filtered_t_minus_2 = None
        self.x_pred_t_minus_1 = None
        self.y_t_minus_1 = None
        self.h_prev = None
        
        self.init_weights()


    def reset(self, batch_size=1, initial_state=None):
        """
        Inicializuje stavy pro novou sekvenci. 
        Sjednoceno s deterministickou verzí pro maximální stabilitu.
        """
        if initial_state is not None:
            # Detach je důležitý, abychom netahali gradienty z předchozího batche
            self.x_filtered_t_minus_1 = initial_state.detach().clone().to(self.device)
        else:
            self.x_filtered_t_minus_1 = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # 1. Historie stavů: Na začátku jsou všechny stejné jako initial_state
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.clone()
        
        # 2. Historie predikce: Nemá smysl volat f(), prostě zkopírujeme init state
        self.x_pred_t_minus_1 = self.x_filtered_t_minus_1.clone()
        
        # 3. Historie měření: Dopočítáme h(x)
        with torch.no_grad():
            self.y_t_minus_1 = self.system_model.h(self.x_filtered_t_minus_1).detach()
        
        # 4. GRU Hidden State: MUSÍ BÝT NULY!
        # Náhodná inicializace (randn) způsobuje "šok" v prvním kroku.
        self.h_prev = torch.zeros(
            self.dnn.gru.num_layers, 
            batch_size, 
            self.dnn.gru.hidden_size, 
            device=self.device
        )

    def step(self, y_t):
        # 1. PREDIKCE (pro aktuální čas t)
        # x_{t|t-1} = f(x_{t-1|t-1})
        x_predicted = self.system_model.f(self.x_filtered_t_minus_1)
        # y_{t|t-1} = h(x_{t|t-1})
        y_predicted = self.system_model.h(x_predicted)

        # Ošetření dimenzí
        if y_predicted.dim() == 1: y_predicted = y_predicted.unsqueeze(-1)
        if y_t.dim() == 1: y_t = y_t.unsqueeze(-1)

        # 2. VÝPOČET VSTUPŮ PRO DNN
        # x_{t-1|t-1} - x_{t-1|t-2}
        state_inno = self.x_filtered_t_minus_1 - self.x_pred_t_minus_1
        norm_state_inno = self.log_modulus(state_inno)
        # y_t - y_{t|t-1}
        inovation = y_t - y_predicted
        norm_innovation = self.log_modulus(inovation)
        # x_{t-1|t-1} - x_{t-2|t-2}
        diff_state = self.x_filtered_t_minus_1 - self.x_filtered_t_minus_2
        norm_diff_state = self.log_modulus(diff_state)
        # y_t - y_{t-1}
        diff_obs = y_t - self.y_t_minus_1
        norm_diff_obs = self.log_modulus(diff_obs)

        # 3. PRŮCHOD SÍTÍ A KOREKCE
        K_vec, h_new, regs = self.dnn(norm_state_inno, norm_innovation, norm_diff_state, norm_diff_obs, self.h_prev)
        
        K = K_vec.reshape(-1, self.state_dim, self.obs_dim)
        # K = torch.clamp(K, min=-5.0, max=5.0)
        correction = torch.bmm(K, inovation.unsqueeze(-1)).squeeze(-1)
        # x_{t|t} = x_{t|t-1} + K * inovace
        x_filtered = x_predicted + correction
        
        # 4. AKTUALIZACE STAVŮ PRO PŘÍŠTÍ VOLÁNÍ (pro t+1)
        # Posuneme stavy v čase
        self.x_filtered_t_minus_2 = self.x_filtered_t_minus_1.detach()
        self.x_filtered_t_minus_1 = x_filtered.detach() # .clone() není nutné
        self.x_pred_t_minus_1 = x_predicted.detach()
        self.y_t_minus_1 = y_t.detach()
        self.h_prev = h_new.detach()
        
        return x_filtered, regs
    
    # --- POMOCNÁ FUNKCE: LOG MODULUS ---
    def log_modulus(self, x):
        """Log-compression zachovávající znaménko."""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    def init_weights(self) -> None:
            """Stabilní inicializace vah."""
            print("INFO: Aplikuji upravenou inicializaci pro BKN.")
            for name, m in self.dnn.named_modules():
                if isinstance(m, nn.Linear):
                    # Použijeme kaiming, ale s malým gainem, aby váhy nebyly moc divoké
                    # Ale NE tak malé, aby zabily gradient
                    init.kaiming_uniform_(m.weight, a=0.1, nonlinearity='relu') 
                    if m.bias is not None: init.zeros_(m.bias)
                
                elif isinstance(m, nn.LayerNorm):
                    init.constant_(m.bias, 0)
                    init.constant_(m.weight, 1.0)
                
                elif isinstance(m, nn.GRU):
                    for param_name, param in m.named_parameters():
                        if 'weight' in param_name: 
                            # Ortogonální init je pro GRU nejlepší
                            init.orthogonal_(param.data) 
                        elif 'bias' in param_name: 
                            param.data.fill_(0)

            # Výstupní vrstva - zde chceme začít s menšími hodnotami, ale ne nulou!
            if hasattr(self.dnn, 'output_layer') and len(self.dnn.output_layer) > 0:
                last_layer = self.dnn.output_layer[-1] 
                if isinstance(last_layer, nn.Linear):
                    # Změna z 1e-4 na 0.01 nebo 0.1
                    # To zajistí, že K bude malé (třeba 0.1), ale různé pro každý dropout pass
                    init.uniform_(last_layer.weight, -0.1, 0.1) 
                    if last_layer.bias is not None:
                        init.zeros_(last_layer.bias)
                    print("DEBUG: Výstupní vrstva inicializována konzervativně (interval -0.1 až 0.1).")
                    
    def detach_hidden(self):
        """
        Odpojí interní stavy od výpočetního grafu (pro TBPTT).
        Tím se 'zapomene' historie gradientů, ale hodnoty stavů zůstanou.
        """
        if self.x_filtered_t_minus_1 is not None:
            self.x_filtered_t_minus_1 = self.x_filtered_t_minus_1.detach()
        if self.x_filtered_t_minus_2 is not None:
            self.x_filtered_t_minus_2 = self.x_filtered_t_minus_2.detach()
        if self.x_pred_t_minus_1 is not None:
            self.x_pred_t_minus_1 = self.x_pred_t_minus_1.detach()
        if self.y_t_minus_1 is not None:
            self.y_t_minus_1 = self.y_t_minus_1.detach()
        if self.h_prev is not None:
            self.h_prev = self.h_prev.detach()