import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_KalmanNet import DNN_KalmanNet

class StateKalmanNet(nn.Module):
    def __init__(self,system_model, device, hidden_size_multiplier=10):
        super(StateKalmanNet, self).__init__()
        self.returns_covariance = False
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_KalmanNet(system_model, hidden_size_multiplier).to(device)


    def reset(self,batch_size=1, initial_state=None):
        """
        Resetuje vnitřní stav filtru. Volá se na začátku každé nové sekvence.
        """

        if initial_state is not None:
            # x_{t-1|t-1}: A posteriori odhad stavu z minulého kroku.
            # Tvar: [batch_size, state_dim]
            self.x_filtered_prev = initial_state.clone()
        else:
            self.x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # delta_x_{t-1}: Rozdíl x_{t-1|t-1} - x_{t-1|t-2}.
        # Tento vstup do DNN reprezentuje velikost korekce v minulém kroku.
        # Na začátku je nulový.
        # Tvar: [batch_size, state_dim]
        self.delta_x_prev = torch.zeros_like(self.x_filtered_prev)

        # h_{t-1}: Skrytý stav ("paměť") GRU z minulého kroku.
        # Tvar: [Num_Layers, batch_size, Hidden_Dim]
        self.h_prev = torch.zeros(1, batch_size, self.dnn.gru.hidden_size, device=self.device)
        
    def step(self,y_t):
        """
        Provede jeden kompletní krok filtrace pro jedno měření y_t s dimenzí batch_size.
        """

        batch_size = y_t.shape[0]

        # --- PREDIKCE ---
        # x_{t|t-1} = f(x_{t-1|t-1})
        # Vypočítáme apriori odhad stavu na základě odhadu z minulého kroku.
        x_predicted = self.system_model.f(self.x_filtered_prev)
        # `x_predicted` Tvar: [batch_size, state_dim]


        # y_{t|t-1} = h(x_{t|t-1})
        # Vypočítáme predikované měření.
        y_predicted = self.system_model.h(x_predicted)
        # `y_predicted` Tvar: [batch_size, obs_dim]

        # Inovace: Rozdíl mezi skutečným a predikovaným měřením.
        # Tvar: [batch_size, obs_dim]
        innovation = y_t - y_predicted

        # --- VSTUPY PRO DNN ---
        # V článku KalmanNet se tyto normalizované vstupy nazývají F2 a F4.
        # Normalizace stabilizuje trénink, síť se učí jen ze "směru" chyb.
        
        # normalizovaný rozdíl z minulého kroku
        norm_delta_x = F.normalize(self.delta_x_prev, p=2, dim=1, eps=1e-12)
        # `norm_delta_x` Tvar: [batch_size, state_dim]
        
        # normalizovaná inovace
        norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
        # `norm_innovation` Tvar: [batch_size, obs_dim]

        # Zřetězení vstupů pro DNN
        nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)
        # `nn_input` Tvar: [batch_size, state_dim + obs_dim]


        # --- VÝPOČET ZISKU A KOREKCE ---
        # Voláme naši DNN, abychom získali vektorizovaný zisk a nový skrytý stav.
        K_vec, h_new = self.dnn(nn_input, self.h_prev)
        # `K_vec` Tvar: [batch_size, state_dim * obs_dim]
        # `h_new` Tvar: [1, batch_size, hidden_dim]

        # Přeformátujeme vektor na matici Kalmanova zisku pro každou položku v dávce.
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
        # `K` Tvar: [batch_size, state_dim, obs_dim]

        # `innovation.unsqueeze(-1)` -> [batch_size, obs_dim, 1]
        # `K @ ...` -> [batch_size, state_dim, obs_dim] @ [batch_size, obs_dim, 1] -> [batch_size, state_dim, 1]
        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)

        # x_{t|t} = x_{t|t-1} + K_t * (y_t - y_{t|t-1})
        # Finální a posteriori odhad stavu.
        # x_filtered = x_predicted + correction
        # `x_filtered` Tvar: [batch_size, state_dim]

        x_filtered_unclamped = x_predicted + correction
        # `x_filtered_unclamped` Tvar: [batch_size, state_dim]

        # --- NOVÝ BLOK: OMEZENÍ STAVU (BEZ INPLACE) ---
        
        # Ořežeme jednotlivé komponenty. 
        # .clamp() vrací NOVÝ tenzor, nemění původní.
        px_clamped = x_filtered_unclamped[:, 0].clamp(self.system_model.min_x, self.system_model.max_x)
        py_clamped = x_filtered_unclamped[:, 1].clamp(self.system_model.min_y, self.system_model.max_y)

        max_vel = 200.0  
        vx_clamped = x_filtered_unclamped[:, 2].clamp(-max_vel, max_vel)
        vy_clamped = x_filtered_unclamped[:, 3].clamp(-max_vel, max_vel)
        if torch.any((vx_clamped == max_vel) | (vx_clamped == -max_vel)):
            print(f"Varování: Došlo k omezení rychlosti v ose X (max_vel={max_vel}).")
        
        if torch.any((vy_clamped == max_vel) | (vy_clamped == -max_vel)):
            print(f"Varování: Došlo k omezení rychlosti v ose Y (max_vel={max_vel}).")
        # --- KONEC KONTROLY ---
        # Sestavíme z ořezaných komponent *nový* tenzor `x_filtered`.
        # Toto je teď náš finální, ořezaný stav s platným grafem.
        x_filtered = torch.stack([px_clamped, py_clamped, vx_clamped, vy_clamped], dim=1)
        # --- KONEC BLOKU ---
        # --- AKTUALIZACE STAVŮ PRO PŘÍŠTÍ KROK ---
        # Uložíme si hodnoty z aktuálního kroku `t` pro použití v kroku `t+1`.
        
        # delta_x_t = x_{t|t} - x_{t|t-1}
        self.delta_x_prev = x_filtered - x_predicted
        
        # x_{t-1|t-1} se pro další krok stane x_{t|t}
        self.x_filtered_prev = x_filtered
        
        # h_{t-1} se pro další krok stane h_t
        self.h_prev = h_new

        # finální odhad stavu.
        # Tvar: [batch_size, state_dim]
        return x_filtered