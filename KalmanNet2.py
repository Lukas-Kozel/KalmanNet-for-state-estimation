# Soubor: KalmanNet2.py
import torch
import torch.nn as nn
from argparse import Namespace

# Importujeme oficiální třídu z jejího souboru
from kalman_net_base import KalmanNetNN 

class KalmanNet2(nn.Module):
    """
    Adapter (wrapper) pro oficiální implementaci KalmanNet Architektury #2.
    
    Tato třída má stejné rozhraní jako naše předchozí modely (přijímá celou
    sekvenci ve forward), ale interně používá a správně ovládá oficiální, 
    stavový kód, který zpracovává data krok po kroku.
    """
    def __init__(self, system_model, in_mult=20, out_mult=40):
        super(KalmanNet2, self).__init__()
        
        # Interně si držíme instanci oficiálního modelu
        self.model = KalmanNetNN()

        # Oficiální kód vyžaduje 'args' objekt, tak si ho vytvoříme
        # Namespace je jednoduchý způsob, jak vytvořit objekt s atributy
        args = Namespace(
            use_cuda=torch.cuda.is_available(),
            n_batch=0, # Bude aktualizováno dynamicky ve forward
            in_mult_KNet=in_mult,
            out_mult_KNet=out_mult
        )
        
        # Oficiální kód má jinou strukturu pro SysModel, přizpůsobíme ji
        OfficialSysModel = Namespace(
            f=torch.vmap(system_model.f, in_dims=(0,)), # Musíme použít vmap pro batching
            h=torch.vmap(system_model.h, in_dims=(0,)), # Musíme použít vmap pro batching
            m=system_model.state_dim,
            n=system_model.obs_dim,
            prior_Q=torch.zeros(system_model.state_dim, system_model.state_dim),
            prior_Sigma=torch.zeros(system_model.state_dim, system_model.state_dim),
            prior_S=torch.zeros(system_model.obs_dim, system_model.obs_dim)
        )

        # Zavoláme komplexní inicializační metodu oficiálního modelu
        self.model.NNBuild(OfficialSysModel, args)

    def forward(self, y_seq):
        # y_seq má očekávaný tvar (batch_size, seq_len, obs_dim)
        batch_size, seq_len, _ = y_seq.shape
        device = y_seq.device

        # Dynamicky nastavíme batch_size pro oficiální model
        self.model.batch_size = batch_size

        # Inicializace stavů oficiálního modelu pro novou dávku sekvencí
        initial_state = torch.zeros(batch_size, self.model.m, 1, device=device)
        self.model.InitSequence(initial_state, seq_len)
        self.model.init_hidden_KNet()

        x_hat_list = []
        # Smyčka přes časové kroky (jak to očekává oficiální kód)
        for t in range(seq_len):
            y_t = y_seq[:, t, :].unsqueeze(-1) # Tvar (batch, obs_dim, 1)
            x_hat_t = self.model.forward(y_t)
            x_hat_list.append(x_hat_t.squeeze(-1)) # Uložíme s tvarem (batch, state_dim)

        # Sestavíme výsledky do jednoho tenzoru (batch_size, seq_len, state_dim)
        return torch.stack(x_hat_list, dim=1)