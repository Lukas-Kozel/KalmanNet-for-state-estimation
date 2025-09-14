# Soubor: KalmanNet2.py
import torch
import torch.nn as nn
from argparse import Namespace

from .kalman_net_base import KalmanNetNN 

class KalmanNet2(nn.Module):
    def __init__(self, system_model, in_mult=20, out_mult=40):
        super(KalmanNet2, self).__init__()
        
        self.model = KalmanNetNN()


        args = Namespace(
            use_cuda=torch.cuda.is_available(),
            n_batch=0,
            in_mult_KNet=in_mult,
            out_mult_KNet=out_mult
        )

        OfficialSysModel = Namespace(
            f=torch.vmap(system_model.f, in_dims=(0,)), 
            h=torch.vmap(system_model.h, in_dims=(0,)),
            m=system_model.state_dim,
            n=system_model.obs_dim,
            prior_Q=torch.zeros(system_model.state_dim, system_model.state_dim),
            prior_Sigma=torch.zeros(system_model.state_dim, system_model.state_dim),
            prior_S=torch.zeros(system_model.obs_dim, system_model.obs_dim)
        )

        self.model.NNBuild(OfficialSysModel, args)

    def forward(self, y_seq):
        batch_size, seq_len, _ = y_seq.shape
        device = y_seq.device

        self.model.batch_size = batch_size

        initial_state = torch.zeros(batch_size, self.model.m, 1, device=device)
        self.model.InitSequence(initial_state, seq_len)
        self.model.init_hidden_KNet()

        x_hat_list = []
        for t in range(seq_len):
            y_t = y_seq[:, t, :].unsqueeze(-1) 
            x_hat_t = self.model.forward(y_t)
            x_hat_list.append(x_hat_t.squeeze(-1))


        return torch.stack(x_hat_list, dim=1)