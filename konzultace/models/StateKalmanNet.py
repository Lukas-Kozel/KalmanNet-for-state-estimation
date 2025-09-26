import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_KalmanNet import DNN_KalmanNet

class StateKalmanNet(nn.Module):
    def __init__(self,system_model, device, hidden_size_multiplier=10):
        super(StateKalmanNet, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_KalmanNet(system_model, hidden_size_multiplier).to(device)

        self.reset()

    def reset(self,batch_size=1, initial_state=None):

        if initial_state is not None:
            self.x_filtered_prev = initial_state.clone()
        else:
            self.x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        self.delta_x_prev = torch.zeros_like(self.x_filtered_prev)
        self.h_prev = torch.zeros(1, batch_size, self.dnn.gru.hidden_size, device=self.device)
        
    def step(self,y_t):
        """
        Provede jeden kompletní krok filtrace pro jedno měření y_t s dimenzí batch_size.
        """

        batch_size = y_t.shape[0]

        x_predicted_list = [self.system_model.f(x.unsqueeze(-1)) for x in self.x_filtered_prev]
        x_predicted = torch.stack(x_predicted_list).squeeze(-1)

        y_predicted_list = [self.system_model.h(x.unsqueeze(-1)) for x in x_predicted]
        y_predicted = torch.stack(y_predicted_list).squeeze(-1)

        innovation = y_t - y_predicted

        norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_delta_x = F.normalize(self.delta_x_prev, p=2, dim=1, eps=1e-12)
        nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

        K_vec, h_new = self.dnn(nn_input, self.h_prev)
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)

        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        x_filtered = x_predicted + correction

        self.delta_x_prev = x_filtered - x_predicted
        self.x_filtered_prev = x_filtered
        self.h_prev = h_new

        return x_filtered