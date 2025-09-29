import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_KalmanNet import DNN_KalmanNet

class StateKalmanNetWithKnownR(nn.Module):
    def __init__(self,system_model, device, hidden_size_multiplier=10):
        super(StateKalmanNetWithKnownR, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.R = self.system_model.R.to(device)
        self.P0 = system_model.P0.to(device)

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

        # self.x_filtered_prev je již dávka [B, D_state], můžeme ji poslat přímo
        x_predicted = self.system_model.f(self.x_filtered_prev)

        # x_predicted je také dávka, můžeme ji poslat přímo
        y_predicted = self.system_model.h(x_predicted)

        innovation = y_t - y_predicted

        norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
        norm_delta_x = F.normalize(self.delta_x_prev, p=2, dim=1, eps=1e-12)
        nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

        K_vec, h_new = self.dnn(nn_input, self.h_prev)
        K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)

        correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
        x_filtered = x_predicted + correction

        P_filtered_list = []
        with torch.no_grad(): # pro jistotu vypnutí gradienty
            for i in range(batch_size):
                K_i = K[i]  # Kalmanův zisk pro i-tý prvek v batche

                x_filtered_i = x_filtered[i]

                try:
                    if self.system_model.is_linear_h:
                        H_i = self.system_model.H
                    else:
                        H_i = torch.autograd.functional.jacobian(self.system_model.h, x_filtered_i).reshape(self.obs_dim, self.state_dim)
                    I = torch.eye(self.state_dim, device=self.device)
                    if self.state_dim == 1 or self.obs_dim == 1:
                        Htilde_i= 1/(H_i**2)
                        I_KH_i = (1 - K_i * H_i)
                        P_predict_i = 1/ (I_KH_i) * K_i * self.R * H_i * Htilde_i
                        P_filtered_i = I_KH_i * P_predict_i * I_KH_i + K_i * self.R * K_i
                        P_filtered_list.append(P_filtered_i)
                    else:    
                        Htilde_i = torch.linalg.inv(H_i.T @ H_i)
                        I_KH_i = (torch.eye(self.state_dim, device=self.device) - K_i @ H_i)
                        P_predict_i = torch.linalg.inv(I_KH_i) @ K_i @ self.R @ H_i @ Htilde_i
                        P_filtered_i = I_KH_i @ P_predict_i @ I_KH_i.T + K_i @ self.R @ K_i.T
                        P_filtered_list.append(P_filtered_i)

                except:
                    print("Failed at Uncertainty Analysis of KalmanNet V1")

        P_filtered = torch.stack(P_filtered_list)
        self.delta_x_prev = x_filtered - x_predicted
        self.x_filtered_prev = x_filtered
        self.h_prev = h_new

        return x_filtered, P_filtered