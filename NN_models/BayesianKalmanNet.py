import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ConcreteDropout import ConcreteDropout

class BayesianKalmanNet(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1):
        super(BayesianKalmanNet, self).__init__()
        
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.f = system_model.f
        self.h = system_model.h

        H1 = (self.state_dim + self.obs_dim) * hidden_size_multiplier
        H2 = (self.state_dim * self.obs_dim) * output_layer_multiplier
        input_dim = self.state_dim + self.obs_dim
        output_dim = self.state_dim * self.obs_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, H1),
            nn.ReLU()
        )
        self.concrete_dropout1 = ConcreteDropout(device=device)

        self.gru = nn.GRU(H1, H1, num_layers=num_gru_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, output_dim)
        )
        self.concrete_dropout2 = ConcreteDropout(device=device)

    def forward(self, y_seq, num_samples=20):
        batch_size, seq_len, _ = y_seq.shape
        
        x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        delta_x_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        h_gru_ensemble = torch.zeros(num_samples, 1, batch_size, self.hidden_dim, device=self.device)

        x_filtered_trajectory = []
        P_filtered_trajectory = []

        # Seznam pro sběr regularizačních členů z celé sekvence
        regularization_trajectory = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            x_predicted_list = [self.f(x.unsqueeze(-1)) for x in x_filtered_prev]
            x_predicted = torch.stack(x_predicted_list).squeeze(-1)

            y_predicted_list = [self.h(x.unsqueeze(-1)) for x in x_predicted]
            y_predicted = torch.stack(y_predicted_list).squeeze(-1)
            
            innovation = y_t - y_predicted
            
            x_filtered_ensemble_t = []
            new_h_gru_list = []
            regularization_at_t_list = []

            for j in range(num_samples):
                norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
                norm_delta_x = F.normalize(delta_x_prev, p=2, dim=1, eps=1e-12)

                nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

                
                activated_input, reg1 = self.concrete_dropout1(nn_input, self.input_layer)
                
                
                h_j_old = h_gru_ensemble[j]
                out_gru, h_j_new = self.gru(activated_input.unsqueeze(0), h_j_old)
                new_h_gru_list.append(h_j_new)
                
                out_gru_squeezed = out_gru.squeeze(0)

                K_vec, reg2 = self.concrete_dropout2(out_gru_squeezed, self.output_layer)

                regularization_at_t_list.append(reg1 + reg2)

                K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
                
                correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
                x_filtered_j = x_predicted + correction
                x_filtered_ensemble_t.append(x_filtered_j)

            h_gru_ensemble = torch.stack(new_h_gru_list, dim=0)

            avg_regularization_at_t = torch.mean(torch.stack(regularization_at_t_list))
            regularization_trajectory.append(avg_regularization_at_t)
            
            ensemble_at_t = torch.stack(x_filtered_ensemble_t, dim=0)
            x_filtered_at_t = torch.mean(ensemble_at_t, dim=0)
            
            diff = ensemble_at_t - x_filtered_at_t
            P_filtered_at_t = (diff.unsqueeze(-1) * diff.unsqueeze(-2)).mean(dim=0)

            x_filtered_trajectory.append(x_filtered_at_t)
            P_filtered_trajectory.append(P_filtered_at_t)
            
            
            delta_x_prev = x_filtered_at_t - x_predicted
            x_filtered_prev = x_filtered_at_t.clone()

        total_regularization = torch.mean(torch.stack(regularization_trajectory))
        
        return torch.stack(x_filtered_trajectory, dim=1), torch.stack(P_filtered_trajectory, dim=1), total_regularization