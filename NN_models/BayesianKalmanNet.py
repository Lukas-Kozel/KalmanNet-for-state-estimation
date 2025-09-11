import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianKalmanNet(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, dropout_prob=0.2):
        super(BayesianKalmanNet, self).__init__()
        
        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.dropout_prob = dropout_prob

        self.f = system_model.f
        self.h = system_model.h

        moments_dim = self.state_dim * self.state_dim + self.obs_dim * self.obs_dim
        self.hidden_dim = moments_dim * hidden_size_multiplier
        input_dim = self.state_dim + self.obs_dim

        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.state_dim * self.obs_dim)
        self.dropout2 = nn.Dropout(self.dropout_prob)

    def forward(self, y_seq, num_samples=20):
        batch_size, seq_len, _ = y_seq.shape
        
        x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        delta_x_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        h_gru_ensemble = torch.zeros(num_samples, 1, batch_size, self.hidden_dim, device=self.device)

        x_filtered_trajectory = []
        P_filtered_trajectory = []

        for t in range(seq_len):
            y_t = y_seq[:, t, :]
            
            x_predicted_list = [self.f(x.unsqueeze(-1)) for x in x_filtered_prev]
            x_predicted = torch.stack(x_predicted_list).squeeze(-1)

            y_predicted_list = [self.h(x.unsqueeze(-1)) for x in x_predicted]
            y_predicted = torch.stack(y_predicted_list).squeeze(-1)
            
            innovation = y_t - y_predicted
            
            x_filtered_ensemble_t = []
            new_h_gru_list = []

            for j in range(num_samples):
                norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
                norm_delta_x = F.normalize(delta_x_prev, p=2, dim=1, eps=1e-12)
                nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)

                activated_input = F.relu(self.input_layer(nn_input))
                activated_input_dropped = self.dropout1(activated_input)
                
                h_j_old = h_gru_ensemble[j]
                out_gru, h_j_new = self.gru(activated_input_dropped.unsqueeze(0), h_j_old)
                new_h_gru_list.append(h_j_new)
                
                out_gru_squeezed = out_gru.squeeze(0)
                K_vec_raw = self.output_layer(out_gru_squeezed)
                K_vec = self.dropout2(K_vec_raw)
                K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
                
                correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)
                x_filtered_j = x_predicted + correction
                x_filtered_ensemble_t.append(x_filtered_j)

            h_gru_ensemble = torch.stack(new_h_gru_list, dim=0)
            
            ensemble_at_t = torch.stack(x_filtered_ensemble_t, dim=0)
            x_filtered_at_t = torch.mean(ensemble_at_t, dim=0)
            
            diff = ensemble_at_t - x_filtered_at_t
            P_filtered_at_t = (diff.unsqueeze(-1) * diff.unsqueeze(-2)).mean(dim=0)

            x_filtered_trajectory.append(x_filtered_at_t)
            P_filtered_trajectory.append(P_filtered_at_t)
            
            
            delta_x_prev = x_filtered_at_t - x_predicted
            x_filtered_prev = x_filtered_at_t.clone()
            
        return torch.stack(x_filtered_trajectory, dim=1), torch.stack(P_filtered_trajectory, dim=1)