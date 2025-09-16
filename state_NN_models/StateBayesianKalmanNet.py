import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNet

class StateBayesianKalmanNet(nn.Module):
    def __init__(self,system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(StateBayesianKalmanNet, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_BayesianKalmanNet(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        self.reset()

    def reset(self, batch_size=1, num_samples=10,initial_state=None):
        if initial_state is not None:
            # Pokud je stav zadán (při tréninku), použijeme ho
            self.x_filtered_prev = initial_state.clone()
        else:
            # Pokud není zadán (při __init__), použijeme nuly
            self.x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
        
        # Ostatní stavy se resetují relativně k počátečnímu stavu
        self.x_filtered_prev_prev = torch.zeros_like(self.x_filtered_prev)
        self.y_prev = torch.zeros(batch_size, self.obs_dim, device=self.device)
        self.delta_x_prev = torch.zeros_like(self.x_filtered_prev)
        
        self.h_prev_ensemble = torch.zeros(
            num_samples, self.dnn.gru.num_layers, 
            batch_size, self.dnn.gru.hidden_size, 
            device=self.device
        )

    def step(self,y_t, num_samples=10):
        """
        Provede jeden kompletní krok filtrace pro jedno měření y_t s dimenzí batch_size.
        """

        batch_size = y_t.shape[0]

        x_predicted_list = [self.system_model.f(x.unsqueeze(-1)) for x in self.x_filtered_prev]
        x_predicted = torch.stack(x_predicted_list).squeeze(-1)

        y_predicted_list = [self.system_model.h(x.unsqueeze(-1)) for x in x_predicted]
        y_predicted = torch.stack(y_predicted_list).squeeze(-1)

        state_inno = self.delta_x_prev
        residual = y_t - y_predicted
        diff_state = self.x_filtered_prev - self.x_filtered_prev_prev
        diff_obs = self.y_prev

        x_filtered_ensemble = []
        h_new_ensemble = []
        regularization_ensemble = []
        for i in range(num_samples):
            h_j_prev = self.h_prev_ensemble[i]
            
            K_vec, h_j_new, regs = self.dnn(state_inno, residual, diff_state, diff_obs, h_j_prev)
            
            K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
            correction = (K @ residual.unsqueeze(-1)).squeeze(-1)
            x_filtered_j = x_predicted + correction
            
            x_filtered_ensemble.append(x_filtered_j)
            h_new_ensemble.append(h_j_new)
            regularization_ensemble.append(torch.sum(torch.stack(regs)))

        x_filtered_ensemble_tensor = torch.stack(x_filtered_ensemble, dim=0)
        x_filtered_final = x_filtered_ensemble_tensor.mean(dim=0)
        
        diff = x_filtered_ensemble_tensor - x_filtered_final
        P_filtered_final = (diff.unsqueeze(-1) * diff.unsqueeze(-2)).mean(dim=0)
        
        raw_regularization = torch.stack(regularization_ensemble)

        self.delta_x_prev = x_filtered_final - x_predicted
        self.x_filtered_prev_prev = self.x_filtered_prev.clone()
        self.y_prev = y_t.clone()
        self.x_filtered_prev = x_filtered_final
        self.h_prev_ensemble = torch.stack(h_new_ensemble, dim=0)
        
        return x_filtered_final, P_filtered_final, raw_regularization