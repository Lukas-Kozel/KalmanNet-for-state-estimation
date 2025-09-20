import torch
import torch.nn as nn
import torch.nn.functional as F
from .DNN_BayesianKalmanNet import DNN_BayesianKalmanNet

class StateBayesianKalmanNet(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(StateBayesianKalmanNet, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dnn = DNN_BayesianKalmanNet(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)
        
        self.reset()

    def reset(self, batch_size=1, num_samples=10, initial_state=None):
        """
        Resetuje vnitřní stavy filtru. Udržujeme celý ansámbl stavů.
        """
        if initial_state is not None:
            self.x_filtered_prev_ensemble = initial_state.detach().clone().unsqueeze(0).repeat(num_samples, 1, 1)
        else:
            self.x_filtered_prev_ensemble = torch.zeros(num_samples, batch_size, self.state_dim, device=self.device)
        
        self.x_filtered_prev_prev_mean = torch.zeros(batch_size, self.state_dim, device=self.device)
        self.x_pred_prev_mean = torch.zeros(batch_size, self.state_dim, device=self.device)
        self.y_prev = torch.zeros(batch_size, self.obs_dim, device=self.device)
        
        self.h_prev_ensemble = torch.randn(
            num_samples, self.dnn.gru.num_layers, 
            batch_size, self.dnn.gru.hidden_size, 
            device=self.device
        ).detach()

    def step(self, y_t, num_samples=10):
        num_samples, batch_size, _ = self.x_filtered_prev_ensemble.shape
        
        # KROK 1: PREDIKCE (pro celý ansámbl stavů)
        x_flat = self.x_filtered_prev_ensemble.reshape(-1, self.state_dim)
        x_predicted_flat = self.system_model.f(x_flat)
        x_predicted_ensemble = x_predicted_flat.reshape(num_samples, batch_size, self.state_dim)
        
        # KROK 2: VÝPOČET PRŮMĚRŮ pro DNN vstupy
        x_predicted_mean = x_predicted_ensemble.mean(dim=0)
        x_filtered_prev_mean = self.x_filtered_prev_ensemble.mean(dim=0)
        y_predicted_mean = self.system_model.h(x_predicted_mean)
        
        # Vstupy do DNN (jsou pro všechny stejné)
        state_inno = x_filtered_prev_mean - self.x_pred_prev_mean
        inovation = y_t - y_predicted_mean
        diff_state = x_filtered_prev_mean - self.x_filtered_prev_prev_mean
        diff_obs = y_t - self.y_prev

        # KROK 3: GENEROVÁNÍ ANSÁMBLU KOREKCÍ
        x_filtered_ensemble_list = []
        h_new_ensemble_list = []
        regularization_ensemble = []

        for j in range(num_samples):
            h_j_prev = self.h_prev_ensemble[j]
            K_vec, h_j_new, regs = self.dnn(state_inno, inovation, diff_state, diff_obs, h_j_prev)
            K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
            
            # Používáme průměrnou inovaci pro všechny, ale aplikujeme ji na j-tý predikovaný stav
            correction = (K @ inovation.unsqueeze(-1)).squeeze(-1)
            
            x_filtered_j = x_predicted_ensemble[j] + correction
            
            x_filtered_ensemble_list.append(x_filtered_j)
            h_new_ensemble_list.append(h_j_new)
            
            # Sečteme regularizace z obou dropout vrstev pro tento jeden vzorek
            regularization_ensemble.append(torch.sum(torch.stack(regs)))

        x_filtered_ensemble_tensor = torch.stack(x_filtered_ensemble_list, dim=0)
        
        # KROK 4: AKTUALIZACE STAVŮ
        self.x_filtered_prev_prev_mean = x_filtered_prev_mean.clone().detach()
        self.x_pred_prev_mean = x_predicted_mean.clone().detach()
        self.y_prev = y_t.clone().detach()
        
        self.x_filtered_prev_ensemble = x_filtered_ensemble_tensor.clone().detach()
        self.h_prev_ensemble = torch.stack(h_new_ensemble_list, dim=0).clone().detach()
        
        raw_regularization = torch.stack(regularization_ensemble)
        
        return x_filtered_ensemble_tensor, raw_regularization