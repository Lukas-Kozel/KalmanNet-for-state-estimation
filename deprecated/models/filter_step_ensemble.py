import torch.nn as nn
from .dnn import DNN_BayesianKalmanNet
import torch

class step_StateBayesianKalmanNet(nn.Module):
    def __init__(self, system_model, device, hidden_size_multiplier=10, output_layer_multiplier=4, num_gru_layers=1,init_min_dropout=0.5,init_max_dropout=0.8):
        super(step_StateBayesianKalmanNet, self).__init__()

        self.device = device
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.num_gru_layers = num_gru_layers

        self.dnn = DNN_BayesianKalmanNet(system_model, hidden_size_multiplier, output_layer_multiplier, num_gru_layers, init_min_dropout, init_max_dropout).to(device)

        self.h_init_master = torch.randn(
            self.dnn.gru.num_layers, 
            1, # batch_size=1
            self.dnn.gru.hidden_size, 
            device=self.device
        ).detach()

        # Atributy pro ukládání stavů mezi kroky
        self.x_filtered_prev = None
        self.x_filtered_prev_prev = None
        self.x_pred_prev = None
        self.y_prev = None
        self.h_prev_ensemble = None 
        self.is_first_step = True
        self.reset()

    def reset(self, batch_size=1, initial_state=None):
        if initial_state is not None:
            self.x_filtered_prev = initial_state.detach().clone()
        else:
            self.x_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)
            
        self.is_first_step = True

        self.h_prev_ensemble = None

    def step(self, y_t, J_samples=20):
        batch_size = y_t.shape[0]

        x_predicted = self.system_model.f(self.x_filtered_prev)
        
        if self.is_first_step:
            self.x_filtered_prev_prev = self.x_filtered_prev.clone()
            self.x_pred_prev = x_predicted.clone()
            self.y_prev = self.system_model.h(x_predicted)
            
            # `h_init_master` má tvar [L, 1, H]. `expand` ho roztáhne na [L, B, H].
            h_init_batch = self.h_init_master.expand(-1, batch_size, -1)
            self.h_prev_ensemble = torch.stack([h_init_batch] * J_samples, dim=0) # Tvar [J, L, B, H]

        y_predicted = self.system_model.h(x_predicted)
        if y_predicted.dim() == 1: y_predicted = y_predicted.unsqueeze(-1)
        if y_t.dim() == 1: y_t = y_t.unsqueeze(-1)
            
        # Výpočet vstupních příznaků pro DNN (zůstává stejný)
        state_inno = self.x_filtered_prev - self.x_pred_prev
        inovation = y_t - y_predicted
        diff_state = self.x_filtered_prev - self.x_filtered_prev_prev
        diff_obs = y_t - self.y_prev

        
        x_filtered_samples = []
        regularization_samples = []
        h_new_samples = []

        
        for j in range(J_samples):
            h_prev_j = self.h_prev_ensemble[j]
            
            K_vec, h_new_j, total_reg = self.dnn(state_inno, inovation, diff_state, diff_obs, h_prev_j)
            
            K_j = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)
            
            correction_batch = []
            for b in range(batch_size):
                K_b = K_j[b]          
                inovation_b = inovation[b]
                
                correction_b = K_b @ inovation_b
                correction_batch.append(correction_b)
            
            correction = torch.stack(correction_batch, dim=0)
            x_filtered_j = x_predicted + correction
            
            x_filtered_samples.append(x_filtered_j)
            regularization_samples.append(total_reg)
            h_new_samples.append(h_new_j)

        x_filtered_ensemble = torch.stack(x_filtered_samples, dim=0).permute(1, 0, 2)
        final_x_filtered = x_filtered_ensemble.mean(dim=1)
        final_P_filtered_diag = x_filtered_ensemble.var(dim=1)

        regularization_tensor = torch.stack(regularization_samples, dim=0)
        regularization_for_step = regularization_tensor.mean(dim=0)

        self.x_pred_prev = x_predicted.detach()
        self.x_filtered_prev_prev = self.x_filtered_prev.clone().detach()
        self.y_prev = y_t.clone().detach()
        
        self.x_filtered_prev = final_x_filtered.clone().detach()
        self.h_prev_ensemble = torch.stack(h_new_samples, dim=0).detach()
        
        self.is_first_step = False

        return final_x_filtered, final_P_filtered_diag, regularization_for_step, x_filtered_ensemble