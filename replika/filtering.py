from dnn import DNN_KalmanNet_GSS
import torch



class KalmanNet_Filter():
    def __init__(self,system_model):
        self.num_of_bayesian_layers = int(2)

        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim
        self.system_model = system_model

        self.kf_net = DNN_KalmanNet_GSS(self.state_dim, self.obs_dim)
        self.init_state = system_model.Ex0
        self.reset(clean_history=True)

        # For Uncertainty analysis:
        # self.R = float(config['EKF']['r2']) * torch.eye(self.obs_dim)
        self.R = system_model.R
        self.R = torch.diag(torch.diag(self.R)) # The Filter Does not know about the off diagonal

    def reset(self, clean_history=False):
        # --- ZJEDNODUŠENÍ: Odstraněno `scz` ---
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        self.regularization = torch.zeros((self.num_of_bayesian_layers, 1), device=self.device).detach()

        if clean_history:
            self.state_history = self.init_state.detach().clone()
            
            # --- ZJEDNODUŠENÍ: Pracujeme s plnou kovarianční maticí ---
            init_cov_full = self.system_model.P0.detach().clone()

            self.cov_trace_history = torch.zeros((1,), device=self.device)
            # Ukládáme historii plných matic. `unsqueeze(2)` přidá dimenzi pro čas.
            self.cov_pred_byK_opt1_history = init_cov_full.unsqueeze(2)
            self.cov_pred_byK_opt2_history = init_cov_full.unsqueeze(2)
            self.cov_post_byK_optA_history = init_cov_full.unsqueeze(2)
            self.cov_post_byK_optB_history = init_cov_full.unsqueeze(2)
            self.regularization_history = torch.zeros((self.num_of_bayesian_layers, 1), device=self.device).detach()

        # Připojování k historii (zůstává logicky stejné, ale s plnými maticemi)
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        
        # --- ZJEDNODUŠENÍ: Pracujeme s plnou kovarianční maticí ---
        init_cov_full_for_cat = self.system_model.P0.detach().clone().unsqueeze(2)
        self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, init_cov_full_for_cat), axis=2)
        self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, init_cov_full_for_cat), axis=2)
        self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, init_cov_full_for_cat), axis=2)
        self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, init_cov_full_for_cat), axis=2)
        self.regularization_history = torch.cat((self.regularization_history, self.regularization), axis=1)

    def filtering(self, observation):

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.system_model.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.system_model.h(x_predict)

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        residual = observation - y_predict
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain, regularization = self.kf_net(state_inno, residual, diff_state, diff_obs)

        self.regularization_history = torch.cat((self.regularization_history, torch.unsqueeze(regularization,dim=1).clone()), axis=1)

        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)