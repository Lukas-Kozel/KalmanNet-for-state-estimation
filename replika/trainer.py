from typing import Union
import torch
from dnn import DNN_KalmanNet_GSS
from filtering import KalmanNet_Filter
import os
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

if not os.path.exists('./.model_saved'):
    os.mkdir('./.model_saved')

print_num = 25
save_num = int(config['Train']['valid_period'])

def mse(target, predicted):
    """Mean Squared Error"""
    return torch.mean(torch.square(target - predicted))

def empirical_averaging(target, predicted_mean, predicted_var, beta):
    L1 = mse(target, predicted_mean)
    L2 = torch.sum(torch.abs((target - predicted_mean)**2 - predicted_var))
    return (1-beta)*L1 + beta * L2

wd_split = float(config['Train.Split']['weight_decay'])
wd_kalman = float(config['Train.Kalman']['weight_decay'])

def gaussian_nll(target, predicted_mean, predicted_var):
    """Gaussian Negative Log Likelihood (assuming diagonal covariance)"""
    predicted_var += 1e-12
    mahal = torch.square(target - predicted_mean) / torch.abs(predicted_var)
    element_wise_nll = 0.5 * (torch.log(torch.abs(predicted_var)) + torch.log(torch.tensor(2 * torch.pi)) + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-2)
    return torch.mean(sample_wise_error)

class Trainer():

    def __init__(self, 
                    dnn:Union[KalmanNet_Filter], 
                    data_path, save_path, mode=0, isBNN:bool=False):
        config.read('./config.ini')

        lr_kalman = float(config['Train.Kalman']['learning_rate'])

        self.save_num = save_num

        self.dnn = dnn
        self.x_dim = self.dnn.x_dim
        self.y_dim = self.dnn.y_dim
        self.data_path = data_path
        self.save_path = save_path
        self.mode = mode

        self.loss_best = 1e4
        self.beta = float(config['Train']['beta'])

        self.data_x = torch.load(data_path + 'state.pt')
        self.data_x = self.data_x[:,0:(self.x_dim),:]
        self.data_y = torch.load(data_path + 'obs.pt')
        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]
        assert(self.x_dim == self.data_x.shape[1])
        assert(self.y_dim == self.data_y.shape[1])
        assert(self.seq_len == self.data_y.shape[2])
        assert(self.data_num == self.data_y.shape[0])

        if self.mode == 0:
            if isinstance(self.dnn, KalmanNet_Filter):
                self.loss_fn = torch.nn.MSELoss()
        
        if self.mode == 0:
            self.optimizer = torch.optim.Adam(self.dnn.kf_net.parameters(), lr=lr_kalman, weight_decay=wd_kalman)

        cal_num_param = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(cal_num_param(self.dnn.kf_net))

        self.batch_size = int(config['Train']['batch_size'])
        self.alter_num = int(config['Train.Split']['alter_period'])

        self.train_count = 0
        self.data_idx = 0


        self.num_of_mc_iterations = int(config['Bayesian']['num_of_mc_iterations'])
        self.num_of_bayesian_layers = int(config['Bayesian']['num_of_bayesian_layers'])

    def train_batch(self):
        if self.mode == 0: # KalmanNet
            self.train_batch_joint()


    def train_batch_joint(self):

        self.optimizer.zero_grad()

        if self.data_idx + self.batch_size >= self.data_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(self.data_x.shape[0])
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]
        batch_x = self.data_x[self.data_idx : self.data_idx+self.batch_size]
        batch_y = self.data_y[self.data_idx : self.data_idx+self.batch_size]

        x_hat = torch.zeros_like(batch_x)
        cov_hat = torch.zeros_like(batch_x)
        cov_hat_bnn = torch.zeros_like(batch_x)
            # torch.autograd.set_detect_anomaly(True)
        regularization = torch.zeros((self.batch_size, self.num_of_bayesian_layers, self.seq_len))
        x_hat_temp = torch.zeros((batch_x.shape + (self.num_of_mc_iterations,)))
        regularization_temp = torch.zeros(((batch_x.shape[0],self.num_of_bayesian_layers,batch_x.shape[2]) + (self.num_of_mc_iterations,)))
        for i in range(self.batch_size):
            for mc_ind in range(self.num_of_mc_iterations):
                self.dnn.state_post = batch_x[i,:,0].reshape((-1,1))
                for ii in range(1, self.seq_len):
                    self.dnn.filtering(batch_y[i,:,ii].reshape((-1,1)))
                x_hat_temp[i,:,:,mc_ind] = self.dnn.state_history[:,-self.seq_len:]
                regularization_temp[i,:,:,mc_ind] = self.dnn.regularization_history[:,-self.seq_len:]

            x_hat[i] = torch.mean(x_hat_temp[i].clone(), dim=-1)
            cov_hat_bnn[i] = torch.var(x_hat_temp[i].clone(), dim=-1)
            regularization[i] = torch.mean(regularization_temp[i], dim=-1)
            self.dnn.reset(clean_history=False)
        
        avg_cov = torch.mean(cov_hat_bnn)
        avg_hat = torch.mean(torch.abs(x_hat))
        print(f'avg cov: {avg_cov}, avg hat: {avg_hat}')

        # Loss Functions: MSE / GNLL / Empirical Averaging
        BETA = self.beta * (self.train_count / int(config['Train']['train_iter']))
        loss = \
            + (1-BETA) * torch.mean( torch.square( batch_x[:, :, 1:]-x_hat[:, :, 1:] ) ) \
            + (BETA) * torch.sum(torch.abs((batch_x[:, :, 1:]-x_hat[:, :, 1:])**2 - cov_hat_bnn[:, :, 1:] )) \
            + torch.sum(regularization[:, :, 1:])
        print(f'L1: {torch.mean( torch.square( batch_x[:, :, 1:]-x_hat[:, :, 1:] ))}, L2: {torch.sum(torch.abs((batch_x[:, :, 1:]-x_hat[:, :, 1:])**2 - cov_hat_bnn[:, :, 1:] ))}, Reg: {torch.sum(regularization[:, :, 1:])}')
        print(f'Loss: {loss}')


        loss.backward()

        ## gradient clipping with maximum value 10
        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)

        self.optimizer.step()

        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            try:
                torch.save(self.dnn.kf_net, './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt')
            except:
                print('here')
                pass
        if self.train_count % print_num == 1:
            print(f'[Model {self.save_path}] [Train {self.train_count}] loss [dB] = {10*torch.log10(loss):.4f}')
            print(f'   ~[P1 {torch.sigmoid(self.dnn.kf_net.l1_conc.p_logit)}]')
            print(f'   ~[P2 {torch.sigmoid(self.dnn.kf_net.l2_conc.p_logit)}]')

    def validate(self, tester):            
        if tester.loss.item() < self.loss_best:
            try:
                torch.save(tester.filter.kf_net, './.model_saved/' + self.save_path[:-3] + '_best.pt')
                print(f'Save best model at {self.save_path} & train {self.train_count} & loss [dB] = {tester.loss:.4f}')        
                self.loss_best = tester.loss.item()    
            except:
                pass            
        self.valid_loss = tester.loss.item()