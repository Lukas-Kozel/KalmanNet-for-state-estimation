import torch


class UnscentedKalmanFilter:
    def __init__(self, system_model):
   
        self.device = system_model.Q.device
        
        self.f = system_model.f
        self.h = system_model.h
        
 
        self.Q = system_model.Q.clone().detach().to(self.device)
        self.R = system_model.R.clone().detach().to(self.device)
        self.Ex0 = system_model.Ex0.clone().detach().to(self.device)
        self.P0 = system_model.P0.clone().detach().to(self.device)
        
        self.state_dim = self.Q.shape[0]
        self.obs_dim = self.R.shape[0]

        
    def _unscented_transform(self,x,P):
        P_stable = P #+ torch.eye(self.state_dim, device=self.device) * 1e-9
        S = torch.linalg.cholesky(P_stable)
        scale = torch.sqrt(torch.tensor(self.state_dim, device=self.device,dtype=torch.float32))

        sigma_points = x.repeat(1,2*self.state_dim+1)
        sigma_points[:,1:self.state_dim+1] += scale * S
        sigma_points[:,self.state_dim+1:] -= scale * S
        
        weights = torch.full((2 * self.state_dim + 1,), 1.0 / (2 * self.state_dim + 1)).to(self.device)
        return sigma_points, weights
    
    def predict_step(self, x_filtered, P_filtered):
        sigma_points, weights = self._unscented_transform(x_filtered, P_filtered)
        propagated_points = self.f(sigma_points)
   
        x_predict = (propagated_points @ weights).unsqueeze(1)
        diff = propagated_points - x_predict
        P_predict = (diff * weights) @ diff.T + self.Q

        return x_predict, P_predict

    def update_step(self, x_predict,P_predict, y_t):
        sigma_points, weights = self._unscented_transform(x_predict, P_predict)
        measurement_points = torch.cat([self.h(pt.unsqueeze(1)) for pt in sigma_points.T], dim=1)
        y_hat = (measurement_points * weights).sum(dim=1).unsqueeze(1)
        
        y_diff = measurement_points - y_hat
        x_diff = sigma_points - x_predict
        
        P_yy = (y_diff * weights) @ y_diff.T + self.R
        P_xy = (x_diff * weights) @ y_diff.T
        
        innovation = y_t - y_hat
        K = P_xy @ torch.linalg.inv(P_yy)
        
        x_filtered = x_predict + K @ innovation
        P_filtered = P_predict - K @ P_yy @ K.T
        
        return x_filtered, P_filtered, innovation



    def apply_filter(self, y_seq):
        seq_len = y_seq.shape[0]
        x_est_list = []
        
        x_est = self.Ex0.clone()
        P_est = self.P0.clone()

        for t in range(seq_len):
            x_predict, P_predict = self.predict_step(x_est, P_est)
            y_t = y_seq[t].unsqueeze(-1)
            x_est, P_est, _ = self.update_step(x_predict, P_predict, y_t)
            x_est_list.append(x_est.squeeze())

        return {'x_filtered': torch.stack(x_est_list)}