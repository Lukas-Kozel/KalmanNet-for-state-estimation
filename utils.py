import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from KalmanNet_withCovMatrix import KalmanNet_withCovMatrix

# Funkce pro generování dat
def generate_data(system, num_trajectories, seq_len):

    # Zjištění zařízení, na kterém systém pracuje
    device = system.Ex0.device

    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim,device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim,device=device)
    for i in range(num_trajectories):
        x = system.get_initial_state()
        # x = system.get_deterministic_initial_state()  # Použijeme fixní počáteční stav
        for t in range(seq_len):
            if t>0:
                x = system.step(x)            
            
            y = system.measure(x)
            x_data[i, t, :] = x.squeeze()
            y_data[i, t, :] = y.squeeze()
    return x_data, y_data

# Trénovací funkce
def train(model, train_loader,device, epochs=50, lr=1e-4, clip_grad=1.0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Zahajuji trénování KalmanNetu...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_norm = 0.0

        for x_true_batch, y_meas_batch in train_loader:

            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            x_hat_batch = model(y_meas_batch)
            
            loss = criterion(x_hat_batch, x_true_batch)

            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            print(f'Epocha [{epoch+1}/{epochs}], Prům. chyba: {avg_loss:.6f}, Celková norma grad.: {total_norm:.4f}')
    print("Trénování dokončeno.")

def train_with_scheduler(model, train_loader, val_loader, device, 
                         epochs=200, lr=1e-3, clip_grad=1.0, 
                         early_stopping_patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- Trénovací fáze ---
        model.train()
        train_loss = 0.0
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()

            x_hat_batch = model(y_meas_batch)

            if isinstance(model,KalmanNet_withCovMatrix):
                x_hat_batch = x_hat_batch.squeeze(0)

            loss = criterion(x_hat_batch, x_true_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validační fáze ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)


                x_hat_val = model(y_meas_val)
                if isinstance(model, KalmanNet_withCovMatrix):
                    x_hat_val = x_hat_val.squeeze(0)
                loss = criterion(x_hat_val, x_true_val)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epocha [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping spuštěno po {epoch + 1} epochách.")
            break
            
    print("Trénování dokončeno.")
    
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model


def gaussian_nll(target, predicted_mean, predicted_var):
    """Gaussian Negative Log Likelihood (assuming diagonal covariance)"""
    predicted_var += 1e-12
    mahal = torch.square(target - predicted_mean) / torch.abs(predicted_var)
    element_wise_nll = 0.5 * (torch.log(torch.abs(predicted_var)) + torch.log(torch.tensor(2 * torch.pi)) + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-2)
    return torch.mean(sample_wise_error)

def calculate_regularization_loss(model,c1,c2):
    reg_loss = 0.0
    p_float = model.dropout_prob
    eps = 1e-8

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param
            p = torch.tensor(p_float,device=param.device)
            n_input_features = weights.shape[1]

            entropy = -p * torch.log(p+eps) - (1 - p) * torch.log(1 - p+eps)

            l2_norm_squared = torch.sum(weights ** 2)
            term1 = c1*l2_norm_squared/(1-p)

            term2 = c2*entropy *n_input_features

            reg_loss += term1
            reg_loss += term2
    return reg_loss

def train_bayesian_kalmanNet(model, train_loader, val_loader, device, epochs=200, lr=1e-3, clip_grad=1, early_stopping_patience=20, num_samples_train=5):
    """
    Trénovací funkce pro BayesianKalmanNet, která používá vestavěnou 
    PyTorch nn.GaussianNLLLoss pro optimální kalibraci nejistoty.
    """
    criterion = nn.GaussianNLLLoss(full=True,eps=1e-6,reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=10,verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train() # kvuli aktivnimu dropoutu
        train_loss=0.0

        for x_true_batch,y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()

            ensemble_runs = []
            for _ in range(num_samples_train):
                one_run = model(y_meas_batch)
                ensemble_runs.append(one_run)
            ensemble_x_hat = torch.stack(ensemble_runs, dim=0)

            x_hat_mean = ensemble_x_hat.mean(dim=0)
            variance_floor = 1e-8 # Malá, ale nenulová minimální variance
            x_hat_var = ensemble_x_hat.var(dim=0).clamp(min=variance_floor)
            print("variance: ", x_hat_var)
            loss = criterion(x_hat_mean, x_true_batch, x_hat_var)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validační fáze ---
        # model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)

                ensemble_runs_val = [model(y_meas_val) for _ in range(num_samples_train)]
                ensemble_x_hat_val = torch.stack(ensemble_runs_val, dim=0)
                
                x_hat_mean_val = ensemble_x_hat_val.mean(dim=0)
                x_hat_var_val = ensemble_x_hat_val.var(dim=0)
                
                loss = criterion(input=x_hat_mean_val, target=x_true_val, var=x_hat_var_val)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epocha [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            model.eval()
            best_model_state = deepcopy(model.state_dict())
            model.train() 
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping spuštěno po {epoch + 1} epochách.")
            break

    print("Trénování dokončeno.")

    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)

    model.eval()
    return model

def train_bkn_article(model, train_loader, val_loader, device, epochs=200, lr=1e-4, clip_grad=1.0, early_stopping_patience=20):
    """
    Finální a kompletní verze trénovací funkce pro BayesianKalmanNet.
    Implementuje loss funkci z článku s použitím .detach() pro stabilní trénink
    a správnou kalibraci nejistoty.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Hyperparametry, které by měly fungovat lépe s .detach()
    J_samples = 5
    beta = 0.7  # S .detach() by mělo fungovat i vyvážené beta
    c1 = 1e-8
    c2 = 1e-8

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train() # KLÍČOVÉ: Aktivní dropout pro Monte Carlo sampling
        train_loss = 0.0
        
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            
            optimizer.zero_grad()
            
            # --- Získání ensemblu J predikcí ---
            ensemble_x_hat = torch.stack([model(y_meas_batch) for _ in range(J_samples)], dim=0)
            
            # Průměrný odhad
            x_hat_mean = ensemble_x_hat.mean(dim=0)

            # Odhad kovariance
            diff = ensemble_x_hat - x_hat_mean
            diff_permuted = diff.permute(1, 2, 0, 3) # [batch, seq, J, state_dim]
            Sigma_hat = torch.matmul(diff_permuted.unsqueeze(-1), diff_permuted.unsqueeze(-2)).mean(dim=2)

            # --- VÝPOČET KOMPONENT LOSS FUNKCE ---
            
            # Člen L_l2(θ) - počítá se s připojeným tenzorem, aby se model učil přesnosti
            loss_l2 = F.mse_loss(x_hat_mean, x_true_batch)

            # --- KLÍČOVÁ ZMĚNA S .detach() ---
            # Člen L_M2(θ) - pro výpočet cíle (empirical_cov) použijeme odpojený tenzor
            # Tím zabráníme tomu, aby snaha o minimalizaci loss_l2 ovlivňovala cíl pro loss_m2.
            e_for_m2 = (x_true_batch - x_hat_mean.detach()).unsqueeze(-1)
            flat_e_for_m2 = e_for_m2.flatten(0, 1)
            empirical_cov_flat = torch.bmm(flat_e_for_m2, flat_e_for_m2.transpose(1, 2))
            empirical_cov = empirical_cov_flat.reshape_as(Sigma_hat)
            
            loss_m2 = F.mse_loss(Sigma_hat, empirical_cov)
            
            # Celková data-matching loss
            data_matching_loss = (1 - beta) * loss_l2 + beta * loss_m2

            # Regularizační člen
            regularization_loss = calculate_regularization_loss(model, c1, c2)

            # Celková loss
            total_loss = data_matching_loss + regularization_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validační fáze (se stejnou logikou) ---
        model.train() # I zde necháme aktivní dropout pro konzistentní výpočet
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                ensemble_x_hat_val = torch.stack([model(y_meas_val) for _ in range(J_samples)], dim=0)
                
                x_hat_mean_val = ensemble_x_hat_val.mean(dim=0)
                diff_val = ensemble_x_hat_val - x_hat_mean_val
                diff_permuted_val = diff_val.permute(1, 2, 0, 3)
                Sigma_hat_val = torch.matmul(diff_permuted_val.unsqueeze(-1), diff_permuted_val.unsqueeze(-2)).mean(dim=2)
                
                loss_l2_val = F.mse_loss(x_hat_mean_val, x_true_val)

                # Zde není .detach() technicky nutné, protože nepočítáme gradienty,
                # ale pro 100% shodu logiky s tréninkem ho můžeme použít.
                e_val = (x_true_val - x_hat_mean_val.detach()).unsqueeze(-1)
                flat_e_val = e_val.flatten(0, 1)
                emp_cov_flat_val = torch.bmm(flat_e_val, flat_e_val.transpose(1, 2))
                emp_cov_val = emp_cov_flat_val.reshape_as(Sigma_hat_val)
                loss_m2_val = F.mse_loss(Sigma_hat_val, emp_cov_val)
                
                # Pro validační loss se obvykle regularizace nezahrnuje
                total_val_loss_sample = (1 - beta) * loss_l2_val + beta * loss_m2_val
                val_loss += total_val_loss_sample.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
        # Early stopping a uložení nejlepšího modelu
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping spuštěno po {epoch + 1} epochách.")
            break

    print("Trénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    
    model.eval() # Po dokončení tréninku přepneme model do evaluačního módu
    return model