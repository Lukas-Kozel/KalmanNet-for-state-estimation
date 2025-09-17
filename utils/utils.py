from xml.parsers.expat import model
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

from NN_models.KalmanNet_withCovMatrix import KalmanNet_withCovMatrix

from state_NN_models.StateKalmanNetWithKnownR import StateKalmanNetWithKnownR


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

            l2_norm_squared = torch.norm(weights, p=2)**2
            term1 = c1*l2_norm_squared/(1-p + eps)

            term2 = c2*entropy *n_input_features

            reg_loss += term1
            reg_loss += term2
    return reg_loss


def train_bkn(model, train_loader, val_loader, device, 
              epochs=200, lr=1e-4, clip_grad=1.0, early_stopping_patience=20, 
              J_samples=10, 
              initial_beta=0.01, final_beta=0.9, beta_warmup_epochs=50,
              c1=1e-8, c2=1e-8):
    """
    Funkce pro trénování BayesianKalmanNet.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):

        if epoch < beta_warmup_epochs:
            # Lineární nárůst od initial_beta k final_beta
            beta = initial_beta + (final_beta - initial_beta) * (epoch / beta_warmup_epochs)
        else:
            # Po warm-up fázi zůstane beta na finální hodnotě
            beta = final_beta

        model.train() # dropout musí být aktivní
        train_loss = 0.0
        
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            
            optimizer.zero_grad()
            
            # --- Získání ensemblu J predikcí ---
            # Tvar: [J_samples, batch_size, seq_len, state_dim]
            ensemble_x_hat = torch.stack([model(y_meas_batch) for _ in range(J_samples)], dim=0)
            
            num_samples, batch_size, seq_len, state_dim = ensemble_x_hat.shape
            
            total_loss_for_batch = 0.0

            # --- SMYČKA PŘES JEDNOTLIVÉ TRAJEKTORIE V DÁVCE ---
            for i in range(batch_size):
                ensemble_for_one_trajectory = ensemble_x_hat[:, i, :, :]
                true_trajectory = x_true_batch[i, :, :]
                
                # --- SMYČKA PŘES JEDNOTLIVÉ ČASOVÉ KROKY ---
                for t in range(seq_len):
                    ensemble_at_t = ensemble_for_one_trajectory[:, t, :]
                    true_state_at_t = true_trajectory[t, :]
                    
                    mean_hat_at_t = torch.mean(ensemble_at_t, dim=0)
                    
                    sigma_hat_at_t = torch.zeros((state_dim, state_dim), device=device)

                    # --- SMYČKA J realizací ---
                    for j in range(num_samples):
                        diff = ensemble_at_t[j] - mean_hat_at_t
                        sigma_hat_at_t += torch.outer(diff, diff)
                    sigma_hat_at_t /= num_samples

                    loss_l2_at_t = F.mse_loss(mean_hat_at_t, true_state_at_t)
                    
                    error_detached = true_state_at_t - mean_hat_at_t.detach()
                    empirical_cov_at_t = torch.outer(error_detached, error_detached)
                    loss_m2_at_t = F.mse_loss(sigma_hat_at_t, empirical_cov_at_t)
                    
                    data_loss_at_t = (1 - beta) * loss_l2_at_t + beta * loss_m2_at_t
                    total_loss_for_batch += data_loss_at_t
            
            avg_data_loss = total_loss_for_batch / (batch_size * seq_len)
            
            regularization_loss = calculate_regularization_loss(model, c1, c2)
            total_loss = avg_data_loss + regularization_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validační fáze  ---
        model.train() # nutné držet aktivní dropout
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                ensemble_x_hat_val = torch.stack([model(y_meas_val) for _ in range(J_samples)], dim=0)
                
                num_samples_val, batch_size_val, seq_len_val, state_dim_val = ensemble_x_hat_val.shape
                
                total_val_loss_for_batch = 0.0
                
                for i in range(batch_size_val):
                    ensemble_traj_val = ensemble_x_hat_val[:, i, :, :]
                    true_traj_val = x_true_val[i, :, :]
                    
                    for t in range(seq_len_val):
                        ensemble_t_val = ensemble_traj_val[:, t, :]
                        true_t_val = true_traj_val[t, :]
                        
                        mean_hat_t_val = torch.mean(ensemble_t_val, dim=0)
                        
                        sigma_hat_t_val = torch.zeros((state_dim_val, state_dim_val), device=device)
                        for j in range(num_samples_val):
                            diff_val = ensemble_t_val[j] - mean_hat_t_val
                            sigma_hat_t_val += torch.outer(diff_val, diff_val)
                        sigma_hat_t_val /= num_samples_val
                        
                        loss_l2_val = F.mse_loss(mean_hat_t_val, true_t_val)
                        
                        error_val_detached = true_t_val - mean_hat_t_val.detach()
                        emp_cov_val = torch.outer(error_val_detached, error_val_detached)
                        loss_m2_val = F.mse_loss(sigma_hat_t_val, emp_cov_val)
                        
                        # Pro validační loss se obvykle regularizace nezahrnuje
                        data_loss_val = (1 - beta) * loss_l2_val + beta * loss_m2_val
                        total_val_loss_for_batch += data_loss_val

                avg_val_loss_for_batch = total_val_loss_for_batch / (batch_size_val * seq_len_val)
                val_loss += avg_val_loss_for_batch.item()

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

def train_bkn_optimized(model, train_loader, val_loader, device, 
                        epochs=200, lr=1e-4, clip_grad=1.0, early_stopping_patience=20, 
                        J_samples=10, 
                        initial_beta=0.01, final_beta=0.9, beta_warm_up_epochs=50,
                        c1=1e-8, c2=1e-8):
    """
    Optimalizovaná verze trénovací funkce, která využívá tenzorové operace
    pro rychlost, ale vyhýbá se matoucím funkcím jako permute nebo bmm.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # --- Beta Scheduling ---
        if epoch < beta_warm_up_epochs:
            beta = initial_beta + (final_beta - initial_beta) * (epoch / beta_warm_up_epochs)
        else:
            beta = final_beta

        model.train()
        train_loss = 0.0
        
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            
            optimizer.zero_grad()
            
            # Tvar: [J_samples, batch_size, seq_len, state_dim]
            ensemble_x_hat = torch.stack([model(y_meas_batch) for _ in range(J_samples)], dim=0)
            
            x_hat_mean = torch.mean(ensemble_x_hat, dim=0)
            
            # Tvar: [J_samples, batch_size, seq_len, state_dim]
            diff = ensemble_x_hat - x_hat_mean
            
            # Tvar: [J_samples, batch_size, seq_len, state_dim, 1]
            diff_col = diff.unsqueeze(-1)
            # Tvar: [J_samples, batch_size, seq_len, 1, state_dim]
            diff_row = diff.unsqueeze(-2)
            
            # Broadcasting `(J, B, S, D, 1) * (J, B, S, 1, D)` dá `(J, B, S, D, D)`
            outer_product_ensemble = diff_col * diff_row
            
            # Tvar: [batch_size, seq_len, state_dim, state_dim]
            Sigma_hat = torch.mean(outer_product_ensemble, dim=0)

            # L_l2: MSE na průměrném odhadu
            loss_l2 = F.mse_loss(x_hat_mean, x_true_batch)

            # L_M2: MSE na kovarianci
            error_detached = x_true_batch - x_hat_mean.detach()

            error_col = error_detached.unsqueeze(-1)
            error_row = error_detached.unsqueeze(-2)
            empirical_cov = error_col * error_row
            
            loss_m2 = F.mse_loss(Sigma_hat, empirical_cov)
            
            # Celková datová loss
            data_matching_loss = (1 - beta) * loss_l2 + beta * loss_m2
            
            # Přičtení regularizace
            regularization_loss = calculate_regularization_loss(model, c1, c2)
            total_loss = data_matching_loss + regularization_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validační fáze  ---
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                ensemble_x_hat_val = torch.stack([model(y_meas_val) for _ in range(J_samples)], dim=0)
                
                x_hat_mean_val = torch.mean(ensemble_x_hat_val, dim=0)

                diff_val = ensemble_x_hat_val - x_hat_mean_val
                outer_product_val = diff_val.unsqueeze(-1) * diff_val.unsqueeze(-2)
                Sigma_hat_val = torch.mean(outer_product_val, dim=0)

                loss_l2_val = F.mse_loss(x_hat_mean_val, x_true_val)

                error_val_detached = x_true_val - x_hat_mean_val.detach()
                emp_cov_val = error_val_detached.unsqueeze(-1) * error_val_detached.unsqueeze(-2)
                loss_m2_val = F.mse_loss(Sigma_hat_val, emp_cov_val)
                
                total_val_loss_sample = (1 - beta) * loss_l2_val + beta * loss_m2_val
                val_loss += total_val_loss_sample.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Current Beta: {beta:.4f}')
            
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
    
    model.eval()
    return model


def train_bkn_with_logging(model, train_loader, val_loader, device, 
                           epochs=200, lr=1e-4, clip_grad=1.0, early_stopping_patience=20, 
                           J_samples=10, num_val_data_samples=10, 
                           initial_beta=0.01, final_beta=0.9, beta_warmup_epochs=50,
                           c1=1e-8, c2=1e-8):
    """
    Verze trénovací funkce s detailním logováním jednotlivých komponent loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # linearni interpolace parametru beta => model se nejdriv zameri na minimalizaci chyby odhadu a pak kovariance
    for epoch in range(epochs):
        if epoch < beta_warmup_epochs:
            beta = initial_beta + (final_beta - initial_beta) * (epoch / beta_warmup_epochs)
        else:
            beta = final_beta

        model.train() # dulezite kvuli dropout vrstvam

        epoch_total_losses = []
        epoch_l2_losses = []
        epoch_m2_losses = []
        epoch_reg_losses = []
        
        # zpracovani mini-batche
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            
            optimizer.zero_grad()
            
            # predikce stavu a cov matice z modelu
            x_hat_mean, Sigma_hat = model(y_meas_batch, num_samples=J_samples)

            # Vypocet L2 ztraty
            loss_l2 = F.mse_loss(x_hat_mean, x_true_batch)

            # vypocet chyby odhadu s pouzitim .detach() pro spravnou funkci backpropagation
            e_for_m2 = (x_true_batch - x_hat_mean.detach()) #[batch, seq, state_dim]

            # ve formátu vektoru (není to matice, protože mi stačí jen diagonální prvky)
            empirical_variances = e_for_m2 ** 2  #[batch, seq, state_dim]
            
            # vypocet diagonálních prvku matice cov matice a jejich přeuspořádání do vektoru
            predicted_variances = torch.diagonal(Sigma_hat, dim1=-2, dim2=-1) #[batch, seq, state_dim, state_dim]

            #loss_m2 porovnává dva VEKTORY o tvaru [batch, seq, state_dim]
            loss_m2 = F.l1_loss(predicted_variances, empirical_variances)
            
            # rovnice pro celkovou datovou ztrátu
            data_matching_loss = (1 - beta) * loss_l2 + beta * loss_m2
            
            # vypocet regularizace
            regularization_loss = calculate_regularization_loss(model, c1, c2)

            # celková ztráta
            total_loss = data_matching_loss + regularization_loss

            # zpětná propagace
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            epoch_total_losses.append(total_loss.item())
            epoch_l2_losses.append(loss_l2.item())
            epoch_m2_losses.append(loss_m2.item())
            epoch_reg_losses.append(regularization_loss.item())
            
        avg_train_loss = np.mean(epoch_total_losses)
        avg_l2_loss = np.mean(epoch_l2_losses)
        avg_m2_loss = np.mean(epoch_m2_losses)
        avg_reg_loss = np.mean(epoch_reg_losses)
        
        # --- Validační fáze ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                model.train()
                x_hat_mean_val, _ = model(y_meas_val, num_samples=num_val_data_samples)
                model.eval()

                # pro validaci by melo stacit jen MSE, protože cílem je mít co nejpřesnější odhad
                val_loss += F.mse_loss(x_hat_mean_val, x_true_val).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"--- Epocha [{epoch+1}/{epochs}] ---")
            print(f"  Val Loss (MSE): {avg_val_loss:.6f} | Current Beta: {beta:.4f}")
            print(f"  Train Loss:     {avg_train_loss:.6f}")
            print(f"    ├─ L2 (MSE):   {avg_l2_loss:.6f} (váha: {1-beta:.2f})")
            print(f"    ├─ M2 (L1 Var):{avg_m2_loss:.6f} (váha: {beta:.2f})")
            print(f"    └─ Regularize: {avg_reg_loss:.6f}")

        # early stopping při dlouhodobém zhoršení validační ztráty
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
    
    model.eval()
    return model

def train_state_KalmanNet(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Zjistíme si předem, zda model vrací i kovarianci, abychom mohli upravit logiku
    returns_covariance = isinstance(model, StateKalmanNetWithKnownR)

    for epoch in range(epochs):
        # --- Trénovací fáze ---
        model.train()
        train_loss = 0.0
        
        # Seznam pro sběr průměrných stop kovariancí z každé dávky
        epoch_traces = []

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape

            # restart vnitřního stavu filtru pro novou dávku
            initial_state_batch = x_true_batch[:, 0, :]
            model.reset(batch_size=batch_size, initial_state=initial_state_batch)

            predictions_x = []
            predictions_P = []

            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                
                step_output = model.step(y_t)
                if returns_covariance:
                    # Pokud step vrací (x, P), rozbalíme tuple
                    x_filtered_t, P_filtered_t = step_output
                    predictions_P.append(P_filtered_t)
                else:
                    # Jinak je výstup přímo x
                    x_filtered_t = step_output

                predictions_x.append(x_filtered_t)

            predicted_trajectory = torch.stack(predictions_x, dim=1)

            # Uložení stopy pro logování
            if returns_covariance and predictions_P:
                predicted_cov_trajectory = torch.stack(predictions_P, dim=1)
                # Spočítáme stopu pro každou matici v dávce a zprůměrujeme
                # vmap zajistí efektivní operaci přes batch a seq_len dimenze
                avg_trace_batch = torch.mean(torch.vmap(torch.trace)(predicted_cov_trajectory.flatten(0, 1))).item()
                epoch_traces.append(avg_trace_batch)

            loss = criterion(predicted_trajectory, x_true_batch[:, 1:, :])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_epoch_trace = np.mean(epoch_traces) if epoch_traces else 0.0

         # --- Validační fáze ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                batch_size_val, seq_len_val, _ = x_true_val.shape
                
                initial_state_val = x_true_val[:, 0, :]
                model.reset(batch_size=batch_size_val, initial_state=initial_state_val)
                
                val_predictions = []
                for t in range(1, seq_len_val):
                    y_t_val = y_meas_val[:, t, :]

                    step_output_val = model.step(y_t_val)
                    if returns_covariance:
                        x_filtered_t_val = step_output_val[0]
                    else:   
                        x_filtered_t_val = step_output_val

                    val_predictions.append(x_filtered_t_val)
                    
                predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                val_loss_batch = criterion(predicted_val_trajectory, x_true_val[:, 1:, :])
                epoch_val_loss += val_loss_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            if returns_covariance:
                log_message += f', Avg Cov Trace: {avg_epoch_trace:.6f}'
            print(log_message)
        
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