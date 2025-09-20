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

def store_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model byl uložen do {path}")

# Trénovací funkce
def train(model, train_loader,device, epochs=50, lr=1e-4, clip_grad=10):
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
                         epochs=200, lr=1e-3, clip_grad=10, 
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
              epochs=200, lr=1e-4, clip_grad=10, early_stopping_patience=20, 
              J_samples=10, 
              final_beta=0.5):
    """
    Funkce pro trénování BayesianKalmanNet.
    """
    optimizer = torch.optim.Adam([
    {'params': model.input_layer.parameters()},
    {'params': model.gru.parameters()},
    {'params': model.output_layer.parameters()},
    {'params': model.concrete_dropout1.parameters()},
    {'params': model.concrete_dropout2.parameters()}
], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    total_train_iter = len(train_loader)*epochs
    train_count=0

    for epoch in range(epochs):

        model.train() # dropout musí být aktivní
        train_loss = 0.0
        
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            
            # --- Beta Scheduling ---
            beta = final_beta * (train_count / total_train_iter)
            train_count+=1

            optimizer.zero_grad()
            
            # --- Získání ensemblu J predikcí ---
            # Tvar: [J_samples, batch_size, seq_len, state_dim]
            x_hat_mean, sigma_hat, regularization_loss = model(y_meas_batch, num_samples=J_samples)
            
            predicted_variances = torch.diagonal(sigma_hat, dim1=-2, dim2=-1)

            data_loss = empirical_averaging(target=x_true_batch, 
                                                      predicted_mean=x_hat_mean, 
                                                      predicted_var=predicted_variances.detach(),
                                                      beta=beta)
            
            reg_multiplier = 1.0
            total_loss = data_loss + reg_multiplier * regularization_loss
            
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
                
                x_hat_mean_val, _, _ = model(y_meas_val, num_samples=5)
                val_loss += F.mse_loss(x_hat_mean_val, x_true_val).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            p1 = torch.sigmoid(model.concrete_dropout1.p_logit).item()
            p2 = torch.sigmoid(model.concrete_dropout2.p_logit).item()
            print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")
            print(f"  Regularization Loss: {regularization_loss.item():.6f}")
            
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
                        epochs=200, lr=1e-4, clip_grad=10, early_stopping_patience=20, 
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
                           epochs=200, lr=1e-4, clip_grad=10, early_stopping_patience=20, 
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
                    x_filtered_t, P_filtered_t = step_output
                    predictions_P.append(P_filtered_t)
                else:
                    x_filtered_t = step_output

                predictions_x.append(x_filtered_t)

            predicted_trajectory = torch.stack(predictions_x, dim=1)

            if returns_covariance and predictions_P:
                predicted_cov_trajectory = torch.stack(predictions_P, dim=1)
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

def gaussian_nll_robust(target, predicted_mean, predicted_var):
    variance_floor = 1e-6
    
    is_below_floor = predicted_var < variance_floor
    num_clamped = torch.sum(is_below_floor).item()
    
    clamped_var = torch.clamp(predicted_var, min=variance_floor)

    mahal = torch.square(target - predicted_mean) / clamped_var
    log_term = torch.log(clamped_var)
    pi_tensor = torch.tensor(2 * torch.pi, device=target.device)
    element_wise_nll = 0.5 * (log_term + torch.log(pi_tensor) + mahal)
    
    return torch.mean(element_wise_nll), num_clamped


def train_bkn_nll_final(model, train_loader, val_loader, device, 
                        epochs=200, lr=1e-4, clip_grad=10, early_stopping_patience=15):
    """
    Finální verze, která používá přesnou implementaci `gaussian_nll`
    Očekává, že model vrací (x_hat_mean, Sigma_hat).
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train() # Aktivuje dropout pro regularizaci a MC sampling
        train_loss = 0.0

        total_clamped_in_epoch = 0
        total_vars_in_epoch = 0 # Pro výpočet procenta

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            
            optimizer.zero_grad()
            
            # Předpokládáme, že model vrací (x_hat_mean, Sigma_hat)
            x_hat_mean, Sigma_hat = model(y_meas_batch, num_samples=5)
            
            predicted_variances = torch.diagonal(Sigma_hat, dim1=-2, dim2=-1)
            
            loss, num_clamped_batch = gaussian_nll_robust(
                target=x_true_batch, 
                predicted_mean=x_hat_mean, 
                predicted_var=predicted_variances
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += loss.item()
            total_clamped_in_epoch += num_clamped_batch
            total_vars_in_epoch += predicted_variances.numel()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validační fáze ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                model.train() # Dočasně zapneme dropout
                x_hat_val, sigma_hat_val = model(y_meas_val, num_samples=5)
                model.eval()

                predicted_variances_val = torch.diagonal(sigma_hat_val, dim1=-2, dim2=-1)
                
                loss,_ = gaussian_nll_robust(
                    target=x_true_val, 
                    predicted_mean=x_hat_val, 
                    predicted_var=predicted_variances_val
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        clamped_percentage = (total_clamped_in_epoch / total_vars_in_epoch) * 100 if total_vars_in_epoch > 0 else 0

        if (epoch + 1) % 5 == 0:
            print(f"--- Epocha [{epoch+1}/{epochs}] ---")
            print(f"  Val Loss (NLL): {avg_val_loss:.4f}")
            print(f"  Train Loss (NLL): {avg_train_loss:.4f}")
            print(f"  Clamped Variances: {clamped_percentage:.2f}% ({total_clamped_in_epoch}/{total_vars_in_epoch})")

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
        print(f"Načítám nejlepší model s validační chybou (NLL): {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)
        
    return model

def empirical_averaging(target, predicted_mean, predicted_var, beta):
    """
    Vrací celkovou datovou loss a její jednotlivé komponenty pro logování.
    """
    # L1 trénuje průměrný odhad.
    l1_loss = F.mse_loss(predicted_mean, target)
    
    L2 = torch.sum(torch.abs((target - predicted_mean)**2 - predicted_var))
    
    # Celková datová loss
    total_data_loss = (1 - beta) * l1_loss + beta * L2

    return total_data_loss, l1_loss, L2

def train_stateful_bkn(model, train_loader, val_loader, device, 
                       epochs=100, lr=1e-4, clip_grad=10.0,
                       J_samples=10, 
                       final_beta=0.01,
                       beta_warmup_epochs=None):
    """
    Finální trénovací funkce pro BKN, která správně inicializuje filtr a ořezává sekvence.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # best_val_loss = float('inf')
    # epochs_no_improve = 0
    best_model_state = None
    
    total_train_iter = len(train_loader) * epochs
    train_count = 0

    for epoch in range(epochs):
        model.train()
        
        epoch_total_losses, epoch_l1_losses, epoch_l2_losses, epoch_reg_losses = [], [], [], []
        all_predicted_vars = []
        all_empirical_sq_errors = []

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
            
            if beta_warmup_epochs is not None and beta_warmup_epochs > 0:
                beta = 0.0 if epoch < beta_warmup_epochs else final_beta
            else:
                beta = final_beta * (train_count / total_train_iter)
            train_count += 1

            optimizer.zero_grad()
            
            batch_size, seq_len, _ = x_true_batch.shape
            
            # --- ZMĚNA č. 1: Správná inicializace modelu ---
            initial_state_batch = x_true_batch[:, 0, :]
            model.reset(initial_state=initial_state_batch, batch_size=batch_size, num_samples=J_samples)
            
            x_hat_mean_list, sigma_hat_list, reg_list = [], [], []
            
            # --- ZMĚNA č. 2: Smyčka přes čas začíná od 1 ---
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_filtered_t, P_filtered_t, reg_t = model.step(y_t, num_samples=J_samples)
                x_hat_mean_list.append(x_filtered_t)
                sigma_hat_list.append(P_filtered_t)
                reg_list.append(reg_t)

            # --- ZMĚNA č. 3: Oříznutí dat pro výpočet loss ---
            x_hat_mean = torch.stack(x_hat_mean_list, dim=1)
            sigma_hat = torch.stack(sigma_hat_list, dim=1)
            x_true_batch_sliced = x_true_batch[:, 1:, :] # Ořízneme i skutečná data
            
            # Doporučuji průměr pro stabilitu, ale nechávám sum dle vaší preference
            all_regs_tensor = torch.stack(reg_list, dim=0)
            regularization_loss = torch.sum(all_regs_tensor)
            
            predicted_variances = torch.diagonal(sigma_hat, dim1=-2, dim2=-1)
            
            all_predicted_vars.append(predicted_variances.detach().cpu())
            empirical_squared_error = torch.square(x_true_batch_sliced - x_hat_mean)
            all_empirical_sq_errors.append(empirical_squared_error.detach().cpu())
            
            
            data_loss, l1_loss, l2_loss = empirical_averaging(
                target=x_true_batch_sliced, 
                predicted_mean=x_hat_mean, 
                predicted_var=predicted_variances,
                beta=beta
            )
            
            total_loss = data_loss + regularization_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            epoch_total_losses.append(total_loss.item())
            epoch_l1_losses.append(l1_loss.item())
            epoch_l2_losses.append(l2_loss.item())
            epoch_reg_losses.append(regularization_loss.item())

        avg_train_loss, avg_l1_loss, avg_l2_loss, avg_reg_loss = (np.mean(epoch_total_losses), 
                                                                  np.mean(epoch_l1_losses), 
                                                                  np.mean(epoch_l2_losses), 
                                                                  np.mean(epoch_reg_losses))
        
        all_vars_tensor = torch.cat(all_predicted_vars) if all_predicted_vars else torch.tensor([0.])
        avg_pred_var, min_pred_var, max_pred_var = (torch.mean(all_vars_tensor).item(), 
                                                    torch.min(all_vars_tensor).item(), 
                                                    torch.max(all_vars_tensor).item())

        all_sq_errors_tensor = torch.cat(all_empirical_sq_errors) if all_empirical_sq_errors else torch.tensor([0.])
        avg_true_var, min_true_var, max_true_var = (torch.mean(all_sq_errors_tensor).item(), 
                                                    torch.min(all_sq_errors_tensor).item(), 
                                                    torch.max(all_sq_errors_tensor).item())

        # --- Validační fáze se stejnou logikou ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                batch_size_val, seq_len_val, _ = x_true_val.shape
                
                # --- ZMĚNA č. 4 (VALIDACE) ---
                initial_state_val = x_true_val[:, 0, :]
                model.reset(batch_size=batch_size_val, num_samples=5, initial_state=initial_state_val)
                
                val_predictions = []
                for t in range(1, seq_len_val): # Smyčka také od 1
                    y_t_val = y_meas_val[:, t, :]
                    x_filtered_t_val, _, _ = model.step(y_t_val, num_samples=5)
                    val_predictions.append(x_filtered_t_val)
                
                predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                # Porovnáváme s oříznutými skutečnými daty
                val_loss += F.mse_loss(predicted_val_trajectory, x_true_val[:, 1:, :]).item()

        avg_val_loss = val_loss / len(val_loader)
        # scheduler.step(avg_val_loss)

        # Logování a Early Stopping zůstávají stejné...
        if (epoch + 1) % 5 == 0:
            print(f"--- Epocha [{epoch+1}/{epochs}] ---")
            print(f"  Val Loss (MSE): {avg_val_loss:.6f} | Current Beta: {beta:.4f}")
            print(f"  Train Loss:     {avg_train_loss:.6f}")
            print(f"    ├─ L1 (MSE):   {avg_l1_loss:.6f} (váha: {1-beta:.2f})")
            print(f"    ├─ L2 (Var L1):{avg_l2_loss:.6f} (váha: {beta:.4f})")
            print(f"    └─ Regularize: {avg_reg_loss:.6f}")
            print(f"  Predicted Var Stats: Avg={avg_pred_var:.6f}, Min={min_pred_var:.6f}, Max={max_pred_var:.6f}")
            print(f"  True Var Stats     : Avg={avg_true_var:.6f}, Min={min_true_var:.6f}, Max={max_true_var:.6f}")
            if hasattr(model, 'dnn') and hasattr(model.dnn, 'concrete_dropout1'):
                p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")
            print("-" * (len(f"--- Epocha [{epoch+1}/{epochs}] ---")))
        
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     best_model_state = deepcopy(model.state_dict())
        # else:
        #     epochs_no_improve += 1

        # if epochs_no_improve >= early_stopping_patience:
        #     print(f"\nEarly stopping spuštěno po {epoch + 1} epochách.")
        #     break
    best_model_state = deepcopy(model.state_dict())
    print("Trénování dokončeno.")
    if best_model_state:
        # print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    
    model.eval()
    return model

def train_stateful_bkn_authors_replication(
    model, train_loader, val_loader, device, 
    # --- ZMĚNA č. 1: Hlavním řídícím prvkem jsou iterace, ne epochy ---
    total_train_iter=10000, # Celkový počet kroků optimalizátoru (odpovídá 'train_iter' v config.ini)
    lr=1e-4, 
    clip_grad=10.0, 
    J_samples=20,          # Počet MC vzorků dle config.ini
    final_beta=0.01,       # Finální hodnota beta dle config.ini
    # --- ZMĚNA č. 2: Odstranili jsme parametry, které nebudeme používat ---
    # early_stopping_patience a beta_warmup_epochs se nepoužívají.
    
    # --- NOVINKA: Parametry pro validaci a logování ---
    validation_period=100, # Jak často provádět validaci (v iteracích)
    logging_period=25      # Jak často logovat tréninkové statistiky
):
    """
    Trénovací funkce upravená tak, aby co nejvěrněji replikovala proces autorů.
    - Řízeno pevným počtem iterací, ne epochami.
    - Používá `torch.sum` v L2 a malou `final_beta`.
    - Lineární `beta` scheduling bez warmupu.
    - Žádný early stopping ani LR scheduler.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # --- ZMĚNA č. 3: Logika ukládání modelu je zjednodušená ---
    # Uložíme model na konci, případně nejlepší model podle validační MSE
    best_val_loss = float('inf')
    best_model_state = None

    train_iter_count = 0
    
    # --- ZMĚNA č. 4: Hlavní smyčka běží, dokud nedosáhneme `total_train_iter` ---
    done = False
    while not done:
        model.train()
        
        # Tato vnitřní smyčka zajistí, že projedeme data, i když dojdou
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
            
            # --- ZMĚNA č. 5: Beta scheduling je přesně podle kódu autorů ---
            beta = final_beta * (train_iter_count / total_train_iter)
            
            optimizer.zero_grad()
            
            batch_size, seq_len, _ = x_true_batch.shape
            
            # Inicializace a smyčka přes sekvenci zůstávají stejné
            initial_state_batch = x_true_batch[:, 0, :]
            model.reset(initial_state=initial_state_batch, batch_size=batch_size, num_samples=J_samples)
            
            x_hat_mean_list, sigma_hat_list, reg_list = [], [], []
            
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_filtered_t, P_filtered_t, reg_t = model.step(y_t, num_samples=J_samples)
                x_hat_mean_list.append(x_filtered_t)
                sigma_hat_list.append(P_filtered_t)
                reg_list.append(reg_t)

            x_hat_mean = torch.stack(x_hat_mean_list, dim=1)
            sigma_hat = torch.stack(sigma_hat_list, dim=1)
            x_true_batch_sliced = x_true_batch[:, 1:, :]
            
            # --- ZMĚNA č. 6: Loss funkce používá `torch.sum` pro L2 i regularizaci ---
            all_regs_tensor = torch.stack(reg_list, dim=0)
            regularization_loss = torch.sum(all_regs_tensor)
            
            predicted_variances = torch.diagonal(sigma_hat, dim1=-2, dim2=-1)
            
            # POZNÁMKA: Ujistěte se, že tato funkce používá `torch.sum` pro L2
            # data_loss, l1_loss, l2_loss = empirical_averaging(
            #     target=x_true_batch_sliced, 
            #     predicted_mean=x_hat_mean, 
            #     predicted_var=predicted_variances,
            #     beta=beta
            # )
            
            
            l1_loss = 0
            l2_loss = only_variance_loss(target=x_true_batch_sliced, 
                predicted_mean=x_hat_mean, 
                predicted_var=predicted_variances)
            data_loss = l2_loss
            regularization_loss=0
            total_loss = data_loss + regularization_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            train_iter_count += 1

            # --- ZMĚNA č. 7: Logování a validace jsou řízeny počtem iterací ---
            if train_iter_count % logging_period == 0:
                print(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                print(f"  Train Loss:     {total_loss.item():.6f} | Current Beta: {beta:.6f}")
                print(f"    ├─ L1 (MSE):   {0:.6f} (váha: {1-beta:.4f})")
                print(f"    ├─ L2 (Var Sum):{l2_loss.item():.6f} (váha: {beta:.6f})")
                print(f"    └─ Regularize: {0:.6f}")
                if hasattr(model, 'dnn'):
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                    print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")

            if train_iter_count % validation_period == 0:
                model.eval()
                avg_val_loss = 0.0
                with torch.no_grad():
                    for x_true_val, y_meas_val in val_loader:
                        # ... validační logika zůstává stejná ...
                        x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                        batch_size_val, seq_len_val, _ = x_true_val.shape
                        initial_state_val = x_true_val[:, 0, :]
                        model.reset(batch_size=batch_size_val, num_samples=5, initial_state=initial_state_val)
                        val_predictions = []
                        for t in range(1, seq_len_val):
                            y_t_val = y_meas_val[:, t, :]
                            x_filtered_t_val, _, _ = model.step(y_t_val, num_samples=5)
                            val_predictions.append(x_filtered_t_val)
                        predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                        avg_val_loss += F.mse_loss(predicted_val_trajectory, x_true_val[:, 1:, :]).item()
                
                avg_val_loss /= len(val_loader)
                print(f"  Validation Loss (MSE): {avg_val_loss:.6f}")
                print("-" * (len(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")))
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = deepcopy(model.state_dict())
                
                model.train() # Vrátíme model zpět do trénovacího módu

    print("Trénování dokončeno.")
    
    # --- ZMĚNA č. 8: Uložíme nejlepší model podle validační MSE (jako fallback) nebo finální model ---
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    else:
        # Pokud se validace nikdy neprovedla, uložíme poslední stav
        print("Ukládám finální stav modelu.")
    
    model.eval()
    return model

def only_variance_loss(target, predicted_mean, predicted_var):
    """Hladká ztrátová funkce pouze pro varianci."""
    # .detach() je klíčové - gradient teče jen do predicted_var, neovlivňuje mean.
    empirical_variance = torch.square(target - predicted_mean).detach()
    l2_loss = F.mse_loss(predicted_var, empirical_variance)
    return l2_loss

def train_variance_diagnostic(
    model, train_loader, val_loader, device, 
    total_train_iter=2000, 
    lr=1e-4, # Vrátil jsem na 1e-4, protože očekáváme silnější gradienty
    clip_grad=1.0,         
    J_samples=10,          
    validation_period=100,
    logging_period=25
):
    """
    Nová diagnostická trénovací funkce, která počítá statistiky z ansámblu
    pomocí torch.mean a torch.var pro co nejlepší tok gradientů.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_iter_count = 0
    
    done = False
    while not done:
        model.train()
        
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
            optimizer.zero_grad()
            
            batch_size, seq_len, _ = x_true_batch.shape
            initial_state_batch = x_true_batch[:, 0, :]
            model.reset(initial_state=initial_state_batch, batch_size=batch_size, num_samples=J_samples)
            
            # --- ZMĚNA: Sbíráme ansámbly pro celou sekvenci ---
            ensemble_list_seq = []
            
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_ensemble_t, _ = model.step(y_t, num_samples=J_samples)
                ensemble_list_seq.append(x_ensemble_t)

            # Výsledný tvar: [num_samples, batch_size, seq_len-1, state_dim]
            full_ensemble = torch.stack(ensemble_list_seq, dim=2)
            
            # --- ZMĚNA: Výpočet statistik přesně jako u autorů ---
            # Permutujeme, aby num_samples bylo na konci: [batch_size, seq_len-1, state_dim, num_samples]
            full_ensemble_permuted = full_ensemble.permute(1, 2, 3, 0)
            
            # 1. Spočítáme průměr (finální odhad stavu)
            x_hat_mean = torch.mean(full_ensemble_permuted, dim=-1)
            
            # 2. Spočítáme varianci pomocí torch.var pro nejlepší tok gradientů
            # unbiased=False pro populaci (N), True pro vzorek (N-1). Obojí je v pořádku.
            predicted_variances = torch.var(full_ensemble_permuted, dim=-1, unbiased=True)
            
            # --- Zbytek je stejný jako v předchozí diagnostické funkci ---
            x_true_batch_sliced = x_true_batch[:, 1:, :]
            total_loss = only_variance_loss(
                target=x_true_batch_sliced, 
                predicted_mean=x_hat_mean, 
                predicted_var=predicted_variances
            )

            if torch.isnan(total_loss):
                print(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                print("  !!! Training collapsed: loss is NaN, stopping. !!!")
                done = True
                continue

            total_loss.backward()

            if train_iter_count % logging_period == 0:
                print(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                print(f"  Variance Loss (MSE): {total_loss.item():.6f}")

                with torch.no_grad():
                    empirical_variances = torch.square(x_true_batch_sliced - x_hat_mean)
                    print(f"  Stats for Predicted Var -> Avg: {torch.mean(predicted_variances).item():.6f}, "
                          f"Min: {torch.min(predicted_variances).item():.6f}, "
                          f"Max: {torch.max(predicted_variances).item():.6f}")
                    print(f"  Stats for True Var      -> Avg: {torch.mean(empirical_variances).item():.6f}, "
                          f"Min: {torch.min(empirical_variances).item():.6f}, "
                          f"Max: {torch.max(empirical_variances).item():.6f}")

                print("--- Gradient Stats ---")
                # ... (zbytek logování a validace)
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm**2
                        if torch.isnan(param.grad).any():
                           print(f'  Layer: {name:<30} | Grad is NaN!')
                        else:
                           print(f'  Layer: {name:<30} | Grad norm: {grad_norm:.6f}, '
                                 f'Max grad: {torch.max(torch.abs(param.grad)).item():.6f}')
                total_grad_norm = total_grad_norm**0.5
                print(f"  Total Grad Norm (before clipping): {total_grad_norm:.6f}")
                print("----------------------")


            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1

            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                model.eval()
                avg_val_loss = 0.0
                with torch.no_grad():
                    for x_true_val, y_meas_val in val_loader:
                        x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                        batch_size_val, seq_len_val, _ = x_true_val.shape
                        initial_state_val = x_true_val[:, 0, :]
                        model.reset(batch_size=batch_size_val, num_samples=5, initial_state=initial_state_val)
                        val_predictions = []
                        for t in range(1, seq_len_val):
                            y_t_val = y_meas_val[:, t, :]
                            # Zde musíme vzít průměr, protože step vrací ansámbl
                            x_ensemble_val_t, _ = model.step(y_t_val, num_samples=5)
                            x_filtered_t_val = x_ensemble_val_t.mean(dim=0)
                            val_predictions.append(x_filtered_t_val)
                        predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                        avg_val_loss += F.mse_loss(predicted_val_trajectory, x_true_val[:, 1:, :]).item()
                
                avg_val_loss /= len(val_loader)
                print(f"  Validation Loss (MSE): {avg_val_loss:.6f}")
                print("-" * (len(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")))
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = deepcopy(model.state_dict())
                
                model.train()
    
    print("Trénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    
    model.eval()
    return model

# def train_bkn_with_empirical_averaging(
#     model, train_loader, val_loader, device, 
#     total_train_iter=2000,
#     lr=1e-4,
#     clip_grad=1.0,
#     J_samples=20,
#     # Následující parametr se nepoužije, pokud je aktivní dvoufázový trénink
#     final_beta=0.5, 
#     reg_weight=1e-5,
#     validation_period=100,
#     logging_period=25,
#     # Parametr pro dvoufázový trénink
#     pretrain_iters=300
# ):
#     """
#     Plnohodnotná trénovací funkce s `empirical_averaging` a opraveným tokem
#     gradientů přes `torch.var`, rozšířená o hloubkovou diagnostiku gradientů.
#     Implementuje dvoufázový trénink.
#     """
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     best_val_loss = float('inf')
#     best_model_state = None
#     train_iter_count = 0
    
#     # Seznamy pro sběr gradientů pro diagnostiku
#     mean_grads_log = []
#     var_grads_log = []

#     def save_mean_grad(grad):
#         if grad is not None:
#             mean_grads_log.append(grad.norm().item())

#     def save_var_grad(grad):
#         if grad is not None:
#             var_grads_log.append(grad.norm().item())

#     done = False
#     while not done:
#         model.train()
        
#         for x_true_batch, y_meas_batch in train_loader:
#             if train_iter_count >= total_train_iter:
#                 done = True
#                 break

#             x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
#             optimizer.zero_grad()
            
#             # Dvoufázový beta scheduling
#             # if train_iter_count < pretrain_iters:
#             #     beta = 0.0
#             # else:
#             #     beta = 1.0 # nebo 0.99
#             beta = final_beta * (train_iter_count / total_train_iter)

#             batch_size, seq_len, _ = x_true_batch.shape
#             initial_state_batch = x_true_batch[:, 0, :]
#             model.reset(initial_state=initial_state_batch, batch_size=batch_size, num_samples=J_samples)
            
#             ensemble_list_seq = []
#             reg_list_seq = []
            
#             for t in range(1, seq_len):
#                 y_t = y_meas_batch[:, t, :]
#                 x_ensemble_t, reg_t = model.step(y_t, num_samples=J_samples)
#                 ensemble_list_seq.append(x_ensemble_t)
#                 reg_list_seq.append(reg_t)

#             full_ensemble = torch.stack(ensemble_list_seq, dim=2)
#             full_ensemble_permuted = full_ensemble.permute(1, 2, 3, 0)
            
#             x_hat_mean = torch.mean(full_ensemble_permuted, dim=-1)
#             predicted_variances = torch.var(full_ensemble_permuted, dim=-1, unbiased=True)
            
#             # Registrace "hooks" pro sběr gradientů
#             if x_hat_mean.requires_grad:
#                 x_hat_mean.register_hook(save_mean_grad)
#             if predicted_variances.requires_grad:
#                 predicted_variances.register_hook(save_var_grad)

#             x_true_batch_sliced = x_true_batch[:, 1:, :]
            
#             data_loss, l1_loss, l2_loss = empirical_averaging_detached(
#                 target=x_true_batch_sliced, 
#                 predicted_mean=x_hat_mean, 
#                 predicted_var=predicted_variances,
#                 beta=beta
#             )
            
#             all_regs_tensor = torch.stack([torch.sum(r) for r in reg_list_seq])
#             regularization_loss = torch.sum(all_regs_tensor)
#             reg_weight=1
#             total_loss = data_loss + reg_weight * regularization_loss

#             if torch.isnan(total_loss):
#                 print(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")
#                 print("  !!! Training collapsed: loss is NaN, stopping. !!!")
#                 done = True
#                 continue

#             total_loss.backward()

#             if train_iter_count % logging_period == 0:
#                 print(f"\n--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                
#                 print("--- Loss Stats ---")
#                 print(f"  Total Loss:       {total_loss.item():.6f} | Current Beta: {beta:.4f}")
#                 print(f"    ├─ L1 (MSE):   {l1_loss.item():.6f} (váha: {1-beta:.2f})")
#                 print(f"    ├─ L2 (Var L1):{l2_loss.item():.6f} (váha: {beta:.4f})")
#                 print(f"    └─ Regularize: {regularization_loss.item():.6f} (váha: {reg_weight})")
                
#                 with torch.no_grad():
#                     empirical_variances = torch.square(x_true_batch_sliced - x_hat_mean)
#                     print("--- Variance Stats ---")
#                     print(f"  Predicted Var -> Avg: {torch.mean(predicted_variances).item():.6f}, "
#                           f"Min: {torch.min(predicted_variances).item():.6f}, "
#                           f"Max: {torch.max(predicted_variances).item():.6f}")
#                     print(f"  True Var      -> Avg: {torch.mean(empirical_variances).item():.6f}, "
#                           f"Min: {torch.min(empirical_variances).item():.6f}, "
#                           f"Max: {torch.max(empirical_variances).item():.6f}")
#                     print("--- Mean (State) Stats ---")
#                     print(f"  Predicted Mean-> Avg: {torch.mean(x_hat_mean).item():.6f}, "
#                           f"Min: {torch.min(x_hat_mean).item():.6f}, "
#                           f"Max: {torch.max(x_hat_mean).item():.6f}")
#                     print(f"  True Mean     -> Avg: {torch.mean(x_true_batch_sliced).item():.6f}, "
#                           f"Min: {torch.min(x_true_batch_sliced).item():.6f}, "
#                           f"Max: {torch.max(x_true_batch_sliced).item():.6f}")
                
#                 print("--- Gradient Stats ---")
#                 total_grad_norm = 0.0
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         grad_norm = param.grad.norm().item()
#                         total_grad_norm += grad_norm**2
#                         if torch.isnan(param.grad).any():
#                            print(f'  Layer: {name:<30} | Grad is NaN!')
#                         else:
#                            print(f'  Layer: {name:<30} | Grad norm: {grad_norm:.6f}')
#                 total_grad_norm = total_grad_norm**0.5
#                 print(f"  Total Grad Norm (before clipping): {total_grad_norm:.6f}")

#                 print("--- Intermediate Gradient Stats ---")
#                 if mean_grads_log:
#                     print(f"  Avg Grad Norm on x_hat_mean:    {np.mean(mean_grads_log):.6f}")
#                 else:
#                     print(f"  Avg Grad Norm on x_hat_mean:    N/A (beta=1 or no grad)")
#                 if var_grads_log:
#                     print(f"  Avg Grad Norm on predicted_var: {np.mean(var_grads_log):.6f}")
#                 else:
#                     print(f"  Avg Grad Norm on predicted_var: N/A (beta=0 or no grad)")
#                 mean_grads_log.clear()
#                 var_grads_log.clear()

#                 print("--- Model Stats ---")
#                 if hasattr(model, 'dnn') and hasattr(model.dnn, 'concrete_dropout1'):
#                     p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
#                     p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
#                     print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")

#             nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#             optimizer.step()
#             train_iter_count += 1

#             if train_iter_count > 0 and train_iter_count % validation_period == 0:
#                 model.eval()
#                 avg_val_loss = 0.0
#                 with torch.no_grad():
#                     for x_true_val, y_meas_val in val_loader:
#                         x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
#                         batch_size_val, seq_len_val, _ = x_true_val.shape
#                         initial_state_val = x_true_val[:, 0, :]
#                         model.reset(batch_size=batch_size_val, num_samples=5, initial_state=initial_state_val)
#                         val_predictions = []
#                         for t in range(1, seq_len_val):
#                             y_t_val = y_meas_val[:, t, :]
#                             x_ensemble_val_t, _ = model.step(y_t_val, num_samples=5)
#                             x_filtered_t_val = x_ensemble_val_t.mean(dim=0)
#                             val_predictions.append(x_filtered_t_val)
#                         predicted_val_trajectory = torch.stack(val_predictions, dim=1)
#                         avg_val_loss += F.mse_loss(predicted_val_trajectory, x_true_val[:, 1:, :]).item()
                
#                 avg_val_loss /= len(val_loader)
#                 print(f"\n  Validation Loss (MSE): {avg_val_loss:.6f}")
#                 print("-" * (len(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---") + 2))
                
#                 if avg_val_loss < best_val_loss:
#                     print(f"  Nová nejlepší validační ztráta! Ukládám model.")
#                     best_val_loss = avg_val_loss
#                     best_model_state = deepcopy(model.state_dict())
                
#                 model.train()
    
#     print("\nTrénování dokončeno.")
#     if best_model_state:
#         print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
#         model.load_state_dict(best_model_state)
#     else:
#         print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")
    
#     model.eval()
#     return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

# Předpokládáme, že tato ztrátová funkce je definována ve stejném souboru nebo správně naimportována
def empirical_averaging_detached_sum(target, predicted_mean, predicted_var, beta):
    """
    Ztrátová funkce s oddělenými gradienty, která používá SUM pro L2 složku.
    """
    l1_loss = F.mse_loss(predicted_mean, target)
    empirical_variance = torch.square(target - predicted_mean).detach()
    l2_loss = torch.sum(torch.abs(empirical_variance - predicted_var))
    total_data_loss = (1 - beta) * l1_loss + beta * l2_loss
    return total_data_loss, l1_loss, l2_loss


def train_bkn_with_empirical_averaging(
    model, train_loader, val_loader, device, 
    total_train_iter=2000,
    lr_phase1=1e-4,
    lr_phase2=1e-5,
    clip_grad=10.0,
    J_samples=20,
    reg_weight=1.0,
    validation_period=100,
    logging_period=25,
    pretrain_iters=1000,
    final_beta=0.01
):
    """
    Přesná replika trénovacího procesu autorů s dvoufázovým tréninkem
    a detailním logováním, přizpůsobená pro vaši iterativní `StateBayesianKalmanNet`.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_phase1)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_iter_count = 0
    phase = 1
    
    done = False
    while not done:
        model.train()
        
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            if train_iter_count == pretrain_iters and phase == 1:
                phase = 2
                print("\n" + "="*50)
                print(f"KONEC FÁZE 1. PŘEPÍNÁM NA FÁZI 2 (trénink variance).")
                print(f"Měním learning rate na {lr_phase2}")
                print("="*50 + "\n")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_phase2

            x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
            optimizer.zero_grad()
            
            # beta = 0.0 if phase == 1 else 1.0
            
            beta = final_beta * (train_iter_count / total_train_iter)
            batch_size, seq_len, state_dim = x_true_batch.shape
            
            x_hat_temp = torch.zeros((batch_size, seq_len, state_dim, J_samples), device=device)
            regularization_temp = torch.zeros((batch_size, 2, seq_len, J_samples), device=device)

            for i in range(batch_size):
                for mc_ind in range(J_samples):
                    initial_state = x_true_batch[i, 0, :].unsqueeze(-1)
                    
                    # --- OPRAVA ZDE ---
                    # Modelu předáme jen měření od t=1, protože stav v t=0 je daný.
                    y_sequence_sliced = y_meas_batch[i, 1:, :]
                    
                    # Nyní model zpracuje `seq_len - 1` měření.
                    state_history, reg_history = model(y_sequence_sliced, initial_state)
                    
                    # `state_history` teď obsahuje `1 (initial) + (seq_len - 1)` kroků = `seq_len` kroků.
                    # Její tvar je [dim, seq_len], po transpozici [seq_len, dim].
                    # To se vejde do x_hat_temp[i, :, :, mc_ind], který má tvar [seq_len, dim].
                    x_hat_temp[i, :, :, mc_ind] = state_history.T
                    
                    reg_hist_padded = F.pad(reg_history, (1, 0), "constant", 0)
                    regularization_temp[i, :, :, mc_ind] = reg_hist_padded

            x_hat = torch.mean(x_hat_temp, dim=-1)
            predicted_variances = torch.var(x_hat_temp, dim=-1, unbiased=True)
            regularization = torch.mean(regularization_temp, dim=-1)

            x_true_sliced = x_true_batch[:, 1:, :]
            x_hat_sliced = x_hat[:, 1:, :]
            predicted_variances_sliced = predicted_variances[:, 1:, :]
            regularization_sliced = regularization[:, :, 1:]
            
            data_loss, l1_loss, l2_loss = empirical_averaging_detached_sum(
                target=x_true_sliced, 
                predicted_mean=x_hat_sliced, 
                predicted_var=predicted_variances_sliced,
                beta=beta
            )
            
            regularization_loss = torch.sum(regularization_sliced)
            total_loss = data_loss + reg_weight * regularization_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"!!! Kolaps v iteraci {train_iter_count}, ztráta je NaN/Inf. Ukončuji. !!!")
                done = True
                break

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1

            if train_iter_count % logging_period == 0:
                print(f"\n--- Iterace [{train_iter_count}/{total_train_iter}] | Fáze: {phase} ---")
                print("--- Loss Stats ---")
                print(f"  Total Loss:       {total_loss.item():.6f} | Current Beta: {beta:.4f}")
                print(f"    ├─ L1 (MSE):   {l1_loss.item():.6f} (váha: {1-beta:.2f})")
                print(f"    ├─ L2 (Var Sum):{l2_loss.item():.6f} (váha: {beta:.4f})")
                print(f"    └─ Regularize: {regularization_loss.item():.6f} (váha: {reg_weight})")
                
                with torch.no_grad():
                    empirical_variances = torch.square(x_true_sliced - x_hat_sliced)
                    print("--- Variance Stats ---")
                    print(f"  Predicted Var -> Avg: {torch.mean(predicted_variances_sliced).item():.6f}, Min: {torch.min(predicted_variances_sliced).item():.6f}, Max: {torch.max(predicted_variances_sliced).item():.6f}")
                    print(f"  True Var      -> Avg: {torch.mean(empirical_variances).item():.6f}, Min: {torch.min(empirical_variances).item():.6f}, Max: {torch.max(empirical_variances).item():.6f}")
                    print("--- Mean (State) Stats ---")
                    print(f"  Predicted Mean-> Avg: {torch.mean(x_hat_sliced).item():.6f}, Min: {torch.min(x_hat_sliced).item():.6f}, Max: {torch.max(x_hat_sliced).item():.6f}")
                    print(f"  True Mean     -> Avg: {torch.mean(x_true_sliced).item():.6f}, Min: {torch.min(x_true_sliced).item():.6f}, Max: {torch.max(x_true_sliced).item():.6f}")
                
                print("--- Gradient Stats ---")
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm**2
                        if torch.isnan(param.grad).any():
                           print(f'  Layer: {name:<30} | Grad is NaN!')
                        else:
                           print(f'  Layer: {name:<30} | Grad norm: {grad_norm:.6f}')
                total_grad_norm = total_grad_norm**0.5
                print(f"  Total Grad Norm (before clipping): {total_grad_norm:.6f}")
                
                print("--- Model Stats ---")
                if hasattr(model, 'dnn') and hasattr(model.dnn, 'concrete_dropout1'):
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                    print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")

            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print("\n--- Spouštím validaci ---")
                model.eval()
                avg_val_loss = 0.0
                with torch.no_grad():
                    for x_true_val, y_meas_val in val_loader:
                        x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                        val_batch_size, val_seq_len, _ = x_true_val.shape
                        x_hat_val_batch = torch.zeros_like(x_true_val)
                        
                        for i_val in range(val_batch_size):
                            initial_state_val = x_true_val[i_val, 0, :].unsqueeze(-1)
                            y_sequence_val_sliced = y_meas_val[i_val, 1:, :]
                            
                            state_history_val, _ = model(y_sequence_val_sliced, initial_state_val)
                            x_hat_val_batch[i_val, :, :] = state_history_val.T
                        
                        avg_val_loss += F.mse_loss(x_hat_val_batch[:, 1:, :], x_true_val[:, 1:, :]).item()

                avg_val_loss /= len(val_loader)
                print(f"  Validation Loss (MSE): {avg_val_loss:.6f}")
                
                if avg_val_loss < best_val_loss:
                    print(f"  Nová nejlepší validační ztráta! Ukládám model.")
                    best_val_loss = avg_val_loss
                    best_model_state = deepcopy(model.state_dict())
                
                print("-" * (len(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---") + 2))
                model.train()
    
    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")
    
    model.eval()
    return model

def gaussian_nll_stable(target, predicted_mean, predicted_var):
    """
    Robustní verze G-NLL ztrátové funkce s ochrannými prvky.
    """
    # 1. Zajištění, že variance není příliš malá a je vždy kladná
    epsilon = 1e-4 
    # Softplus je hladká funkce, která mapuje R -> R+. Je to lepší než clamp nebo abs.
    predicted_var_stable = F.softplus(predicted_var) + epsilon
    
    # 2. Výpočet členů NLL
    mahal = torch.square(target - predicted_mean) / predicted_var_stable
    log_2pi = torch.log(torch.tensor(2 * torch.pi, device=target.device))
    log_det_term = torch.log(predicted_var_stable)
    
    element_wise_nll = 0.5 * (log_det_term + log_2pi + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-1)
    return torch.mean(sample_wise_error)

def train_bkn_ultimate_diagnostic(
    model, train_loader, val_loader, device, 
    total_train_iter=5000,
    lr=1e-5,
    clip_grad=1.0,
    J_samples=20,          
    reg_weight=1e-6,
    validation_period=100, 
    logging_period=25
):
    """
    Ultimátní diagnostická trénovací funkce s maximálním množstvím logů.
    Používá stabilizovanou G-NLL ztrátu a vektorizovaný výpočet `torch.var`.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_iter_count = 0
    
    done = False
    while not done:
        model.train()
        
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
            optimizer.zero_grad()
            
            # --- Získání výsledků z modelu ---
            batch_size, seq_len, _ = x_true_batch.shape
            initial_state_batch = x_true_batch[:, 0, :]
            model.reset(initial_state=initial_state_batch, batch_size=batch_size, num_samples=J_samples)
            
            ensemble_list_seq, reg_list_seq = [], []
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_ensemble_t, reg_t = model.step(y_t, num_samples=J_samples)
                ensemble_list_seq.append(x_ensemble_t)
                reg_list_seq.append(reg_t)

            full_ensemble = torch.stack(ensemble_list_seq, dim=2)
            full_ensemble_permuted = full_ensemble.permute(1, 2, 3, 0)
            
            x_hat_mean = torch.mean(full_ensemble_permuted, dim=-1)
            predicted_variances = torch.var(full_ensemble_permuted, dim=-1, unbiased=True)
            
            # --- Výpočet ztráty ---
            x_true_batch_sliced = x_true_batch[:, 1:, :]
            nll_loss = gaussian_nll_stable(
                target=x_true_batch_sliced,
                predicted_mean=x_hat_mean,
                predicted_var=predicted_variances
            )
            
            regularization_loss = torch.mean(torch.stack([torch.mean(r) for r in reg_list_seq]))
            total_loss = nll_loss + reg_weight * regularization_loss
            
            # --- Diagnostika a Backward Pass ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                print("  !!! Training collapsed: loss is NaN or Inf, stopping. !!!")
                done = True
                continue

            total_loss.backward()

            # --- Detailní logování (před krokem optimalizátoru) ---
            if train_iter_count % logging_period == 0:
                print(f"\n--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                
                print("--- Loss Stats ---")
                print(f"  Total Loss:       {total_loss.item():.6f}")
                print(f"    ├─ NLL Loss:   {nll_loss.item():.6f}")
                print(f"    └─ Regularize: {regularization_loss.item():.6f} (váha: {reg_weight})")
                
                with torch.no_grad():
                    empirical_variances = torch.square(x_true_batch_sliced - x_hat_mean)
                    print("--- Variance Stats ---")
                    print(f"  Predicted Var -> Avg: {torch.mean(predicted_variances).item():.6f}, "
                          f"Min: {torch.min(predicted_variances).item():.6f}, "
                          f"Max: {torch.max(predicted_variances).item():.6f}, "
                          f"Std: {torch.std(predicted_variances).item():.6f}")
                    print(f"  True Var      -> Avg: {torch.mean(empirical_variances).item():.6f}, "
                          f"Min: {torch.min(empirical_variances).item():.6f}, "
                          f"Max: {torch.max(empirical_variances).item():.6f}, "
                          f"Std: {torch.std(empirical_variances).item():.6f}")

                    mse_for_logging = F.mse_loss(x_hat_mean, x_true_batch_sliced).item()
                    print("--- Mean Stats ---")
                    print(f"  MSE (for info):   {mse_for_logging:.6f}")
                    print(f"  Avg Predicted Mean: {torch.mean(x_hat_mean).item():.6f}")
                    print(f"  Avg True Mean:      {torch.mean(x_true_batch_sliced).item():.6f}")

                print("--- Gradient Stats ---")
                total_grad_norm = 0.0
                max_grad_overall = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        max_grad_in_layer = torch.max(torch.abs(param.grad)).item()
                        total_grad_norm += grad_norm**2
                        if max_grad_in_layer > max_grad_overall:
                            max_grad_overall = max_grad_in_layer
                        
                        if torch.isnan(param.grad).any():
                           print(f'  Layer: {name:<30} | Grad is NaN!')
                        else:
                           print(f'  Layer: {name:<30} | Grad norm: {grad_norm:.6f}, Max grad: {max_grad_in_layer:.6f}')
                total_grad_norm = total_grad_norm**0.5
                print(f"  Total Grad Norm (before clipping): {total_grad_norm:.6f}")
                print(f"  Max Grad Value Overall:            {max_grad_overall:.6f}")
                
                print("--- Model Stats ---")
                if hasattr(model, 'dnn') and hasattr(model.dnn, 'concrete_dropout1'):
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                    print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")

            # --- Krok optimalizátoru ---
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1

            # --- Validace ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                model.eval()
                avg_val_loss = 0.0
                with torch.no_grad():
                    for x_true_val, y_meas_val in val_loader:
                        x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                        batch_size_val, seq_len_val, _ = x_true_val.shape
                        initial_state_val = x_true_val[:, 0, :]
                        model.reset(batch_size=batch_size_val, num_samples=5, initial_state=initial_state_val)
                        val_predictions = []
                        for t in range(1, seq_len_val):
                            y_t_val = y_meas_val[:, t, :]
                            x_ensemble_val_t, _ = model.step(y_t_val, num_samples=5)
                            x_filtered_t_val = x_ensemble_val_t.mean(dim=0)
                            val_predictions.append(x_filtered_t_val)
                        predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                        avg_val_loss += F.mse_loss(predicted_val_trajectory, x_true_val[:, 1:, :]).item()
                
                avg_val_loss /= len(val_loader)
                print(f"\n  Validation Loss (MSE): {avg_val_loss:.6f}")
                print("-" * (len(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---") + 2))
                
                if avg_val_loss < best_val_loss:
                    print(f"  Nová nejlepší validační ztráta! Ukládám model.")
                    best_val_loss = avg_val_loss
                    best_model_state = deepcopy(model.state_dict())
                
                model.train()
    
    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")
    
    model.eval()
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

def train_bkn_dumb_replication_verbose(
    model, train_loader, val_loader, device, 
    total_train_iter=5000,
    lr=1e-4,
    clip_grad=10.0,
    J_samples=20,          
    final_beta=0.01,
    validation_period=100, 
    logging_period=25
):
    """
    Přímá, "hloupá" replika trénovacího procesu autorů s maximálním logováním.
    - Iteruje přes batch a MC-vzorky.
    - Ztrátová funkce s `torch.sum` je počítána na konci pro celou dávku.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_iter_count = 0
    
    done = False
    while not done:
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            x_true_batch, y_meas_batch = x_true_batch.to(device), y_meas_batch.to(device)
            optimizer.zero_grad()
            
            batch_size, seq_len, state_dim = x_true_batch.shape
            
            x_hat_temp = torch.zeros((batch_size, seq_len, state_dim, J_samples), device=device)
            regularization_temp = torch.zeros((batch_size, 2, seq_len, J_samples), device=device)

            for i in range(batch_size):
                for mc_ind in range(J_samples):
                    model.reset() 
                    model.state_post = x_true_batch[i, 0, :].unsqueeze(-1)
                    
                    for t in range(1, seq_len):
                        observation_t = y_meas_batch[i, t, :].unsqueeze(-1)
                        model.step(observation_t)
                    
                    x_hat_temp[i, :, :, mc_ind] = model.state_history.T
                    reg_hist_padded = F.pad(model.regularization_history, (1, 0), "constant", 0)
                    regularization_temp[i, :, :, mc_ind] = reg_hist_padded

            x_hat = torch.mean(x_hat_temp, dim=-1)
            cov_hat_bnn = torch.var(x_hat_temp, dim=-1, unbiased=True)
            regularization = torch.mean(regularization_temp, dim=-1)

            beta = final_beta * (train_iter_count / total_train_iter)
            
            x_true_sliced = x_true_batch[:, 1:, :]
            x_hat_sliced = x_hat[:, 1:, :]
            cov_hat_bnn_sliced = cov_hat_bnn[:, 1:, :]
            regularization_sliced = regularization[:, :, 1:]

            l1_loss = torch.mean(torch.square(x_true_sliced - x_hat_sliced))
            l2_loss = torch.sum(torch.abs(torch.square(x_true_sliced - x_hat_sliced) - cov_hat_bnn_sliced))
            reg_loss = torch.sum(regularization_sliced)

            total_loss = (1 - beta) * l1_loss + beta * l2_loss + reg_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"!!! Kolaps v iteraci {train_iter_count}, ztráta je NaN/Inf. Ukončuji. !!!")
                done = True
                break

            total_loss.backward()

            # --- Detailní logování (před krokem optimalizátoru) ---
            if train_iter_count % logging_period == 0:
                print(f"\n--- Iterace [{train_iter_count}/{total_train_iter}] ---")
                
                print("--- Loss Stats ---")
                print(f"  Total Loss:       {total_loss.item():.6f} | Current Beta: {beta:.4f}")
                print(f"    ├─ L1 (MSE):   {l1_loss.item():.6f} (váha: {1-beta:.2f})")
                print(f"    ├─ L2 (Var Sum):{l2_loss.item():.6f} (váha: {beta:.4f})")
                print(f"    └─ Regularize: {reg_loss.item():.6f}")
                
                with torch.no_grad():
                    empirical_variances = torch.square(x_true_sliced - x_hat_sliced)
                    print("--- Variance Stats ---")
                    print(f"  Predicted Var -> Avg: {torch.mean(cov_hat_bnn_sliced).item():.6f}, "
                          f"Min: {torch.min(cov_hat_bnn_sliced).item():.6f}, "
                          f"Max: {torch.max(cov_hat_bnn_sliced).item():.6f}, "
                          f"Std: {torch.std(cov_hat_bnn_sliced).item():.6f}")
                    print(f"  True Var      -> Avg: {torch.mean(empirical_variances).item():.6f}, "
                          f"Min: {torch.min(empirical_variances).item():.6f}, "
                          f"Max: {torch.max(empirical_variances).item():.6f}, "
                          f"Std: {torch.std(empirical_variances).item():.6f}")
                
                print("--- Gradient Stats ---")
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm**2
                        if torch.isnan(param.grad).any():
                           print(f'  Layer: {name:<30} | Grad is NaN!')
                        else:
                           print(f'  Layer: {name:<30} | Grad norm: {grad_norm:.6f}')
                total_grad_norm = total_grad_norm**0.5
                print(f"  Total Grad Norm (before clipping): {total_grad_norm:.6f}")
                
                print("--- Model Stats ---")
                if hasattr(model, 'dnn') and hasattr(model.dnn, 'concrete_dropout1'):
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                    print(f"  Learned p: p1={p1:.4f}, p2={p2:.4f}")

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1

            # --- Validace ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print("\n--- Spouštím validaci ---")
                model.eval()
                avg_val_loss = 0.0
                with torch.no_grad():
                    for x_true_val, y_meas_val in val_loader:
                        x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                        
                        # Zde musíme iterovat, protože validujeme model, který není vektorizovaný
                        val_batch_size = x_true_val.shape[0]
                        val_seq_len = x_true_val.shape[1]
                        
                        x_hat_val_batch = torch.zeros_like(x_true_val)
                        
                        for i_val in range(val_batch_size):
                            # Pro validaci stačí jeden MC vzorek pro odhad stavu
                            model.reset()
                            model.state_post = x_true_val[i_val, 0, :].unsqueeze(-1)
                            for t_val in range(1, val_seq_len):
                                model.step(y_meas_val[i_val, t_val, :].unsqueeze(-1))
                            x_hat_val_batch[i_val, :, :] = model.state_history.T
                        
                        avg_val_loss += F.mse_loss(x_hat_val_batch[:, 1:, :], x_true_val[:, 1:, :]).item()

                avg_val_loss /= len(val_loader)
                print(f"  Validation Loss (MSE): {avg_val_loss:.6f}")
                
                if avg_val_loss < best_val_loss:
                    print(f"  Nová nejlepší validační ztráta! Ukládám model.")
                    best_val_loss = avg_val_loss
                    best_model_state = deepcopy(model.state_dict())
                
                print("-" * (len(f"--- Iterace [{train_iter_count}/{total_train_iter}] ---") + 2))
                model.train()
    
    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")
    
    model.eval()
    return model