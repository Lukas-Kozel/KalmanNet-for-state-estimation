import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

from NN_models.KalmanNet_withCovMatrix import KalmanNet_withCovMatrix


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
                          epochs=100, lr=1e-3, clip_grad=1.0, early_stopping_patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
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

            # restart vnitřního stavu filtru pro novou dávku
            model.reset(batch_size=x_true_batch.shape[0])


            predictions = []
            seq_len = y_meas_batch.shape[1]

            for t in range(seq_len):

                y_t = y_meas_batch[:, t, :]

                x_filtered_t = model.step(y_t)

                predictions.append(x_filtered_t)

            predicted_trajectory = torch.stack(predictions, dim=1)

            loss = criterion(predicted_trajectory, x_true_batch)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

         # --- Validační fáze ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                
                batch_size_val = x_true_val.shape[0]
                seq_len_val = x_true_val.shape[1]
                
                # Resetujeme stav i pro validaci
                model.reset(batch_size=batch_size_val)
                
                val_predictions = []
                for t in range(seq_len_val):
                    y_t_val = y_meas_val[:, t, :]
                    x_filtered_t_val = model.step(y_t_val)
                    val_predictions.append(x_filtered_t_val)
                    
                predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                
                val_loss_batch = criterion(predicted_val_trajectory, x_true_val)
                epoch_val_loss += val_loss_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
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
                       early_stopping_patience=20, 
                       J_samples=10, 
                       final_beta=0.01,
                       beta_warmup_epochs=None):
    """
    Finální trénovací funkce pro BKN, která správně inicializuje filtr a ořezává sekvence.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
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
        scheduler.step(avg_val_loss)

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