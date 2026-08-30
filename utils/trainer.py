import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .losses import gaussian_nll_safe, hybrid_loss,empirical_loss_mean,empirical_loss_sum,gaussian_nll,gaussian_nll_loss_with_alpha
from .utils import calculate_anees_vectorized
from torch.optim.lr_scheduler import ReduceLROnPlateau


def training_session_trajectory_with_gaussian_nll_training_fcn(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    warmup_iterations=0, store_model_based_on_hybrid_score=False, calibration_parameter=1.0
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_anees = float('inf')
    best_hybrid_score = float('inf')
    score_at_best = {"val_nll": 0.0, "val_mse": 0.0, "hybrid_score": None}
    best_iter_count = 0
    best_model_state = None
    train_iter_count = 0
    done = False
    state_dim = model.system_model.state_dim
    while not done:
        model.train()
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter: done = True; break
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            # --- Training ---
            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            
            all_trajectories_for_ensemble = []
            all_regs_for_ensemble = []

            for j in range(J_samples):
                model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])
                current_trajectory_x_hats = []
                current_trajectory_regs = []
                for t in range(1, seq_len):
                    y_t = y_meas_batch[:, t, :]
                    x_filtered_t, reg_t = model.step(y_t)
                    current_trajectory_x_hats.append(x_filtered_t)
                    current_trajectory_regs.append(reg_t)
                all_trajectories_for_ensemble.append(torch.stack(current_trajectory_x_hats, dim=1))
                all_regs_for_ensemble.append(torch.sum(torch.stack(current_trajectory_regs)))

            ensemble_trajectories = torch.stack(all_trajectories_for_ensemble, dim=0)
            x_hat_sequence = ensemble_trajectories.mean(dim=0)
            cov_diag_sequence = ensemble_trajectories.var(dim=0)
            regularization_loss = torch.stack(all_regs_for_ensemble).mean()
            target_sequence = x_true_batch[:, 1:, :]

            if train_iter_count < warmup_iterations:
                loss = F.mse_loss(x_hat_sequence, target_sequence) + regularization_loss
                nll_loss = torch.tensor(0.0)
            else:
                nll_loss = gaussian_nll(target_sequence, x_hat_sequence, cov_diag_sequence)
                loss = nll_loss + regularization_loss
            
            if torch.isnan(loss): print("Collapse detected (NaN loss)"); done = True; break
            
            loss.backward()
            if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1
            
            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---", f"Total Loss: {loss.item():.4f}", f"NLL: {nll_loss.item():.4f}", f"Reg: {regularization_loss.item():.4f}", f"p1={p1:.3f}, p2={p2:.3f}", sep="\n    - ")

            # --- Validation step ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validation at iteration {train_iter_count} ---")
                model.eval()
                val_mse_list = []
                all_val_x_true_cpu, all_val_x_hat_cpu, all_val_P_hat_cpu = [], [], []

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        x_true_val_batch = x_true_val_batch.to(device)
                        y_meas_val_batch = y_meas_val_batch.to(device)
                        val_ensemble_trajectories = []
                        for j in range(J_samples):
                            model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])
                            val_current_x_hats = []
                            for t in range(1, val_seq_len):
                                y_t_val = y_meas_val_batch[:, t, :]
                                x_filtered_t, _ = model.step(y_t_val)
                                val_current_x_hats.append(x_filtered_t)
                            val_ensemble_trajectories.append(torch.stack(val_current_x_hats, dim=1))
                        
                        # Shape: [J, B_val, T_val-1, D_state]
                        val_ensemble = torch.stack(val_ensemble_trajectories, dim=0)
                        
                        # Mean over J gives the final trajectory estimate
                        val_preds_seq = val_ensemble.mean(dim=0)
                        
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())
                        
                        initial_state_val = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state_val, val_preds_seq], dim=1)
                        
                        diff = val_ensemble - val_preds_seq.unsqueeze(0)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=0) # Mean over J (dimension 0)

                        P0 = model.system_model.P0.unsqueeze(0).repeat(val_batch_size, 1, 1).unsqueeze(1)
                        full_P_hat = torch.cat([P0, val_covs_full], dim=1)
                        
                        all_val_x_true_cpu.append(x_true_val_batch.cpu())
                        all_val_x_hat_cpu.append(full_x_hat.cpu())
                        all_val_P_hat_cpu.append(full_P_hat.cpu())

                avg_val_mse = np.mean(val_mse_list)
                final_x_true_list = torch.cat(all_val_x_true_cpu, dim=0)
                final_x_hat_list = torch.cat(all_val_x_hat_cpu, dim=0)
                final_P_hat_list = torch.cat(all_val_P_hat_cpu, dim=0)
                avg_val_anees = calculate_anees_vectorized(final_x_true_list, final_x_hat_list, final_P_hat_list)
                
                print(f"  Average MSE: {avg_val_mse:.4f}, Average ANEES: {avg_val_anees:.4f}")
                
                # --- MODEL SAVING LOGIC ---
                if not np.isnan(avg_val_anees) and avg_val_anees > 0:
                    
                    if store_model_based_on_hybrid_score:
                        # NEW LOGIC: Saving based on Hybrid Score
                        anees_penalty = abs(avg_val_anees - state_dim)
                        current_hybrid_score = avg_val_mse + (calibration_parameter * anees_penalty)
                        print(f"  Hybrid Score: {current_hybrid_score:.4f} (Penalty: {anees_penalty:.4f} * {calibration_parameter})")
                        
                        if current_hybrid_score < best_hybrid_score:
                            print(f"  >>> New best HYBRID SCORE! ({current_hybrid_score:.4f} < {best_hybrid_score:.4f}) Saving model. <<<")
                            best_hybrid_score = current_hybrid_score
                            best_val_anees = avg_val_anees
                            best_iter_count = train_iter_count
                            score_at_best['val_mse'] = avg_val_mse
                            score_at_best['hybrid_score'] = current_hybrid_score
                            best_model_state = deepcopy(model.state_dict())
                            
                    else:
                        # ORIGINAL LOGIC: Saving based only on ANEES
                        if avg_val_anees < best_val_anees:
                            print("  >>> New best VALIDATION ANEES! Saving model. <<<")
                            best_val_anees = avg_val_anees
                            best_iter_count = train_iter_count
                            score_at_best['val_mse'] = avg_val_mse
                            best_model_state = deepcopy(model.state_dict())
                            
                print("-" * 50)
                model.train()

    print("\nTraining completed.")
    if best_model_state:
        if store_model_based_on_hybrid_score:
            print(f"Loading best model from iteration {best_iter_count} with Hybrid Score {best_hybrid_score:.4f} (MSE: {score_at_best['val_mse']:.4f}, ANEES: {best_val_anees:.4f})")
        else:
            print(f"Loading best model from iteration {best_iter_count} with ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("No best model was saved; returning last state.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_hybrid_score": score_at_best.get('hybrid_score', None),
        "best_iter": best_iter_count,
        "final_model": model
    }

def train_state_KalmanNet(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20, optimizer_type=torch.optim.AdamW,weight_decay=1e-5,
                          print_gradient=False):
    """
    Universal training function for StateKalmanNet and StateKalmanNetWithKnownR.
    Automatically detects whether the model returns covariance and adapts accordingly.
    """

    criterion = nn.MSELoss()
    metric_mse = nn.MSELoss()

    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Error: Model does not define attribute 'returns_covariance'.")
        
    returns_covariance = model.returns_covariance

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        epoch_traces = []
        train_pos_loss_sum = 0.0
        train_vel_loss_sum = 0.0

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            
            initial_state = x_true_batch[:, 0, :]

            model.reset(batch_size=batch_size, initial_state=initial_state)

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
            target_trajectory = x_true_batch[:, 1:, :]

            if returns_covariance and predictions_P:
                predicted_cov_trajectory = torch.stack(predictions_P, dim=1)
                avg_trace_batch = torch.mean(torch.sum(torch.diagonal(predicted_cov_trajectory.detach(), offset=0, dim1=-2, dim2=-1), dim=-1)).item()
                epoch_traces.append(avg_trace_batch)

            loss = criterion(predicted_trajectory, x_true_batch[:, 1:, :])
            # diff = predicted_trajectory - x_true_batch[:, 1:, :]
            # pos_loss = torch.mean(diff[:, :, :2]**2) # Pozice X, Y
            # vel_loss = torch.mean(diff[:, :, 2:]**2) # Rychlost
            # loss = 100.0 * pos_loss + 1.0 * vel_loss

            loss.backward()

            if print_gradient:
                print(f"gradient: {model.dnn.input_layer[0].weight.grad.abs().mean().item()}")
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_epoch_trace = np.mean(epoch_traces) if epoch_traces else 0.0

        # --- Validation phase ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                batch_size_val, seq_len_val, _ = x_true_val.shape

                initial_state = x_true_val[:,0,:]
 
                model.reset(batch_size=batch_size_val, initial_state=initial_state)
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
                target_val_trajectory = x_true_val[:, 1:, :]

                val_loss_batch = criterion(predicted_val_trajectory, target_val_trajectory)
                epoch_val_loss += val_loss_batch.item()
                
                
                mse_batch = metric_mse(predicted_val_trajectory, target_val_trajectory)
                epoch_val_mse += mse_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} (Pos: {train_pos_loss_sum/len(train_loader):.2f}, Vel: {train_vel_loss_sum/len(train_loader):.2f}) | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.2f}')
            if returns_covariance:
                log_message += f', Avg Cov Trace: {avg_epoch_trace:.6f}'
            print(log_message)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            print(f"New best model saved! {log_message}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
            
    print("Training completed.")
    if best_model_state:
        print(f"Loading best model with validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def train_state_KalmanNet_sliding_window(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20,
                          tbptt_k=2, tbptt_w=10,optimizer_=torch.optim.Adam, weight_decay_=1e-4):
    criterion = nn.MSELoss()
    optimizer = optimizer_(model.parameters(), lr=lr, weight_decay=weight_decay_)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Error: Model does not define attribute 'returns_covariance'.")
    if not hasattr(model, '_detach'):
            raise AttributeError("Error: Model does not implement method `_detach()`, "
                                 "which is required for TBPTT(k,w,D).")
        
    returns_covariance = model.returns_covariance
    print(f"INFO: Detected from model attribute that it returns covariance: {returns_covariance}")
    print(f"INFO: Starting training with TBPTT(k={tbptt_k}, w={tbptt_w})")

    # if init_noise_std is None:
    #     init_noise_std = torch.tensor([5.0, 5.0, 2.0, 2.0], device=device)
    # else:
    #     init_noise_std = init_noise_std.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_windows_epoch = 0
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            batch_size, seq_len, _ = x_true_batch.shape # seq_len is 'D' from the paper
            
            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

            total_loss_for_batch = 0.0
            num_windows = 0

            # Loop over the entire sequence in windows of 'w'
            for t_start in range(1, seq_len, tbptt_w):
                t_end = min(t_start + tbptt_w, seq_len)
                window_len = t_end - t_start
                
                if window_len == 0:
                    continue
                    
                predictions_x = []

                # 1. Forward pass over window 'w'
                for t in range(t_start, t_end):
                    y_t = y_meas_batch[:, t, :]
                    step_output = model.step(y_t)
                    
                    x_filtered_t = step_output[0] if returns_covariance else step_output
                    predictions_x.append(x_filtered_t)
                    
                    # Detach gradients every 'k' steps
                    if (t - t_start + 1) % tbptt_k == 0:
                        model._detach()

                # 3. Detach at window end (before backward())
                model._detach()
                
                # 4. Compute loss ONLY for this window (HYBRID)
                predicted_window = torch.stack(predictions_x, dim=1)
                true_window = x_true_batch[:, t_start:t_end, :]
                
                loss = criterion(predicted_window, true_window)

                # 5. Backward pass and weight update (per window)
                optimizer.zero_grad()
                loss.backward()
                
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                total_loss_for_batch += loss.item()                
                num_windows += 1
            
            if num_windows > 0:
                train_loss += (total_loss_for_batch / num_windows)
                total_windows_epoch += num_windows
        
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0 # For the pure metric

        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                batch_size_val, seq_len_val, _ = x_true_val.shape
                model.reset(batch_size=batch_size_val, initial_state=x_true_val[:, 0, :])
                
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
                target_val_trajectory = x_true_val[:, 1:, :]
                val_loss_batch = criterion(predicted_val_trajectory, target_val_trajectory)
                epoch_val_loss += val_loss_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        
        # Logging with breakdown
        if (epoch + 1) % 5 == 0:
            log_message = (f'Epoch [{epoch+1}/{epochs}] | '
                           f'Train Loss: {avg_train_loss:.4f} | '
                           f'Val Loss: {avg_val_loss:.4f}')
            print(log_message)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
            print(f"New best model saved! {log_message}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
            
    print("Training completed.")
    if best_model_state:
        print(f"Loading best model with validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model

import torch
import torch.nn as nn
from copy import deepcopy


def train_state_KalmanNet_sliding_windowNCLT(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20,
                          tbptt_k=2, tbptt_w=10, optimizer_=torch.optim.Adam, weight_decay_=1e-4):
    
    criterion = nn.MSELoss()
    optimizer = optimizer_(model.parameters(), lr=lr, weight_decay=weight_decay_)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Detection of model capabilities
    if not hasattr(model, 'returns_covariance'):
         # If the model doesn't have it, we assume False (KalmanNet does not return P by default)
         returns_covariance = False
    else:
         returns_covariance = model.returns_covariance
         
    if not hasattr(model, '_detach'):
            raise AttributeError("Error: Model must implement `_detach()` for TBPTT.")

    print(f"INFO: Starting training with TBPTT(k={tbptt_k}, w={tbptt_w})")
    print(f"INFO: Returns covariance: {returns_covariance}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_windows_epoch = 0

        
        for x_true_batch, y_meas_batch, u_input_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            u_input_batch = u_input_batch.to(device) # We also send inputs to the GPU

            batch_size, seq_len, _ = x_true_batch.shape
            
            # Reset state at the beginning of the sequence
            
            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

            total_loss_for_batch = 0.0
            num_windows = 0

            # --- Sliding Window (TBPTT) ---
            for t_start in range(1, seq_len, tbptt_w):
                t_end = min(t_start + tbptt_w, seq_len)
                window_len = t_end - t_start
                
                if window_len == 0:
                    continue
                    
                predictions_x = []

                # 1. Forward pass over window
                for t in range(t_start, t_end):
                    y_t = y_meas_batch[:, t, :]
                    u_t = u_input_batch[:, t, :] 
                    
                    
                    step_output = model.step(y_t, u_t)
                    
                    x_filtered_t = step_output[0] if returns_covariance else step_output
                    predictions_x.append(x_filtered_t)
                    
                    # Detach gradients every 'k' steps (BPTT truncation)
                    if (t - t_start + 1) % tbptt_k == 0:
                        model._detach()

                # 2. Detach at window end
                model._detach()
                
                # 3. Compute loss ONLY for this window
                predicted_window = torch.stack(predictions_x, dim=1)
                
                # Target: Position [px, py] (we ignore theta for MSE, or should we include it?)
                # Here we compute MSE over the entire state (if x_true has 3 dimensions and the prediction too)
                true_window = x_true_batch[:, t_start:t_end, :]
                
                loss = criterion(predicted_window, true_window)

                # 4. Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                total_loss_for_batch += loss.item()                
                num_windows += 1
            
            if num_windows > 0:
                train_loss += (total_loss_for_batch / num_windows)
                total_windows_epoch += num_windows
        
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            
            for x_true_val, y_meas_val, u_input_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                u_input_val = u_input_val.to(device)
                
                batch_size_val, seq_len_val, _ = x_true_val.shape
                model.reset(batch_size=batch_size_val, initial_state=x_true_val[:, 0, :])
                
                val_predictions = []
                for t in range(1, seq_len_val):
                    y_t_val = y_meas_val[:, t, :]
                    u_t_val = u_input_val[:, t, :] # Vstup pro validaci
                    
                    step_output_val = model.step(y_t_val, u_t_val)
                    
                    if returns_covariance:
                        x_filtered_t_val = step_output_val[0]
                    else:   
                        x_filtered_t_val = step_output_val

                    val_predictions.append(x_filtered_t_val)
                    
                predicted_val_trajectory = torch.stack(val_predictions, dim=1)
                target_val_trajectory = x_true_val[:, 1:, :]
                
                val_loss_batch = criterion(predicted_val_trajectory, target_val_trajectory)
                epoch_val_loss += val_loss_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Logging
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            print(f" -> New best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
            
    print("Training completed.")
    if best_model_state:
        print(f"Loading best model with validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model

def train_BayesianKalmanNet_with_input(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    warmup_iterations=0, weight_decay_=1e-5
):
    # torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_)
    best_val_anees = float('inf')
    score_at_best = {"val_nll": 0.0, "val_mse": 0.0}
    best_iter_count = 0
    best_model_state = None
    train_iter_count = 0
    done = False

    while not done:
        model.train()
        for x_true_batch, y_meas_batch, u_input_batch in train_loader:
            if train_iter_count >= total_train_iter: done = True; break
            if torch.isnan(x_true_batch).any():
                print(f"!!! SKIP BATCH iter {train_iter_count}: NaN found in x_true (Ground Truth) !!!")
                continue
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            u_input_batch = u_input_batch.to(device)
            # --- Training ---
            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            
            all_trajectories_for_ensemble = []
            all_regs_for_ensemble = []

            for j in range(J_samples):
                model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])
                current_trajectory_x_hats = []
                current_trajectory_regs = []
                for t in range(1, seq_len):
                    y_t = y_meas_batch[:, t, :]
                    u_t = u_input_batch[:,t,:]
                    x_filtered_t, reg_t = model.step(y_t,u_t)
                    if torch.isnan(x_filtered_t).any():
                            raise ValueError(f"NaN in x_filtered_t at sample {j}, step {t}")
                    current_trajectory_x_hats.append(x_filtered_t)
                    current_trajectory_regs.append(reg_t)
                all_trajectories_for_ensemble.append(torch.stack(current_trajectory_x_hats, dim=1))
                all_regs_for_ensemble.append(torch.sum(torch.stack(current_trajectory_regs)))

            ensemble_trajectories = torch.stack(all_trajectories_for_ensemble, dim=0)
            x_hat_sequence = ensemble_trajectories.mean(dim=0)
            cov_diag_sequence = ensemble_trajectories.var(dim=0)
            regularization_loss = torch.stack(all_regs_for_ensemble).mean()
            target_sequence = x_true_batch[:, 1:, :]
            diff = x_hat_sequence - target_sequence
            
            # 1. Check for "absolute zero" (exact match)
            # If the sum of absolute differences is smaller than epsilon, it is suspicious (Data Leakage)
            is_perfect_match = (diff.abs().sum() < 1e-9).item()
            
            
            mean_error = diff.abs().mean().item()
            
            # 3. Check for "Variance Collapse" (whether all samples match)
            # If the variance is zero, NLL explodes.
            min_variance = cov_diag_sequence.min().item()
            is_zero_variance = (min_variance < 1e-9)

            # OUTPUT (Log when suspicious or regularly)
            if is_perfect_match or is_zero_variance or (train_iter_count % logging_period == 0):
                print(f"\n[DIAGNOSTIKA iter {train_iter_count}]")
                print(f"  -> Difference (Pred - Target): {mean_error:.6f}")
                print(f"  -> IS IT EXACTLY ZERO? {'!!! YES (SUSPICIOUS) !!!' if is_perfect_match else 'No'}")
                print(f"  -> Min Variance v batchi:  {min_variance:.2e}")
                
                if is_zero_variance:
                    print("  !!! WARNING: Variance is 0. NLL Loss will divide by zero -> NaN !!!")
            if train_iter_count < warmup_iterations:
                loss = F.mse_loss(x_hat_sequence, target_sequence) + regularization_loss * 1.0
                nll_loss = torch.tensor(0.0)
            else:
                nll_loss = gaussian_nll_safe(target_sequence, x_hat_sequence, cov_diag_sequence, min_var=1.0e-5,max_error_cap=100.0)
                loss = nll_loss + regularization_loss * 1.0
            
            if torch.isnan(loss): print("Collapse detected (NaN loss)"); done = True; break
            
            loss.backward()
            if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1
            
            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---", f"Total Loss: {loss.item():.4f}", f"NLL: {nll_loss.item():.4f}", f"Reg: {regularization_loss.item():.4f}", f"p1={p1:.3f}, p2={p2:.3f}", sep="\n    - ")

            # --- Validation step ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validation at iteration {train_iter_count} ---")
                model.eval()
                val_mse_list = []
                all_val_x_true_cpu, all_val_x_hat_cpu, all_val_P_hat_cpu = [], [], []

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch, u_input_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        x_true_val_batch = x_true_val_batch.to(device)
                        y_meas_val_batch = y_meas_val_batch.to(device)
                        u_input_val_batch = u_input_val_batch.to(device)
                        val_ensemble_trajectories = []
                        for j in range(J_samples):
                            model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])
                            val_current_x_hats = []
                            for t in range(1, val_seq_len):
                                y_t_val = y_meas_val_batch[:, t, :]
                                u_t_val = u_input_val_batch[:,t,:]
                                x_filtered_t, _ = model.step(y_t_val,u_t_val)
                                val_current_x_hats.append(x_filtered_t)
                            val_ensemble_trajectories.append(torch.stack(val_current_x_hats, dim=1))
                        
                        # Shape: [J, B_val, T_val-1, D_state]
                        val_ensemble = torch.stack(val_ensemble_trajectories, dim=0)
                        
                        # Mean over J gives the final trajectory estimate
                        val_preds_seq = val_ensemble.mean(dim=0)
                        
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())
                        
                        initial_state_val = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state_val, val_preds_seq], dim=1)
                        
                        diff = val_ensemble - val_preds_seq.unsqueeze(0)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=0) # Mean over J (dimension 0)

                        P0 = model.system_model.P0.unsqueeze(0).repeat(val_batch_size, 1, 1).unsqueeze(1)
                        full_P_hat = torch.cat([P0, val_covs_full], dim=1)
                        
                        all_val_x_true_cpu.append(x_true_val_batch.cpu())
                        all_val_x_hat_cpu.append(full_x_hat.cpu())
                        all_val_P_hat_cpu.append(full_P_hat.cpu())

                avg_val_mse = np.mean(val_mse_list)
                final_x_true_list = torch.cat(all_val_x_true_cpu, dim=0)
                final_x_hat_list = torch.cat(all_val_x_hat_cpu, dim=0)
                final_P_hat_list = torch.cat(all_val_P_hat_cpu, dim=0)
                avg_val_anees = calculate_anees_vectorized(final_x_true_list, final_x_hat_list, final_P_hat_list)
                
                print(f"  Average MSE: {avg_val_mse:.4f}, Average ANEES: {avg_val_anees:.4f}")
                if not np.isnan(avg_val_anees) and avg_val_anees < best_val_anees and avg_val_anees > 0:
                    print("  >>> New best VALIDATION ANEES! Saving model. <<<")
                    best_val_anees = avg_val_anees
                    best_iter_count = train_iter_count
                    score_at_best['val_mse'] = avg_val_mse
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTraining completed.")
    if best_model_state:
        print(f"Loading best model from iteration {best_iter_count} with ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("No best model was saved; returning last state.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_iter": best_iter_count,
        "final_model": model
    }

def train_BayesianKalmanNet(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    warmup_iterations=0, weight_decay_=1e-5
):
    # torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_)
    best_val_anees = float('inf')
    score_at_best = {"val_nll": 0.0, "val_mse": 0.0}
    best_iter_count = 0
    best_model_state = None
    train_iter_count = 0
    done = False

    while not done:
        model.train()
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter: done = True; break
            if torch.isnan(x_true_batch).any():
                print(f"!!! SKIP BATCH iter {train_iter_count}: NaN found in x_true (Ground Truth) !!!")
                continue
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            # --- Training ---
            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            
            all_trajectories_for_ensemble = []
            all_regs_for_ensemble = []

            for j in range(J_samples):
                model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])
                current_trajectory_x_hats = []
                current_trajectory_regs = []
                for t in range(1, seq_len):
                    y_t = y_meas_batch[:, t, :]
                    x_filtered_t, reg_t = model.step(y_t)
                    if torch.isnan(x_filtered_t).any():
                            raise ValueError(f"NaN in x_filtered_t at sample {j}, step {t}")
                    current_trajectory_x_hats.append(x_filtered_t)
                    current_trajectory_regs.append(reg_t)
                all_trajectories_for_ensemble.append(torch.stack(current_trajectory_x_hats, dim=1))
                all_regs_for_ensemble.append(torch.sum(torch.stack(current_trajectory_regs)))

            ensemble_trajectories = torch.stack(all_trajectories_for_ensemble, dim=0)
            x_hat_sequence = ensemble_trajectories.mean(dim=0)
            cov_diag_sequence = ensemble_trajectories.var(dim=0)
            regularization_loss = torch.stack(all_regs_for_ensemble).mean()
            target_sequence = x_true_batch[:, 1:, :]
            diff = x_hat_sequence - target_sequence
            
            # 1. Check for "absolute zero" (exact match)
            # If the sum of absolute differences is smaller than epsilon, it is suspicious (Data Leakage)
            is_perfect_match = (diff.abs().sum() < 1e-9).item()
            
            
            mean_error = diff.abs().mean().item()
            
            # 3. Check for "Variance Collapse" (whether all samples match)
            # If the variance is zero, NLL explodes.
            min_variance = cov_diag_sequence.min().item()
            is_zero_variance = (min_variance < 1e-9)

            # OUTPUT (Log when suspicious or regularly)
            if is_perfect_match or is_zero_variance or (train_iter_count % logging_period == 0):
                print(f"\n[DIAGNOSTIKA iter {train_iter_count}]")
                print(f"  -> Difference (Pred - Target): {mean_error:.6f}")
                print(f"  -> IS IT EXACTLY ZERO? {'!!! YES (SUSPICIOUS) !!!' if is_perfect_match else 'No'}")
                print(f"  -> Min Variance v batchi:  {min_variance:.2e}")
                
                if is_zero_variance:
                    print("  !!! WARNING: Variance is 0. NLL Loss will divide by zero -> NaN !!!")
            if train_iter_count < warmup_iterations:
                loss = F.mse_loss(x_hat_sequence, target_sequence) + regularization_loss * 1.0
                nll_loss = gaussian_nll_safe(target_sequence, x_hat_sequence, cov_diag_sequence, min_var=1.0e-5,max_error_cap=100.0) 
            else:
                nll_loss = gaussian_nll_safe(target_sequence, x_hat_sequence, cov_diag_sequence, min_var=1.0e-5,max_error_cap=100.0) 
                loss = nll_loss + regularization_loss * 1.0
            
            if torch.isnan(loss): print("Collapse detected (NaN loss)"); done = True; break
            
            loss.backward()
            if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1
            
            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---", f"Total Loss: {loss.item():.4f}", f"NLL: {nll_loss.item():.4f}", f"Reg: {regularization_loss.item():.4f}", f"p1={p1:.3f}, p2={p2:.3f}", sep="\n    - ")

            # --- Validation step ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validation at iteration {train_iter_count} ---")
                model.eval()
                val_mse_list = []
                all_val_x_true_cpu, all_val_x_hat_cpu, all_val_P_hat_cpu = [], [], []

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        x_true_val_batch = x_true_val_batch.to(device)
                        y_meas_val_batch = y_meas_val_batch.to(device)
                        val_ensemble_trajectories = []
                        for j in range(J_samples):
                            model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])
                            val_current_x_hats = []
                            for t in range(1, val_seq_len):
                                y_t_val = y_meas_val_batch[:, t, :]
                                x_filtered_t, _ = model.step(y_t_val)
                                val_current_x_hats.append(x_filtered_t)
                            val_ensemble_trajectories.append(torch.stack(val_current_x_hats, dim=1))
                        
                        # Shape: [J, B_val, T_val-1, D_state]
                        val_ensemble = torch.stack(val_ensemble_trajectories, dim=0)
                        
                        # Mean over J gives the final trajectory estimate
                        val_preds_seq = val_ensemble.mean(dim=0)
                        
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())
                        
                        initial_state_val = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state_val, val_preds_seq], dim=1)
                        
                        diff = val_ensemble - val_preds_seq.unsqueeze(0)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=0) # Mean over J (dimension 0)

                        P0 = model.system_model.P0.unsqueeze(0).repeat(val_batch_size, 1, 1).unsqueeze(1)
                        full_P_hat = torch.cat([P0, val_covs_full], dim=1)
                        
                        all_val_x_true_cpu.append(x_true_val_batch.cpu())
                        all_val_x_hat_cpu.append(full_x_hat.cpu())
                        all_val_P_hat_cpu.append(full_P_hat.cpu())

                avg_val_mse = np.mean(val_mse_list)
                final_x_true_list = torch.cat(all_val_x_true_cpu, dim=0)
                final_x_hat_list = torch.cat(all_val_x_hat_cpu, dim=0)
                final_P_hat_list = torch.cat(all_val_P_hat_cpu, dim=0)
                avg_val_anees = calculate_anees_vectorized(final_x_true_list, final_x_hat_list, final_P_hat_list)
                
                print(f"  Average MSE: {avg_val_mse:.4f}, Average ANEES: {avg_val_anees:.4f}")
                if not np.isnan(avg_val_anees) and avg_val_anees < best_val_anees and avg_val_anees > 0:
                    print("  >>> New best VALIDATION ANEES! Saving model. <<<")
                    best_val_anees = avg_val_anees
                    best_iter_count = train_iter_count
                    score_at_best['val_mse'] = avg_val_mse
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTraining completed.")
    if best_model_state:
        print(f"Loading best model from iteration {best_iter_count} with ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("No best model was saved; returning last state.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_iter": best_iter_count,
        "final_model": model
    }


def train_state_KalmanNetNCLT(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20, 
                          optimizer_type=torch.optim.AdamW, weight_decay=1e-5,
                          print_gradient=False, add_noise_to_initial_state=False, init_noise_std=None):
    """
    Training function optimized for CPU/Integrated GPU.
    - Removed CUDA Mixed Precision.
    - Added detailed logging (Pos vs Vel Loss) for each epoch.
    """
    
    # 1. Definition of Loss functions
    criterion = nn.MSELoss() 
    metric_mse = nn.MSELoss()
    
    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Detection of whether the model returns covariance (for trace logging)
    if not hasattr(model, 'returns_covariance'):
        model.returns_covariance = False 
    returns_covariance = model.returns_covariance

    # Default noise for start, if not specified
    if init_noise_std is None:
        # [px, py, vx, vy, theta, omega]
        init_noise_std = torch.tensor([5.0, 5.0, 2.0, 2.0, 0.5, 0.1], device=device)
    else:
        init_noise_std = init_noise_std.to(device)

    print(f"🚀 Starting training (Device: {device}, Batches: {len(train_loader)})...")

    for epoch in range(epochs):
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        model.train()
        
        # Accumulators for logging epoch averages
        epoch_train_loss = 0.0
        epoch_pos_loss = 0.0
        epoch_vel_loss = 0.0
        epoch_traces = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpacking data (x=Ground Truth, y=Measurements, u=Control Input)
            if len(batch_data) == 3:
                x_true, y_meas, u_input = batch_data
            else:
                raise ValueError("DataLoader must return (x, y, u)!")

            x_true = x_true.to(device)
            y_meas = y_meas.to(device)
            u_input = u_input.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true.shape
            
            # A) Inicializace stavu
            initial_state = x_true[:, 0, :]
            if add_noise_to_initial_state:
                # Adding noise for robustness against poor initialization
                noise = torch.randn_like(initial_state) * init_noise_std
                model.reset(batch_size=batch_size, initial_state=initial_state + noise)
            else:    
                model.reset(batch_size=batch_size, initial_state=initial_state)

            predictions_x = []
            predictions_P = []

            # B) Loop over the time sequence (Forward pass)
            for t in range(1, seq_len):
                y_t = y_meas[:, t, :]
                u_t = u_input[:, t, :] # Vstup pro predikci
                
                step_output = model.step(y_t, u_t)
                
                if returns_covariance:
                    x_filt, P_filt = step_output
                    predictions_P.append(P_filt)
                else:
                    x_filt = step_output

                predictions_x.append(x_filt)

            # Trajectory assembly
            pred_traj = torch.stack(predictions_x, dim=1)
            target_traj = x_true[:, 1:, :] 

            # C) Loss computation
            diff = pred_traj - target_traj
            
            # MSE pro pozici [px, py] (indexy 0, 1)
            loss_pos = torch.mean(diff[:, :, :2]**2) 
            # MSE pro rychlost [vx, vy] (indexy 2, 3)
            loss_vel = torch.mean(diff[:, :, 2:4]**2) 
            
            # Total weighted loss (Important: tune the weights according to what you want to prioritize)
            # Here we give 100x weight to position (since it is in meters) and 10x to velocity
            loss = criterion(pred_traj, target_traj)

            # D) Backward pass
            loss.backward()

            # E) Gradient clipping (Necessary for recurrent networks/Transformers)
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            optimizer.step()
            
            # F) Logging within the batch
            epoch_train_loss += loss.item()
            epoch_pos_loss += loss_pos.item()
            epoch_vel_loss += loss_vel.item()
            
            if returns_covariance and predictions_P:
                with torch.no_grad():
                     # Computing the trace of the covariance matrix (for uncertainty info)
                     cov_stack = torch.stack(predictions_P, dim=1)
                     trace = torch.mean(torch.diagonal(cov_stack, dim1=-2, dim2=-1).sum(-1)).item()
                     epoch_traces.append(trace)

            # Optional: Print gradient for the first batch (debug)
            if print_gradient and batch_idx == 0:
                # Try to find weights for gradient printing
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        break
                print(f" [DEBUG] Epoch {epoch+1} Grad Norm: {grad_norm:.4f}")

        # Epoch averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_pos_loss = epoch_pos_loss / len(train_loader)
        avg_vel_loss = epoch_vel_loss / len(train_loader)
        avg_trace = np.mean(epoch_traces) if epoch_traces else 0.0

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0
        
        with torch.no_grad():
            for val_data in val_loader:
                x_val, y_val, u_val = val_data
                x_val, y_val, u_val = x_val.to(device), y_val.to(device), u_val.to(device)
                
                batch_size_val, seq_len_val, _ = x_val.shape
                model.reset(batch_size=batch_size_val, initial_state=x_val[:,0,:])
                
                val_preds = []
                for t in range(1, seq_len_val):
                    out = model.step(y_val[:, t, :], u_val[:, t, :])
                    val_preds.append(out[0] if returns_covariance else out)
                    
                pred_traj_val = torch.stack(val_preds, dim=1)
                target_traj_val = x_val[:, 1:, :]
                
                # Val Loss (same metric as train for comparison)
                diff_val = pred_traj_val - target_traj_val
                loss_pos_val = torch.mean(diff_val[:, :, :2]**2)
                loss_vel_val = torch.mean(diff_val[:, :, 2:4]**2)
                
                val_loss_weighted = criterion(pred_traj_val, target_traj_val)
                epoch_val_loss += val_loss_weighted.item()
                
                # Pure MSE (for objective comparison)
                epoch_val_mse += metric_mse(pred_traj_val, target_traj_val).item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        
        # ==========================
        # 3. LOGGING & SAVING
        # ==========================
        
        # Formatted output
        trace_str = f" | CovTrace: {avg_trace:.2f}" if returns_covariance else ""
        print(f"Ep [{epoch+1}/{epochs}] "
              f"Train: {avg_train_loss:.4f} (Pos: {avg_pos_loss:.2f}, Vel: {avg_vel_loss:.2f}) | "
              f"Val: {avg_val_loss:.4f} (MSE: {avg_val_mse:.4f}){trace_str}")
        
        # Saving the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            print(f"    -> New Best Model (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⏹ Early stopping triggered after {epoch + 1} epochs.")
            break
            
    print("Training finished.")
    if best_model_state:
        print(f"Loading best model with Val Loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model


def train_BayesianKalmanNetNCLT_test(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    warmup_iterations=0, weight_decay_=1e-5, optimizer_type=torch.optim.AdamW
):
    # torch.autograd.set_detect_anomaly(True)
    optimizer = optimizer_type(model.parameters(), lr=learning_rate, weight_decay=weight_decay_)
    best_val_anees = float('inf')
    score_at_best = {"val_nll": 0.0, "val_mse": 0.0}
    best_iter_count = 0
    best_model_state = None
    train_iter_count = 0
    done = False

    while not done:
        model.train()
        for x_true_batch, y_meas_batch, u_input_batch in train_loader:
            if train_iter_count >= total_train_iter: done = True; break
            if torch.isnan(x_true_batch).any():
                print(f"!!! SKIP BATCH iter {train_iter_count}: NaN found in x_true (Ground Truth) !!!")
                continue
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            u_input_batch = u_input_batch.to(device)
            # --- Training ---
            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            
            all_trajectories_for_ensemble = []
            all_regs_for_ensemble = []

            for j in range(J_samples):
                model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])
                current_trajectory_x_hats = []
                current_trajectory_regs = []
                for t in range(1, seq_len):
                    y_t = y_meas_batch[:, t, :]
                    u_t = u_input_batch[:,t,:]
                    x_filtered_t, reg_t = model.step(y_t,u_t)
                    if torch.isnan(x_filtered_t).any():
                            raise ValueError(f"NaN in x_filtered_t at sample {j}, step {t}")
                    current_trajectory_x_hats.append(x_filtered_t)
                    current_trajectory_regs.append(reg_t)
                all_trajectories_for_ensemble.append(torch.stack(current_trajectory_x_hats, dim=1))
                all_regs_for_ensemble.append(torch.sum(torch.stack(current_trajectory_regs)))

            ensemble_trajectories = torch.stack(all_trajectories_for_ensemble, dim=0)
            x_hat_sequence = ensemble_trajectories.mean(dim=0)
            cov_diag_sequence = ensemble_trajectories.var(dim=0)
            regularization_loss = torch.stack(all_regs_for_ensemble).mean()
            target_sequence = x_true_batch[:, 1:, :]
            diff = x_hat_sequence - target_sequence
            
            # 1. Check for "absolute zero" (exact match)
            # If the sum of absolute differences is smaller than epsilon, it is suspicious (Data Leakage)
            is_perfect_match = (diff.abs().sum() < 1e-9).item()
            
            
            mean_error = diff.abs().mean().item()
            
            # 3. Check for "Variance Collapse" (whether all samples match)
            # If the variance is zero, NLL explodes.
            mean_variance = cov_diag_sequence.mean().item()
            min_variance = cov_diag_sequence.min().item()
            
            
            # We only care if the MEAN variance is zero (the whole model collapsed)
            if mean_variance < 1e-9:
                 print(f"!!! WARNING: Model collapsed (Mean Var={mean_variance:.2e}) !!!")
            is_zero_variance = (min_variance < 1e-9)

            # OUTPUT (Log when suspicious or periodically)
            if is_perfect_match or is_zero_variance or (train_iter_count % logging_period == 0):
                print(f"\n[DIAGNOSTIKA iter {train_iter_count}]")
                print(f"  -> Difference (Pred - Target): {mean_error:.6f}")
                print(f"  -> Min Variance v batchi:  {min_variance:.2e}")
                print(f"  -> Mean Variance v batchi: {mean_variance:.2e}")


            if train_iter_count < warmup_iterations:
                loss = F.mse_loss(x_hat_sequence, target_sequence) + regularization_loss * 1.0
                nll_loss = gaussian_nll_safe(target_sequence, x_hat_sequence, cov_diag_sequence, min_var=1.0e-5,max_error_cap=100.0)
            else:
                nll_loss = gaussian_nll_safe(target_sequence, x_hat_sequence, cov_diag_sequence, min_var=1.0e-5,max_error_cap=100.0)
                loss = nll_loss + regularization_loss * 1.0
            
            if torch.isnan(loss): print("Collapse detected (NaN loss)"); done = True; break
            
            loss.backward()
            if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1
            
            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---", f"Total Loss: {loss.item():.4f}", f"NLL: {nll_loss.item():.4f}", f"Reg: {regularization_loss.item():.4f}", f"p1={p1:.3f}, p2={p2:.3f}", sep="\n    - ")

            # --- Validation step ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validation at iteration {train_iter_count} ---")
                model.eval()
                val_mse_list = []
                all_val_x_true_cpu, all_val_x_hat_cpu, all_val_P_hat_cpu = [], [], []

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch, u_input_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        x_true_val_batch = x_true_val_batch.to(device)
                        y_meas_val_batch = y_meas_val_batch.to(device)
                        u_input_val_batch = u_input_val_batch.to(device)
                        val_ensemble_trajectories = []
                        for j in range(J_samples):
                            model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])
                            val_current_x_hats = []
                            for t in range(1, val_seq_len):
                                y_t_val = y_meas_val_batch[:, t, :]
                                u_t_val = u_input_val_batch[:,t,:]
                                x_filtered_t, _ = model.step(y_t_val,u_t_val)
                                val_current_x_hats.append(x_filtered_t)
                            val_ensemble_trajectories.append(torch.stack(val_current_x_hats, dim=1))
                        
                        # Shape: [J, B_val, T_val-1, D_state]
                        val_ensemble = torch.stack(val_ensemble_trajectories, dim=0)
                        
                        # Mean over J gives the final trajectory estimate
                        val_preds_seq = val_ensemble.mean(dim=0)
                        
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())
                        
                        initial_state_val = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state_val, val_preds_seq], dim=1)
                        
                        diff = val_ensemble - val_preds_seq.unsqueeze(0)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=0) + 1e-2 # Mean over J (dimension 0)

                        P0 = model.system_model.P0.unsqueeze(0).repeat(val_batch_size, 1, 1).unsqueeze(1)
                        full_P_hat = torch.cat([P0, val_covs_full], dim=1)
                        
                        all_val_x_true_cpu.append(x_true_val_batch.cpu())
                        all_val_x_hat_cpu.append(full_x_hat.cpu())
                        all_val_P_hat_cpu.append(full_P_hat.cpu())

                avg_val_mse = np.mean(val_mse_list)
                final_x_true_list = torch.cat(all_val_x_true_cpu, dim=0)
                final_x_hat_list = torch.cat(all_val_x_hat_cpu, dim=0)
                final_P_hat_list = torch.cat(all_val_P_hat_cpu, dim=0)
                avg_val_anees = calculate_anees_vectorized(final_x_true_list, final_x_hat_list, final_P_hat_list)
                
                print(f"  Average MSE: {avg_val_mse:.4f}, Average ANEES: {avg_val_anees:.4f}")
                if not np.isnan(avg_val_anees) and avg_val_anees < best_val_anees and avg_val_anees > 0:
                    print("  >>> New best VALIDATION ANEES! Saving model. <<<")
                    best_val_anees = avg_val_anees
                    best_iter_count = train_iter_count
                    score_at_best['val_mse'] = avg_val_mse
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTraining completed.")
    if best_model_state:
        print(f"Loading best model from iteration {best_iter_count} with ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("No best model was saved; returning last state.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_iter": best_iter_count,
        "final_model": model
    }

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def train_kalmanformer_nclt_full_sequence(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-4, clip_grad=1.0, early_stopping_patience=20, 
                          optimizer_type=torch.optim.AdamW, weight_decay=1e-4,
                          print_gradient=False, add_noise_to_initial_state=False, init_noise_std=None):
    """
    Training function for KalmanFormer on full sequences (Full BPTT).
    Compatible with a model that has an internal Rolling Buffer.
    """
    
    # 1. Definition of Loss functions
    criterion = nn.MSELoss() 
    metric_mse = nn.MSELoss()
    
    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Detection of model capabilities
    returns_covariance = getattr(model, 'returns_covariance', False)

    # Default noise for start
    if init_noise_std is None:
        init_noise_std = torch.tensor([5.0, 5.0, 2.0, 2.0, 0.5, 0.1], device=device)
    else:
        init_noise_std = init_noise_std.to(device)

    print(f"🚀 Starting Full-Sequence training (Len: 100, Device: {device})...")

    for epoch in range(epochs):
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        model.train()
        
        epoch_train_loss = 0.0
        epoch_pos_loss = 0.0
        epoch_vel_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                x_true, y_meas, u_input = batch_data
            else:
                raise ValueError("DataLoader must return (x, y, u)!")

            x_true = x_true.to(device)
            y_meas = y_meas.to(device)
            u_input = u_input.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true.shape
            
            # A) Inicializace stavu
            # THIS IS WHERE THE MAGIC HAPPENS: model.reset() clears the Transformer's internal buffer,
            # so we start with a "clean slate" for the new sequence.
            initial_state = x_true[:, 0, :]
            if add_noise_to_initial_state:
                noise = torch.randn_like(initial_state) * init_noise_std
                model.reset(batch_size=batch_size, initial_state=initial_state + noise)
            else:    
                model.reset(batch_size=batch_size, initial_state=initial_state)

            predictions_x = []
            
            # B) Loop over the time sequence (1 to 100)
            # Thanks to our modification of the KalmanFormer class, model.step() automatically:
            # 1. Takes the current measurement.
            # 2. Adds it to the internal buffer (window).
            # 3. Runs the Transformer over the entire window.
            # 4. Returns the estimate for the current time.
            # The trainer therefore doesn't have to "slice" or shift anything.
            for t in range(1, seq_len):
                y_t = y_meas[:, t, :]
                u_t = u_input[:, t, :]
                
                step_output = model.step(y_t, u_t)
                
                if returns_covariance:
                    x_filt, _ = step_output # We ignore P during training, if it is present
                else:
                    x_filt = step_output

                predictions_x.append(x_filt)

            # Trajectory assembly
            pred_traj = torch.stack(predictions_x, dim=1)
            target_traj = x_true[:, 1:, :] 

            # C) Loss computation
            diff = pred_traj - target_traj
            loss_pos = torch.mean(diff[:, :, :2]**2) 
            loss_vel = torch.mean(diff[:, :, 2:4]**2) 
            
            loss = criterion(pred_traj, target_traj)

            
            loss.backward()

            # E) Gradient clipping - CRITICAL FOR TRANSFORMERS
            # With length 100 there is a risk of gradient explosion. Clipping saves it.
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_pos_loss += loss_pos.item()
            epoch_vel_loss += loss_vel.item()
            
            # Debug gradientu
            if print_gradient and batch_idx == 0:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                print(f" [DEBUG] Epoch {epoch+1} Grad Norm: {grad_norm:.4f}")

        # Averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_pos_loss = epoch_pos_loss / len(train_loader)
        avg_vel_loss = epoch_vel_loss / len(train_loader)

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for val_data in val_loader:
                x_val, y_val, u_val = val_data
                x_val, y_val, u_val = x_val.to(device), y_val.to(device), u_val.to(device)
                
                batch_size_val, seq_len_val, _ = x_val.shape
                
                # Reset also for validation (so the buffer contains no training data)
                model.reset(batch_size=batch_size_val, initial_state=x_val[:,0,:])
                
                val_preds = []
                for t in range(1, seq_len_val):
                    out = model.step(y_val[:, t, :], u_val[:, t, :])
                    val_preds.append(out[0] if returns_covariance else out)
                    
                pred_traj_val = torch.stack(val_preds, dim=1)
                target_traj_val = x_val[:, 1:, :]
                
                val_loss_weighted = criterion(pred_traj_val, target_traj_val)
                epoch_val_loss += val_loss_weighted.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # ==========================
        # 3. LOGGING
        # ==========================
        print(f"Ep [{epoch+1}/{epochs}] "
              f"Train: {avg_train_loss:.4f} (Pos: {avg_pos_loss:.2f}, Vel: {avg_vel_loss:.2f}) | "
              f"Val: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            print(f"    -> New Best Model (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⏹ Early stopping activated after {epoch + 1} epochs.")
            break
            
    print("Training finished.")
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model


def train_state_KalmanNetTAN(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20, 
                          optimizer_type=torch.optim.AdamW, weight_decay=1e-5,
                          print_gradient=False, add_noise_to_initial_state=False, init_noise_std=None):
    """
    Robust training function for KalmanNet with Huber Loss (prevention of gradient explosion).
    """
    
    
    # Delta=10.0 means: Errors up to 10 meters are penalized quadratically (accuracy), 
    # errors above 10 meters linearly (robustness against divergence).
    # criterion = nn.HuberLoss(delta=10.0) 
    criterion = nn.MSELoss()
    # Metric for human reading (MSE, so we can see the real variance)
    metric_mse = nn.MSELoss()
    
    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Error: Model does not define attribute 'returns_covariance'.")
        
    returns_covariance = model.returns_covariance
    
    if init_noise_std is None:
        init_noise_std = torch.tensor([5.0, 5.0, 2.0, 2.0], device=device)
    else:
        init_noise_std = init_noise_std.to(device)

    print(f"Start training on {device}...")

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        epoch_traces = []
        
        # Pro statistiky
        train_pos_mse_sum = 0.0
        train_vel_mse_sum = 0.0

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            
            # State initialization (with optional noise)
            initial_state = x_true_batch[:, 0, :]
            if add_noise_to_initial_state:
                noise = torch.randn_like(initial_state) * init_noise_std
                noisy_initial_state = initial_state.clone() + noise
                model.reset(batch_size=batch_size, initial_state=noisy_initial_state)
            else:    
                model.reset(batch_size=batch_size, initial_state=initial_state)

            predictions_x = []
            predictions_P = []

            # Forward pass in time
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
            target_trajectory = x_true_batch[:, 1:, :]

            # Collecting the average covariance trace (for diagnostics)
            if returns_covariance and predictions_P:
                predicted_cov_trajectory = torch.stack(predictions_P, dim=1)
                # Computing the trace of the covariance matrix P
                current_trace = torch.mean(torch.sum(torch.diagonal(predicted_cov_trajectory.detach(), offset=0, dim1=-2, dim2=-1), dim=-1)).item()
                epoch_traces.append(current_trace)

            # --- Loss Calculation ---
            loss = criterion(predicted_trajectory, target_trajectory)
            loss.backward()

            # Statistics (for output, not for gradient)
            with torch.no_grad():
                diff = predicted_trajectory - target_trajectory
                pos_mse = torch.mean(diff[:, :, :2]**2).item()
                vel_mse = torch.mean(diff[:, :, 2:]**2).item()
                train_pos_mse_sum += pos_mse
                train_vel_mse_sum += vel_mse

            if print_gradient:
                print(f"Grad: {model.dnn.input_layer[0].weight.grad.abs().mean().item()}")
            
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_pos_mse = train_pos_mse_sum / len(train_loader)
        avg_vel_mse = train_vel_mse_sum / len(train_loader)
        avg_epoch_trace = np.mean(epoch_traces) if epoch_traces else 0.0

        # --- Validation phase ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0
        
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                batch_size_val, seq_len_val, _ = x_true_val.shape

                initial_state = x_true_val[:,0,:]
                model.reset(batch_size=batch_size_val, initial_state=initial_state) # Validation without init state noise!
                
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
                target_val_trajectory = x_true_val[:, 1:, :]

                # Loss
                val_loss_batch = criterion(predicted_val_trajectory, target_val_trajectory)
                epoch_val_loss += val_loss_batch.item()
                
                # Clean MSE for the report
                mse_batch = metric_mse(predicted_val_trajectory, target_val_trajectory)
                epoch_val_mse += mse_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        
        # --- Logging ---
        # 1. First we build the base log
        log_str = f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
        
        # 2. Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            save_msg = " -> New Best!"
        else:
            epochs_no_improve += 1
            save_msg = ""

        # 3. Periodic output (there was a BUG here - log_message did not exist)
        if (epoch + 1) % 5 == 0:
            detailed_msg = f"{log_str} | PosMSE: {avg_pos_mse:.1f}, VelMSE: {avg_vel_mse:.1f}"
            if returns_covariance:
                detailed_msg += f" | CovTrace: {avg_epoch_trace:.1f}"
            print(detailed_msg + save_msg)
        
        # Output only on save (if not periodic)
        elif save_msg: 
             print(log_str + save_msg)

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
            
    print("Training completed.")
    if best_model_state:
        print(f"Loading best model with validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model


def train_state_KalmanNet_sliding_windowTAN(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20,
                          tbptt_k=2, tbptt_w=10,optimizer_=torch.optim.Adam, weight_decay_=1e-4, criterion_=nn.MSELoss()):
    criterion = criterion_
    metric_mse = nn.MSELoss()
    optimizer = optimizer_(model.parameters(), lr=lr, weight_decay=weight_decay_)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Error: Model does not define attribute 'returns_covariance'.")
    if not hasattr(model, '_detach'):
            raise AttributeError("Error: Model does not implement method `_detach()`, "
                                 "which is required for TBPTT(k,w,D).")
        
    returns_covariance = model.returns_covariance
    print(f"INFO: Detected from model attribute that it returns covariance: {returns_covariance}")
    print(f"INFO: Starting training with TBPTT(k={tbptt_k}, w={tbptt_w})")

    # if init_noise_std is None:
    #     init_noise_std = torch.tensor([5.0, 5.0, 2.0, 2.0], device=device)
    # else:
    #     init_noise_std = init_noise_std.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            batch_size, seq_len, _ = x_true_batch.shape # seq_len is 'D' from the paper
            
            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

            total_loss_for_batch = 0.0
            num_windows = 0

            # Loop over the entire sequence in windows of 'w'
            for t_start in range(1, seq_len, tbptt_w):
                t_end = min(t_start + tbptt_w, seq_len)
                window_len = t_end - t_start
                
                if window_len == 0:
                    continue
                    
                predictions_x = []

                # 1. Forward pass over window 'w'
                for t in range(t_start, t_end):
                    y_t = y_meas_batch[:, t, :]
                    step_output = model.step(y_t)
                    
                    x_filtered_t = step_output[0] if returns_covariance else step_output
                    predictions_x.append(x_filtered_t)
                    
                    # Detach gradients every 'k' steps
                    if (t - t_start + 1) % tbptt_k == 0:
                        model._detach()

                # 3. Detach at window end (before backward())
                model._detach()
                
                # 4. Compute loss ONLY for this window (HYBRID)
                predicted_window = torch.stack(predictions_x, dim=1)
                true_window = x_true_batch[:, t_start:t_end, :]
                
                loss = criterion(predicted_window, true_window)

                # 5. Backward pass and weight update (per window)
                optimizer.zero_grad()
                loss.backward()
                
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                total_loss_for_batch += loss.item()                
                num_windows += 1
            
            if num_windows > 0:
                train_loss += (total_loss_for_batch / num_windows)
                # total_windows_epoch += num_windows
        
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mse = 0.0 # For a clean metric

        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                batch_size_val, seq_len_val, _ = x_true_val.shape
                model.reset(batch_size=batch_size_val, initial_state=x_true_val[:, 0, :])
                
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
                target_val_trajectory = x_true_val[:, 1:, :]
                val_loss_batch = criterion(predicted_val_trajectory, target_val_trajectory)
                epoch_val_loss += val_loss_batch.item()
# Validation metric (MSE - for reporting)
                mse_batch = metric_mse(predicted_val_trajectory, target_val_trajectory)
                epoch_val_mse += mse_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        log_str = f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            save_msg = " -> New Best!"
        else:
            epochs_no_improve += 1
            save_msg = ""
            
        if (epoch + 1) % 5 == 0:
             print(f"{log_str} | Val MSE: {avg_val_mse:.2f}{save_msg}")
        elif save_msg:
             print(log_str + save_msg)

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
            
    print("Training completed.")
    if best_model_state:
        print(f"Loading best model with validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model