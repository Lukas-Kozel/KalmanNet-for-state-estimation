from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .losses import hybrid_loss,empirical_loss_mean,empirical_loss_sum,gaussian_nll,gaussian_nll_loss_with_alpha
from .utils import calculate_anees_vectorized
from torch.optim.lr_scheduler import ReduceLROnPlateau

def training_session_single_step_with_hybrid_scaled_training_fcn(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    lambda_mse
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_validation_score = float('inf')
    score_at_best = {"val_nll": 0.0, "val_mse": 0.0, "val_anees": 0.0}
    best_iter_count = 0
    best_model_state = None
    scale_factor = None

    train_iter_count = 0
    done = False

    while not done:
        model.train()
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            initial_state = x_true_batch[:, 0, :]
            model.reset(batch_size=batch_size, initial_state=initial_state)

            all_x_hats, all_covs_diag, all_regs = [], [], []
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_filtered_t, cov_filtered_diag_t, reg_t, _ = model.step(y_t, J_samples=J_samples)
                all_x_hats.append(x_filtered_t)
                all_covs_diag.append(cov_filtered_diag_t)
                all_regs.append(reg_t)

            x_hat_sequence = torch.stack(all_x_hats, dim=1)
            cov_diag_sequence = torch.stack(all_covs_diag, dim=1)
            regularization_loss = torch.sum(torch.stack(all_regs))
            target_sequence = x_true_batch[:, 1:, :]

            nll_term =  gaussian_nll(target_sequence, x_hat_sequence, cov_diag_sequence)
            mse_term = F.mse_loss(x_hat_sequence, target_sequence)

            if scale_factor is None and torch.isfinite(nll_term) and torch.isfinite(mse_term) and mse_term > 0:
                scale_factor = (nll_term.detach() / mse_term.detach())
                print(f"!!! Vypočítán škálovací faktor pro MSE: {scale_factor:.2f} !!!")

            if scale_factor is not None:
                loss = (1 - lambda_mse) * nll_term + lambda_mse * (mse_term * scale_factor) + regularization_loss
            else:
                loss = nll_term + mse_term + regularization_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!!! Kolaps v iteraci {train_iter_count}, ztráta je NaN/Inf. Ukončuji. !!!")
                done = True
                break

            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1

            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---")
                print(f"  Total Scaled Loss: {loss.item():.4f}")
                print(f"    - NLL Component:      {nll_term.item():.4f}")
                print(f"    - MSE Component (raw):  {mse_term.item():.4f}")
                if scale_factor: print(f"    - MSE Scale Factor:     {scale_factor:.2f}")
                print(f"  Dropout Probs: p1={p1:.3f}, p2={p2:.3f}")
                print("-" * 25)

            # --- Validační krok ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validace v iteraci {train_iter_count} ---")
                model.train()
                
                total_val_nll, total_val_mse, total_val_nees_sum, nees_count = 0.0, 0.0, 0.0, 0
                num_val_samples = 0

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])

                        val_all_x_hats_seq, val_all_covs_diag_seq, val_full_ensemble_seq = [], [], []
                        for t in range(1, val_seq_len):
                            y_t_val = y_meas_val_batch[:, t, :]
                            x_filtered_t, cov_diag_t, _, full_ensemble_t = model.step(y_t_val, J_samples=J_samples)
                            val_all_x_hats_seq.append(x_filtered_t)
                            val_all_covs_diag_seq.append(cov_diag_t)
                            val_full_ensemble_seq.append(full_ensemble_t)

                        val_preds_seq = torch.stack(val_all_x_hats_seq, dim=1)
                        val_covs_diag_seq = torch.stack(val_all_covs_diag_seq, dim=1)
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        
                        total_val_nll +=  gaussian_nll(val_target_seq, val_preds_seq, val_covs_diag_seq).item() * val_batch_size
                        total_val_mse += F.mse_loss(val_preds_seq, val_target_seq).item() * val_batch_size
                        num_val_samples += val_batch_size

                        initial_state = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state, val_preds_seq], dim=1)
                        val_ensemble = torch.stack(val_full_ensemble_seq, dim=2)
                        diff = val_ensemble - val_preds_seq.unsqueeze(1)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=2)
                        
                        P0 = model.system_model.P0.unsqueeze(0).repeat(val_batch_size, 1, 1).unsqueeze(1)
                        full_P_hat = torch.cat([P0, val_covs_full], dim=1)
                        
                        jitter = torch.eye(full_P_hat.shape[-1], device=device) * 1e-6
                        for i in range(val_batch_size):
                            for t in range(1, val_seq_len):
                                error = (x_true_val_batch[i, t] - full_x_hat[i, t]).unsqueeze(1)
                                P_t = full_P_hat[i, t]
                                try:
                                    P_inv = torch.inverse(P_t + jitter)
                                    nees_t = (error.T @ P_inv @ error).item()
                                    total_val_nees_sum += nees_t
                                    nees_count += 1
                                except torch.linalg.LinAlgError:
                                    continue
                
                avg_val_nll = total_val_nll / num_val_samples if num_val_samples > 0 else float('nan')
                avg_val_mse = total_val_mse / num_val_samples if num_val_samples > 0 else float('nan')
                avg_val_anees = total_val_nees_sum / nees_count if nees_count > 0 else float('nan')

                w_mse, w_anees = 1.0, 3.0
                anees_penalty = abs(avg_val_anees - model.state_dim)
                validation_score = w_mse * avg_val_mse + w_anees * anees_penalty

                print(f"  Průměrný NLL:        {avg_val_nll:.4f}")
                print(f"  Průměrný MSE:        {avg_val_mse:.4f}")
                print(f"  Průměrný ANEES:      {avg_val_anees:.4f} (cíl: {model.state_dim:.1f})")
                print(f"  VALIDAČNÍ SKÓRE:     {validation_score:.4f} (nižší je lepší)")

                if validation_score < best_validation_score:
                    print(f"  >>> Nové nejlepší VALIDAČNÍ SKÓRE! Ukládám model. <<<")
                    best_validation_score = validation_score
                    best_iter_count = train_iter_count
                    score_at_best['val_nll'] = avg_val_nll
                    score_at_best['val_mse'] = avg_val_mse
                    score_at_best['val_anees'] = avg_val_anees
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model z iterace {best_iter_count} s validačním skóre {best_validation_score:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")

    return {
        "best_validation_score": best_validation_score,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_val_anees": score_at_best['val_anees'],
        "best_iter": best_iter_count,
        "final_model": model
    }


def training_session_single_step_with_gaussian_nll_training_fcn(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            
            # --- Tréninkový krok ---
            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])
            all_x_hats, all_covs_diag, all_regs = [], [], []
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_filtered_t, cov_filtered_diag_t, reg_t, _ = model.step(y_t, J_samples=J_samples)
                all_x_hats.append(x_filtered_t)
                all_covs_diag.append(cov_filtered_diag_t)
                all_regs.append(reg_t)
            x_hat_sequence = torch.stack(all_x_hats, dim=1)
            cov_diag_sequence = torch.stack(all_covs_diag, dim=1)
            regularization_loss = torch.sum(torch.stack(all_regs))
            target_sequence = x_true_batch[:, 1:, :]
            nll_loss = gaussian_nll(target_sequence, x_hat_sequence, cov_diag_sequence)
            loss = nll_loss + regularization_loss
            if torch.isnan(loss): print("!!! Kolaps !!!"); done = True; break
            loss.backward()
            if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1
            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---", f"Total Loss: {loss.item():.4f}", f"NLL: {nll_loss.item():.4f}", f"Reg: {regularization_loss.item():.4f}", f"p1={p1:.3f}, p2={p2:.3f}", sep="\n    - ")

            # --- Validační krok ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validace v iteraci {train_iter_count} ---")
                model.train()
                all_val_x_true_cpu, all_val_x_hat_cpu, all_val_P_hat_cpu = [], [], []
                val_nll_list, val_mse_list = [], []
                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])
                        val_all_x_hats_seq, val_full_ensemble_seq = [], []
                        for t in range(1, val_seq_len):
                            y_t_val = y_meas_val_batch[:, t, :] 
                            x_filtered_t, _, _, full_ensemble_t = model.step(y_t_val, J_samples=J_samples)
                            val_all_x_hats_seq.append(x_filtered_t)
                            val_full_ensemble_seq.append(full_ensemble_t)
                        val_preds_seq = torch.stack(val_all_x_hats_seq, dim=1)
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())
                        initial_state_val = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state_val, val_preds_seq], dim=1)
                        val_ensemble = torch.stack(val_full_ensemble_seq, dim=2)
                        diff = val_ensemble - val_preds_seq.unsqueeze(1)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=1)
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
                print(f"  Průměrný MSE: {avg_val_mse:.4f}, Průměrný ANEES: {avg_val_anees:.4f}")
                if not np.isnan(avg_val_anees) and avg_val_anees < best_val_anees:
                    print("  >>> Nové nejlepší VALIDAČNÍ ANEES! Ukládám model. <<<")
                    best_val_anees = avg_val_anees
                    best_iter_count = train_iter_count
                    score_at_best['val_mse'] = avg_val_mse
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model z iterace {best_iter_count} s ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_iter": best_iter_count,
        "final_model": model
    }


def training_session_single_step_with_empirical_loss_sum(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    final_beta
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_anees = float('inf')
    score_at_best = {"val_loss": 0.0, "val_mse": 0.0}
    best_iter_count = 0
    best_model_state = None

    train_iter_count = 0
    done = False

    while not done:
        model.train()
        for x_true_batch, y_meas_batch in train_loader:
            if train_iter_count >= total_train_iter:
                done = True
                break

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape
            initial_state = x_true_batch[:, 0, :]
            model.reset(batch_size=batch_size, initial_state=initial_state)

            all_x_hats, all_covs_diag, all_regs = [], [], []
            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                x_filtered_t, cov_filtered_diag_t, reg_t, _ = model.step(y_t, J_samples=J_samples)
                all_x_hats.append(x_filtered_t)
                all_covs_diag.append(cov_filtered_diag_t)
                all_regs.append(reg_t)

            x_hat_sequence = torch.stack(all_x_hats, dim=1)
            cov_diag_sequence = torch.stack(all_covs_diag, dim=1)
            regularization_loss = torch.sum(torch.stack(all_regs))
            target_sequence = x_true_batch[:, 1:, :]

            beta = final_beta * (train_iter_count / total_train_iter)
            
            data_loss, mse_component, var_loss_component =  empirical_loss_sum(
                target=target_sequence,
                predicted_mean=x_hat_sequence,
                predicted_var=cov_diag_sequence,
                beta=beta
            )
            
            loss = data_loss + regularization_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"!!! Kolaps v iteraci {train_iter_count}, ztráta je NaN/Inf. Ukončuji. !!!")
                done = True
                break

            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1

            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---")
                print(f"  Total Loss: {loss.item():.4f}")
                print(f"    - Data Loss:          {data_loss.item():.4f} (beta={beta:.3f})")
                print(f"    - L1 (MSE) comp:      {mse_component.item():.4f}")
                print(f"    - L2 (Var Sum) comp:  {var_loss_component.item():.4f}")
                print(f"    - Regularization:     {regularization_loss.item():.4f}")
                print(f"  Dropout Probs: p1={p1:.3f}, p2={p2:.3f}")
                print("-" * 25)

            # --- Validační krok ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validace v iteraci {train_iter_count} ---")
                model.train()
                
                val_x_true_list, val_x_hat_list, val_P_hat_list = [], [], []
                val_loss_list = []
                val_mse_list = []

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])

                        val_all_x_hats_seq, val_all_covs_diag_seq, val_full_ensemble_seq = [], [], []
                        for t in range(1, val_seq_len):
                            y_t_val = y_meas_val_batch[:, t, :]
                            x_filtered_t, cov_diag_t, _, full_ensemble_t = model.step(y_t_val, J_samples=J_samples)
                            val_all_x_hats_seq.append(x_filtered_t)
                            val_all_covs_diag_seq.append(cov_diag_t)
                            val_full_ensemble_seq.append(full_ensemble_t)

                        val_preds_seq = torch.stack(val_all_x_hats_seq, dim=1)
                        val_covs_diag_seq = torch.stack(val_all_covs_diag_seq, dim=1)
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        
                        val_data_loss, _, _ =  empirical_loss_sum(val_target_seq, val_preds_seq, val_covs_diag_seq, beta)
                        val_loss_list.append(val_data_loss.item())
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())

                        initial_state = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state, val_preds_seq], dim=1)
                        val_ensemble = torch.stack(val_full_ensemble_seq, dim=2)
                        diff = val_ensemble - val_preds_seq.unsqueeze(1)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=2)
                        
                        P0 = model.system_model.P0.unsqueeze(0).repeat(val_batch_size, 1, 1).unsqueeze(1)
                        full_P_hat = torch.cat([P0, val_covs_full], dim=1)
                        
                        for i in range(val_batch_size):
                            val_x_true_list.append(x_true_val_batch[i].cpu())
                            val_x_hat_list.append(full_x_hat[i].cpu())
                            val_P_hat_list.append(full_P_hat[i].cpu())

                avg_val_loss = np.mean(val_loss_list)
                avg_val_mse = np.mean(val_mse_list)
                avg_val_anees = calculate_anees_vectorized(val_x_true_list, val_x_hat_list, val_P_hat_list)
                
                print(f"  Průměrná data Loss:  {avg_val_loss:.4f}")
                print(f"  Průměrný MSE:        {avg_val_mse:.4f}")
                print(f"  Průměrný ANEES:      {avg_val_anees:.4f} (cíl: {model.state_dim:.1f})")

                if avg_val_anees < best_val_anees:
                    print(f"  >>> Nové nejlepší VALIDAČNÍ ANEES! Ukládám model. <<<")
                    best_val_anees = avg_val_anees
                    best_iter_count = train_iter_count
                    score_at_best['val_loss'] = avg_val_loss
                    score_at_best['val_mse'] = avg_val_mse
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model z iterace {best_iter_count} s ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_loss": score_at_best['val_loss'],
        "best_val_mse": score_at_best['val_mse'],
        "best_iter": best_iter_count,
        "final_model": model
    }


def training_session_trajectory_with_gaussian_nll_training_fcn(
    model, train_loader, val_loader, device,
    total_train_iter, learning_rate, clip_grad,
    J_samples, validation_period, logging_period,
    warmup_iterations=0
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            
            # --- Trénink---
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
            
            if torch.isnan(loss): print("!!! Kolaps !!!"); done = True; break
            
            loss.backward()
            if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_iter_count += 1
            
            if train_iter_count % logging_period == 0:
                with torch.no_grad():
                    p1 = torch.sigmoid(model.dnn.concrete_dropout1.p_logit).item()
                    p2 = torch.sigmoid(model.dnn.concrete_dropout2.p_logit).item()
                print(f"--- Iteration [{train_iter_count}/{total_train_iter}] ---", f"Total Loss: {loss.item():.4f}", f"NLL: {nll_loss.item():.4f}", f"Reg: {regularization_loss.item():.4f}", f"p1={p1:.3f}, p2={p2:.3f}", sep="\n    - ")

            # --- Validační krok ---
            if train_iter_count > 0 and train_iter_count % validation_period == 0:
                print(f"\n--- Validace v iteraci {train_iter_count} ---")
                model.eval()
                val_mse_list = []
                all_val_x_true_cpu, all_val_x_hat_cpu, all_val_P_hat_cpu = [], [], []

                with torch.no_grad():
                    for x_true_val_batch, y_meas_val_batch in val_loader:
                        val_batch_size, val_seq_len, _ = x_true_val_batch.shape
                        
                        val_ensemble_trajectories = []
                        for j in range(J_samples):
                            model.reset(batch_size=val_batch_size, initial_state=x_true_val_batch[:, 0, :])
                            val_current_x_hats = []
                            for t in range(1, val_seq_len):
                                y_t_val = y_meas_val_batch[:, t, :]
                                x_filtered_t, _ = model.step(y_t_val)
                                val_current_x_hats.append(x_filtered_t)
                            val_ensemble_trajectories.append(torch.stack(val_current_x_hats, dim=1))
                        
                        # Tvar: [J, B_val, T_val-1, D_state]
                        val_ensemble = torch.stack(val_ensemble_trajectories, dim=0)
                        
                        # Průměr přes J dává finální odhad trajektorie
                        val_preds_seq = val_ensemble.mean(dim=0)
                        
                        val_target_seq = x_true_val_batch[:, 1:, :]
                        val_mse_list.append(F.mse_loss(val_preds_seq, val_target_seq).item())
                        
                        initial_state_val = x_true_val_batch[:, 0, :].unsqueeze(1)
                        full_x_hat = torch.cat([initial_state_val, val_preds_seq], dim=1)
                        
                        diff = val_ensemble - val_preds_seq.unsqueeze(0)
                        outer_prods = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
                        val_covs_full = outer_prods.mean(dim=0) # Průměr přes J (dimenze 0)

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
                
                print(f"  Průměrný MSE: {avg_val_mse:.4f}, Průměrný ANEES: {avg_val_anees:.4f}")
                if not np.isnan(avg_val_anees) and avg_val_anees < best_val_anees and avg_val_anees > 0:
                    print("  >>> Nové nejlepší VALIDAČNÍ ANEES! Ukládám model. <<<")
                    best_val_anees = avg_val_anees
                    best_iter_count = train_iter_count
                    score_at_best['val_mse'] = avg_val_mse
                    best_model_state = deepcopy(model.state_dict())
                print("-" * 50)
                model.train()

    print("\nTrénování dokončeno.")
    if best_model_state:
        print(f"Načítám nejlepší model z iterace {best_iter_count} s ANEES {best_val_anees:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Žádný nejlepší model nebyl uložen, vracím poslední stav.")

    return {
        "best_val_anees": best_val_anees,
        "best_val_nll": score_at_best['val_nll'],
        "best_val_mse": score_at_best['val_mse'],
        "best_iter": best_iter_count,
        "final_model": model
    }

def train_state_KalmanNet(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20):
    """
    Univerzální trénovací funkce pro modely StateKalmanNet a StateKalmanNetWithKnownR.
    Automaticky detekuje, zda model vrací kovarianci, a přizpůsobí se.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Chyba: Model nemá definovaný atribut 'returns_covariance'.")
        
    returns_covariance = model.returns_covariance
    print(f"INFO: Detekováno z atributu modelu, že vrací kovarianci: {returns_covariance}")
    
    for epoch in range(epochs):
        # --- Trénovací fáze ---
        model.train()
        train_loss = 0.0
        epoch_traces = []

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape

            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

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
                avg_trace_batch = torch.mean(torch.sum(torch.diagonal(predicted_cov_trajectory.detach(), offset=0, dim1=-2, dim2=-1), dim=-1)).item()
                epoch_traces.append(avg_trace_batch)

            loss = criterion(predicted_trajectory, x_true_batch[:, 1:, :])
            loss.backward()
            if clip_grad > 0:
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
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            print(f"Nové nejlepší model uloženo! {log_message}")
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

def train_state_KalmanNet_with_input(model, train_loader, val_loader, device,
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20):
    """
    Univerzální trénovací funkce pro modely StateKalmanNet...
    UPRAVENO pro systémový vstup u_t.
    Předpokládá, že DataLoadery vrací (x, y, u).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Chyba: Model nemá definovaný atribut 'returns_covariance'. "
                                "Prosím, přidejte `self.returns_covariance = True/False` do __init__ vašeho modelu.")

    returns_covariance = model.returns_covariance
    print(f"INFO: Detekováno z atributu modelu, že vrací kovarianci: {returns_covariance}")

    for epoch in range(epochs):
        # --- Trénovací fáze ---
        model.train()
        train_loss = 0.0
        epoch_traces = []

        for x_true_batch, y_meas_batch, u_input_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)
            u_input_batch = u_input_batch.to(device)

            optimizer.zero_grad()
            batch_size, seq_len, _ = x_true_batch.shape

            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

            predictions_x = []
            predictions_P = []

            for t in range(1, seq_len):
                y_t = y_meas_batch[:, t, :]
                # u_input_batch[:, t-1, :] odpovídá u_{k-1}, který ovlivňuje přechod do x_k (čas t)
                u_t = u_input_batch[:, t-1, :] 

                # Změna 3: Předání y_t a u_t do model.step()
                step_output = model.step(y_t, u_t)

                if returns_covariance:
                    x_filtered_t, P_filtered_t = step_output
                    predictions_P.append(P_filtered_t)
                else:
                    x_filtered_t = step_output

                predictions_x.append(x_filtered_t)

            predicted_trajectory = torch.stack(predictions_x, dim=1)

            if returns_covariance and predictions_P:
                predicted_cov_trajectory = torch.stack(predictions_P, dim=1)
                avg_trace_batch = torch.mean(torch.sum(torch.diagonal(predicted_cov_trajectory.detach(), offset=0, dim1=-2, dim2=-1), dim=-1)).item()
                epoch_traces.append(avg_trace_batch)

            # Loss se počítá stále stejně (porovnání predikovaného a skutečného stavu x)
            loss = criterion(predicted_trajectory, x_true_batch[:, 1:, :])
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_epoch_trace = np.mean(epoch_traces) if epoch_traces else 0.0

        # --- Validační fáze ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val, u_input_val in val_loader:
                x_true_val, y_meas_val = x_true_val.to(device), y_meas_val.to(device)
                u_input_val = u_input_val.to(device)

                batch_size_val, seq_len_val, _ = x_true_val.shape
                model.reset(batch_size=batch_size_val, initial_state=x_true_val[:, 0, :])

                val_predictions = []
                for t in range(1, seq_len_val):
                    y_t_val = y_meas_val[:, t, :]
                    u_t_val = u_input_val[:, t-1, :]

                    step_output_val = model.step(y_t_val, u_t_val)

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
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            print(f"Nové nejlepší model uloženo! {log_message}")
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

def train_state_KalmanNet_sliding_window(model, train_loader, val_loader, device, 
                          epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20,
                          tbptt_k=2, tbptt_w=10,optimizer_=torch.optim.Adam, weight_decay_=1e-4):
    criterion = nn.MSELoss()
    optimizer = optimizer_(model.parameters(), lr=lr, weight_decay=weight_decay_)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Chyba: Model nemá definovaný atribut 'returns_covariance'.")
    if not hasattr(model, '_detach'):
            raise AttributeError("Chyba: Model nemá implementovanou metodu `_detach()`, "
                                 "která je nutná pro TBPTT(k,w,D).")
        
    returns_covariance = model.returns_covariance
    print(f"INFO: Detekováno z atributu modelu, že vrací kovarianci: {returns_covariance}")
    print(f"INFO: Spouštím trénink s TBPTT(k={tbptt_k}, w={tbptt_w})")

    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            batch_size, seq_len, _ = x_true_batch.shape # seq_len je 'D' z článku
            
            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

            total_loss_for_batch = 0.0
            num_windows = 0

            # Smyčka přes celou sekvenci po oknech 'w'
            for t_start in range(1, seq_len, tbptt_w):
                t_end = min(t_start + tbptt_w, seq_len)
                window_len = t_end - t_start
                
                if window_len == 0:
                    continue
                    
                predictions_x = []

                # 1. Forward pass přes okno 'w'
                # Skrytý stav h_t se propaguje z předchozího okna
                for t in range(t_start, t_end):
                    y_t = y_meas_batch[:, t, :]
                    step_output = model.step(y_t)
                    
                    x_filtered_t = step_output[0] if returns_covariance else step_output
                    predictions_x.append(x_filtered_t)
                    
                    # Detach gradientu každých 'k' kroků 
                    if (t - t_start + 1) % tbptt_k == 0:
                        model._detach()

                # 3. Detach na konci okna (před backward())
                model._detach()
                
                # 4. Výpočet loss POUZE pro toto okno
                predicted_window = torch.stack(predictions_x, dim=1)
                true_window = x_true_batch[:, t_start:t_end, :]
                
                loss = criterion(predicted_window, true_window)

                # 5. Backward pass a update vah (pro každé okno)
                optimizer.zero_grad()
                loss.backward() # Gradient teče zpět jen 'k' kroků
                
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                total_loss_for_batch += loss.item()
                num_windows += 1
            
            if num_windows > 0:
                train_loss += (total_loss_for_batch / num_windows)
        
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        epoch_val_loss = 0.0
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
                val_loss_batch = criterion(predicted_val_trajectory, x_true_val[:, 1:, :])
                epoch_val_loss += val_loss_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            print(log_message)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            print(f"Nové nejlepší model uloženo! {log_message}")
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

def train_state_KalmanNet_sliding_window_grid_search(model, train_loader, val_loader, device, 
                                         epochs=100, lr=1e-3, clip_grad=10, early_stopping_patience=20,
                                         tbptt_k=2, tbptt_w=10, optimizer_=torch.optim.Adam, weight_decay_=1e-4,
                                         verbose=True):
    criterion = nn.MSELoss()
    optimizer = optimizer_(model.parameters(), lr=lr, weight_decay=weight_decay_)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=verbose)
    
    best_val_loss = float('inf')
    best_train_loss_at_best_val = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    model.eval()
    if not hasattr(model, 'returns_covariance'):
            raise AttributeError("Chyba: Model nemá definovaný atribut 'returns_covariance'.")
    if not hasattr(model, '_detach'):
            raise AttributeError("Chyba: Model nemá implementovanou metodu `_detach()`, "
                                 "která je nutná pro TBPTT(k,w,D).")
        
    returns_covariance = model.returns_covariance
    
    if verbose:
        print(f"INFO: Detekováno z atributu modelu, že vrací kovarianci: {returns_covariance}")
        print(f"INFO: Spouštím trénink s TBPTT(k={tbptt_k}, w={tbptt_w})")

    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            batch_size, seq_len, _ = x_true_batch.shape
            
            model.reset(batch_size=batch_size, initial_state=x_true_batch[:, 0, :])

            total_loss_for_batch = 0.0
            num_windows = 0

            # Smyčka přes celou sekvenci po oknech 'w'
            for t_start in range(1, seq_len, tbptt_w):
                t_end = min(t_start + tbptt_w, seq_len)
                window_len = t_end - t_start
                
                if window_len == 0:
                    continue
                    
                predictions_x = []

                # 1. Forward pass přes okno 'w'
                for t in range(t_start, t_end):
                    y_t = y_meas_batch[:, t, :]
                    step_output = model.step(y_t)
                    
                    x_filtered_t = step_output[0] if returns_covariance else step_output
                    predictions_x.append(x_filtered_t)
                    
                    # Detach gradientu každých 'k' kroků 
                    if (t - t_start + 1) % tbptt_k == 0:
                        model._detach()

                model._detach()
                
                predicted_window = torch.stack(predictions_x, dim=1)
                true_window = x_true_batch[:, t_start:t_end, :]
                
                loss = criterion(predicted_window, true_window)

                optimizer.zero_grad()
                loss.backward()
                
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                total_loss_for_batch += loss.item()
                num_windows += 1
            
            if num_windows > 0:
                train_loss += (total_loss_for_batch / num_windows)
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        epoch_val_loss = 0.0
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
                val_loss_batch = criterion(predicted_val_trajectory, x_true_val[:, 1:, :])
                epoch_val_loss += val_loss_batch.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if verbose and (epoch + 1) % 5 == 0:
            log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
            print(log_message)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss_at_best_val = avg_train_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            
            if verbose:
                log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}'
                print(f"Nové nejlepší model uloženo! {log_message}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping spuštěno po {epoch + 1} epochách.")
            break
            
    if verbose:
        print("Trénování dokončeno.")
        
    if best_model_state:
        if verbose:
            print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
    else:
        best_train_loss_at_best_val = avg_train_loss
        best_val_loss = avg_val_loss
        if verbose:
            print("Varování: Nebyl nalezen 'nejlepší' model, vracím metriky z poslední epochy.")

        
    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss_at_best_val
    }