import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

def generate_data_deterministic_init_state(system, num_trajectories, seq_len):
    """Generate trajectories for the given dynamic system (deterministic init)."""
    device = system.Ex0.device
    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim, device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim, device=device)

    for i in range(num_trajectories):
        x = system.get_deterministic_initial_state()
        for t in range(seq_len):
            if t > 0:
                x = system.step(x.unsqueeze(0)).squeeze(0)

            y = system.measure(x.unsqueeze(0)).squeeze(0)

            x_data[i, t, :] = x
            y_data[i, t, :] = y

    return x_data, y_data

def store_model(model, path):
    """Store model state to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def calculate_anees_vectorized(x_true_tensor, x_hat_tensor, P_hat_tensor):
    """
    Vectorized ANEES computation.
    Accepts single large tensors per dataset.
    Input shape: [Num_Trajectories, Seq_Length, Dim]
    """
    if x_true_tensor.numel() == 0:
        return float('nan')

    if x_true_tensor.shape[:2] != x_hat_tensor.shape[:2] or x_true_tensor.shape[:2] != P_hat_tensor.shape[:2]:
        print("ERROR: Tensors for ANEES do not have the same number of trajectories or sequence lengths!")
        return float('nan')

    device = x_true_tensor.device
    jitter = torch.eye(P_hat_tensor.shape[-1], device=device) * 1e-6

    error = x_true_tensor[:, 1:, :] - x_hat_tensor[:, 1:, :]
    P_hat_seq = P_hat_tensor[:, 1:, :, :]
    
    try:
        P_inv_seq = torch.linalg.inv(P_hat_seq + jitter)  # Shape [N, T-1, D, D]
        nees_tensor = (error.unsqueeze(-2) @ P_inv_seq @ error.unsqueeze(-1)).squeeze()  # Shape [N, T-1]

        avg_anees = torch.mean(nees_tensor).item()

        return avg_anees

    except torch.linalg.LinAlgError:
        print("WARNING: Matrix inversion failed during ANEES computation.")
        return float('nan')
    
def generate_data(system, num_trajectories, seq_len):
    """
    Generate data in a fully vectorized manner.
    """
    device = system.device

    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim, device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim, device=device)

    x_k = system.get_initial_state_batch(num_trajectories)
    
    y_k = system.measure(x_k)
    
    x_data[:, 0, :] = x_k
    y_data[:, 0, :] = y_k
    
    with torch.no_grad():
        for t in range(1, seq_len):
            
            x_k = system.step(x_k)
            y_k = system.measure(x_k)
            
            x_data[:, t, :] = x_k
            y_data[:, t, :] = y_k
            
    return x_data, y_data

def generate_data_for_map(system, num_trajectories, seq_len,force_initial_state_zero=False):
    device = system.Ex0.device
    
    # Get map boundaries from the system model
    min_x, max_x = system.min_x, system.max_x
    min_y, max_y = system.min_y, system.max_y
    print(f"INFO: Data generator uses bounds X:[{min_x:.2f}-{max_x:.2f}], Y:[{min_y:.2f}-{max_y:.2f}]")
    print(f"INFO: Forced start at zero: {force_initial_state_zero}")
    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim, device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim, device=device)

    generated_count = 0
    total_attempts = 0
    
    while generated_count < num_trajectories:
        total_attempts += 1
        trajectory_is_valid = True
        
        temp_x_traj = torch.zeros(seq_len, system.state_dim, device=device)
        temp_y_traj = torch.zeros(seq_len, system.obs_dim, device=device)
        
        if force_initial_state_zero:
            # Training mode: force start at local origin
            x_current = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
        else:
            x_current = system.get_deterministic_initial_state().view(1, -1)

        # If the start point is outside (should not happen if the map is properly shifted), clamp it
        x_current[0, 0] = x_current[0, 0].clamp(min_x, max_x)
        x_current[0, 1] = x_current[0, 1].clamp(min_y, max_y)
        for t in range(seq_len):

            px_curr, py_curr = x_current[0, 0].item(), x_current[0, 1].item()
            if not (min_x <= px_curr <= max_x and min_y <= py_curr <= max_y):
                trajectory_is_valid = False
                break
            # If the state is within the map, perform measurement and step
            try:
                y_current = system.measure(x_current)
            except Exception as e:
                print(f"Error during system.measure at step {t} for x_current={x_current}: {e}")
                trajectory_is_valid = False
                break
                
            temp_x_traj[t, :] = x_current.squeeze()
            temp_y_traj[t, :] = y_current.squeeze()

            try:
                x_current = system.step(x_current)
            except Exception as e:
                print(f"Error during system.step at step {t} for x_current={x_current}: {e}")
                trajectory_is_valid = False
                break

        if trajectory_is_valid:
            x_data[generated_count, :, :] = temp_x_traj
            y_data[generated_count, :, :] = temp_y_traj
            generated_count += 1
            if generated_count % (num_trajectories // 10 if num_trajectories >= 10 else 1) == 0:
                print(f"  Successfully generated trajectory {generated_count}/{num_trajectories} (Attempts: {total_attempts})")

    print("-" * 30)
    print("Data generation completed.")
    print(f"Total number of attempts: {total_attempts}")
    if total_attempts > 0:
        print(f"Success rate (valid trajectories / attempts): { (num_trajectories / total_attempts) * 100 :.2f}%")
    print(f"Total generated trajectories tensor shape: {x_data.shape}")
    
    return x_data, y_data

def store_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def calculate_anees(x_true_list, x_hat_list, P_hat_list):
    """
    Compute Average NEES (ANEES) from trajectory lists.
    Expects all input tensors to be on CPU.
    """
    num_runs = len(x_true_list)
    if num_runs == 0:
        return 0.0
    
    total_nees = 0.0
    for i in range(num_runs):
        x_true = x_true_list[i]
        x_hat = x_hat_list[i]
        P_hat = P_hat_list[i]
        
        # NEES is computed from t=1 since at t=0 the error is zero and P0 is only an initial estimate
        seq_len_anees = x_true.shape[0] - 1
        if seq_len_anees <= 0:
            continue
            
        state_dim = x_true.shape[1]
        nees_samples_run = torch.zeros(seq_len_anees)

        for t in range(seq_len_anees):
            t_idx = t + 1  # Index starting from 1
            error = x_true[t_idx] - x_hat[t_idx]
            P_t = P_hat[t_idx]
            
            P_stable = P_t + torch.eye(state_dim, device=P_t.device) * 1e-9
            
            try:
                P_inv = torch.inverse(P_stable)
                nees_samples_run[t] = error.unsqueeze(0) @ P_inv @ error.unsqueeze(-1)
            except torch.linalg.LinAlgError:
                print(f"Warning: Matrix P is singular in trajectory {i}, step {t_idx}. Skipping.")
                nees_samples_run[t] = float('nan')
            
        if not torch.isnan(nees_samples_run).all():
            total_nees += torch.nanmean(nees_samples_run).item()
        
    return total_nees / num_runs if num_runs > 0 else 0.0