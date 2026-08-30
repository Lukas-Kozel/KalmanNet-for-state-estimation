import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

def store_model(model, path):
    """Store model state to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

    
import torch
import torch

def calculate_anees_vectorized(x_true, x_hat, P_hat, eps=1e-6):
    """
    Robust ANEES computation using Pseudo-Inverse.
    Handles even the case where J_samples < State_Dim (singular matrix).
    """
    # 1. Flattening
    if x_true.dim() == 3: x_true = x_true.reshape(-1, x_true.shape[-1])
    if x_hat.dim() == 3: x_hat = x_hat.reshape(-1, x_hat.shape[-1])
    if P_hat.dim() == 4: P_hat = P_hat.reshape(-1, P_hat.shape[-2], P_hat.shape[-1])

    # 2. Error computation
    error = (x_true - x_hat).unsqueeze(-1)  # [N, Dim, 1]
    
    # 3. Pseudo-Inverze matice P
    # pinv is much more stable for Low-Rank matrices (when we have few samples)
    # hermitian=True states that the matrix is symmetric (which a covariance is)
    P_inv = torch.linalg.pinv(P_hat, hermitian=True)
    
    # 4. ANEES = error^T * P_inv * error
    # bmm: [N, 1, Dim] @ [N, Dim, Dim] -> [N, 1, Dim]
    #      [N, 1, Dim] @ [N, Dim, 1] -> [N, 1, 1]
    
    temp = torch.bmm(error.transpose(1, 2), P_inv)
    anees_per_sample = torch.bmm(temp, error).squeeze() # [N]
    
    # Protection against numerical artifacts (tiny negative numbers like -1e-10)
    anees_per_sample = torch.clamp(anees_per_sample, min=0.0)

    return anees_per_sample.mean().item()
    
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


import os
import matplotlib.pyplot as plt

def save_plot_eps(filename, folder_name=""):
    """
    Saves the current matplotlib figure as a vector .eps file.
    
    Parameters:
    filename (str): File name (without the .eps extension)
    folder_name (str): Name of the target folder
    """
    # Create the folder if it does not exist (at the same directory level)
    os.makedirs(folder_name, exist_ok=True)
    
    # Build the absolute path to the file
    filepath = os.path.join(folder_name, f"{filename}.eps")
    
    # Save the figure 
    # bbox_inches='tight' is key - it prevents cropping of the legend lying outside the axis
    plt.savefig(filepath, format='eps', bbox_inches='tight')
    print(f"--> Figure was successfully saved as vector: {filepath}")

def save_plot_pdf(filename, folder_name=""):
    os.makedirs(folder_name, exist_ok=True)
    filepath = os.path.join(folder_name, f"{filename}.pdf")
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"--> Figure saved as: {filepath}")
