import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

# =================================================================================
# 1. POMOCNÉ FUNKCE
# =================================================================================

def generate_data(system, num_trajectories, seq_len):
    """Generuje data (trajektorie) pro daný dynamický systém."""
    device = system.Ex0.device
    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim, device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim, device=device)

    for i in range(num_trajectories):
        # x = system.get_initial_state() # Vrací 1D tenzor, např. shape [1]
        x = system.get_deterministic_initial_state()
        for t in range(seq_len):
            if t > 0:
                # `step` nyní očekává dávku, takže x předáme jako dávku o velikosti 1
                x = system.step(x.unsqueeze(0)).squeeze(0)

            # `measure` také očekává dávku
            y = system.measure(x.unsqueeze(0)).squeeze(0)

            x_data[i, t, :] = x
            y_data[i, t, :] = y

    return x_data, y_data

def store_model(model, path):
    """Uloží stav modelu do souboru."""
    torch.save(model.state_dict(), path)
    print(f"Model byl uložen do {path}")


def calculate_anees_vectorized(x_true_tensor, x_hat_tensor, P_hat_tensor):
    """
    Vektorizovaná verze pro výpočet ANEES.
    Přijímá jeden velký tenzor pro každou sadu dat.
    Tvar vstupů: [Počet_Trajektorií, Délka_Sekvence, Dimenze]
    """
    # Kontrola, zda máme vůbec nějaká data
    if x_true_tensor.numel() == 0:
        return float('nan')
        
    # Kontrola, zda sedí tvary (počet trajektorií a délka sekvence)
    if x_true_tensor.shape[:2] != x_hat_tensor.shape[:2] or x_true_tensor.shape[:2] != P_hat_tensor.shape[:2]:
        print("!!! CHYBA: Tenzory pro ANEES nemají stejný počet trajektorií nebo délek sekvencí!")
        return float('nan')

    device = x_true_tensor.device
    jitter = torch.eye(P_hat_tensor.shape[-1], device=device) * 1e-6
    
    # Počítáme chybu od t=1
    # Tvar všech tenzorů bude [Num_Traj, Seq_Len - 1, ...]
    error = x_true_tensor[:, 1:, :] - x_hat_tensor[:, 1:, :]
    P_hat_seq = P_hat_tensor[:, 1:, :, :]
    
    try:
        # Inverze kovariančních matic pro všechny trajektorie a časové kroky najednou
        P_inv_seq = torch.linalg.inv(P_hat_seq + jitter) # Tvar [N, T-1, D, D]
        
        # Vektorizovaný výpočet NEES
        # error.unsqueeze(-2) -> [N, T-1, 1, D]
        # P_inv_seq @ error.unsqueeze(-1) -> [N, T-1, D, 1]
        # ... @ ... -> [N, T-1, 1, 1]
        nees_tensor = (error.unsqueeze(-2) @ P_inv_seq @ error.unsqueeze(-1)).squeeze() # Tvar [N, T-1]
        
        # Zprůměrujeme přes všechny trajektorie a všechny časové kroky
        avg_anees = torch.mean(nees_tensor).item()
        
        return avg_anees
        
    except torch.linalg.LinAlgError:
        print("!!! VAROVÁNÍ: Selhala inverze matice při výpočtu ANEES. !!!")
        return float('nan')