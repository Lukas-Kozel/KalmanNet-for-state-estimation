import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

# =================================================================================
# 1. POMOCNÉ FUNKCE
# =================================================================================

def generate_data_deterministic_init_state(system, num_trajectories, seq_len):
    """Generuje data (trajektorie) pro daný dynamický systém."""
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
    """Uloží stav modelu do souboru."""
    torch.save(model.state_dict(), path)
    print(f"Model byl uložen do {path}")


def calculate_anees_vectorized(x_true_tensor, x_hat_tensor, P_hat_tensor):
    """
    Vektorizovaná verze pro výpočet ANEES.
    Přijímá jeden velký tenzor pro každou sadu dat.
    Tvar vstupů: [Počet_Trajektorií, Délka_Sekvence, Dimenze]
    """
    if x_true_tensor.numel() == 0:
        return float('nan')

    if x_true_tensor.shape[:2] != x_hat_tensor.shape[:2] or x_true_tensor.shape[:2] != P_hat_tensor.shape[:2]:
        print("!!! CHYBA: Tenzory pro ANEES nemají stejný počet trajektorií nebo délek sekvencí!")
        return float('nan')

    device = x_true_tensor.device
    jitter = torch.eye(P_hat_tensor.shape[-1], device=device) * 1e-6

    error = x_true_tensor[:, 1:, :] - x_hat_tensor[:, 1:, :]
    P_hat_seq = P_hat_tensor[:, 1:, :, :]
    
    try:
        P_inv_seq = torch.linalg.inv(P_hat_seq + jitter) # Tvar [N, T-1, D, D]
        nees_tensor = (error.unsqueeze(-2) @ P_inv_seq @ error.unsqueeze(-1)).squeeze() # Tvar [N, T-1]
        
        avg_anees = torch.mean(nees_tensor).item()
        
        return avg_anees
        
    except torch.linalg.LinAlgError:
        print("!!! VAROVÁNÍ: Selhala inverze matice při výpočtu ANEES. !!!")
        return float('nan')
    
# Funkce pro generování dat
def generate_data(system, num_trajectories, seq_len):

    # Zjištění zařízení, na kterém systém pracuje
    device = system.Ex0.device

    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim,device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim,device=device)
    for i in range(num_trajectories):
        x = system.get_initial_state()
        for t in range(seq_len):
            if t>0:
                x = system.step(x)            
            
            y = system.measure(x)
            x_data[i, t, :] = x.squeeze()
            y_data[i, t, :] = y.squeeze()
    return x_data, y_data

def generate_data_for_map(system, num_trajectories, seq_len):
    """
    Generuje data pomocí metody zahození (rejection sampling).
    OPRAVENO: Kontroluje, zda STAV (pozice) neopustil hranice mapy.
    Pokud ano, trajektorie je zahozen a generuje se nová.
    """
    device = system.Ex0.device
    
    # Získáme hranice mapy ze systémového modelu
    min_x, max_x = system.min_x, system.max_x
    min_y, max_y = system.min_y, system.max_y
    print(f"INFO: Generátor dat používá hranice X:[{min_x:.2f}-{max_x:.2f}], Y:[{min_y:.2f}-{max_y:.2f}]")

    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim, device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim, device=device)

    print(f"Generuji {num_trajectories} platných trajektorií (metoda zahození)...")
    
    generated_count = 0
    total_attempts = 0
    
    while generated_count < num_trajectories:
        total_attempts += 1
        trajectory_is_valid = True
        
        temp_x_traj = torch.zeros(seq_len, system.state_dim, device=device)
        temp_y_traj = torch.zeros(seq_len, system.obs_dim, device=device)
        
        # Začneme s počátečním stavem UVNITŘ mapy (můžeš použít Ex0 nebo náhodný)
        # Použijeme get_initial_state, ale pro jistotu ho ořízneme na hranice
        x_current = system.get_initial_state().view(1, -1)
        x_current[0, 0] = x_current[0, 0].clamp(min_x, max_x)
        x_current[0, 1] = x_current[0, 1].clamp(min_y, max_y)

        for t in range(seq_len):
            # --- OPRAVENÁ KONTROLA ---
            # Zkontrolujeme, zda AKTUÁLNÍ STAV je stále uvnitř mapy
            px_curr, py_curr = x_current[0, 0].item(), x_current[0, 1].item()
            if not (min_x <= px_curr <= max_x and min_y <= py_curr <= max_y):
                trajectory_is_valid = False
                # print(f"    Pokus {total_attempts}, krok {t}: Trajektorie opustila mapu [{px_curr:.2f}, {py_curr:.2f}]") # Pro debug
                break # Trajektorie je neplatná, přerušíme a zahodíme
            # --- KONEC OPRAVY ---

            # Pokud je stav v mapě, provedeme měření a krok
            try:
                y_current = system.measure(x_current)
            except Exception as e:
                print(f"Chyba při system.measure v kroku {t} pro x_current={x_current}: {e}")
                trajectory_is_valid = False
                break
                
            # Uložíme platný stav a měření
            temp_x_traj[t, :] = x_current.squeeze()
            temp_y_traj[t, :] = y_current.squeeze()
            
            # Vypočítáme další stav
            try:
                x_current = system.step(x_current)
            except Exception as e:
                print(f"Chyba při system.step v kroku {t} pro x_current={x_current}: {e}")
                trajectory_is_valid = False
                break

        # Uložíme trajektorii, POUZE pokud byla celá platná
        if trajectory_is_valid:
            x_data[generated_count, :, :] = temp_x_traj
            y_data[generated_count, :, :] = temp_y_traj
            generated_count += 1
            if generated_count % (num_trajectories // 10 if num_trajectories >= 10 else 1) == 0:
                 print(f"  Úspěšně vygenerována trajektorie {generated_count}/{num_trajectories} (Pokusů: {total_attempts})")

    print("-" * 30)
    print("Generování dat dokončeno.")
    print(f"Celkový počet pokusů: {total_attempts}")
    if total_attempts > 0:
      print(f"Úspěšnost (platné trajektorie / pokusy): { (num_trajectories / total_attempts) * 100 :.2f}%")
    print(f"Celkový počet vygenerovaných trajektorií: {x_data.shape}")
    
    return x_data, y_data

def generate_data_with_input(system, num_trajectories, u_sequence):

    # Zjištění zařízení, na kterém systém pracuje
    device = system.Ex0.device
    seq_len = u_sequence.shape[1]

    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim,device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim,device=device)
    for i in range(num_trajectories):
        # get_initial_state() vrací tvar [3]
        x = system.get_initial_state_input() 
        
        for t in range(seq_len):
            
            # --- KLÍČOVÁ OPRAVA ---
            # Před každým použitím zajistíme, že 'x' má tvar [1, state_dim].
            # .view(1, -1) převede [3] na [1, 3] bezpečně a efektivně.
            # Pokud už x má tvar [1, 3], .view(1, -1) nic nezmění.
            x_batch = x.view(1, -1)
            
            # Nyní všechny metody voláme s konzistentním tvarem x_batch
            y = system.measure(x_batch)
            x_data[i, t, :] = x_batch.squeeze() # Ukládáme zploštělý vektor
            y_data[i, t, :] = y.squeeze()
            
            u = u_sequence[:,t]
            
            # Metoda step vrátí [1, 3], což uložíme do 'x' pro další iteraci
            x_next = system.step(x_batch, u)
            x = x_next.view(1,-1)
    return x_data, y_data


def store_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model byl uložen do {path}")

def calculate_anees(x_true_list, x_hat_list, P_hat_list):
    """
    Vypočítá Average NEES (ANEES) ze seznamů trajektorií.
    Očekává, že všechny vstupní tenzory jsou na CPU.
    """
    num_runs = len(x_true_list)
    if num_runs == 0:
        return 0.0
    
    total_nees = 0.0
    for i in range(num_runs):
        x_true = x_true_list[i]
        x_hat = x_hat_list[i]
        P_hat = P_hat_list[i]
        
        # NEES se počítá od t=1, protože v t=0 je chyba nulová a P0 je jen odhad
        seq_len_anees = x_true.shape[0] - 1
        if seq_len_anees <= 0:
            continue
            
        state_dim = x_true.shape[1]
        nees_samples_run = torch.zeros(seq_len_anees)

        for t in range(seq_len_anees):
            t_idx = t + 1 # Indexujeme od 1
            error = x_true[t_idx] - x_hat[t_idx]
            P_t = P_hat[t_idx]
            
            P_stable = P_t + torch.eye(state_dim, device=P_t.device) * 1e-9
            
            try:
                P_inv = torch.inverse(P_stable)
                nees_samples_run[t] = error.unsqueeze(0) @ P_inv @ error.unsqueeze(-1)
            except torch.linalg.LinAlgError:
                print(f"Varování: Singularita matice P v trajektorii {i}, kroku {t_idx}. Přeskakuji.")
                nees_samples_run[t] = float('nan')
            
        if not torch.isnan(nees_samples_run).all():
            total_nees += torch.nanmean(nees_samples_run).item()
        
    return total_nees / num_runs if num_runs > 0 else 0.0