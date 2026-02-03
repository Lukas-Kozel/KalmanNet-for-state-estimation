import torch
import numpy as np
import os
import optuna
import logging
import matplotlib.pyplot as plt

# Vypneme logování
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === KONFIGURACE ===
DATA_DIR = 'preprocessed_NCLT_trajectory-2012-01-22'
DT = 1.0
SUBSET_LEN = 5000  
N_TRIALS = 3000     # Dáme tomu hodně pokusů pro jistotu

print(f"--- ULTIMÁTNÍ OPTIMALIZACE Q a R ---")

# 1. Načtení dat
try:
    train_list = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    gt = train_list[0]['ground_truth'].float().numpy()[:SUBSET_LEN]
    meas = train_list[0]['filtered_gps'].float().numpy()[:SUBSET_LEN]
    print(f"Data načtena: {len(gt)} kroků.")
except Exception as e:
    print(f"CHYBA: {e}")
    exit()

# 2. Baseline Statistiky
residuals = meas - gt
var_gps_base = np.var(residuals[:, [0, 2]])
var_odo_base = np.var(residuals[:, [1, 3]])
print(f"Base GPS Var: {var_gps_base:.4f} (Std: {np.sqrt(var_gps_base):.2f}m)")
print(f"Base ODO Var: {var_odo_base:.4f}")

# 3. Rychlá KF implementace (Numpy optimized)
def run_kf_fast(q_std, r_factor_gps, r_factor_odo):
    n = len(meas)
    F = np.array([[1, DT, 0, 0], [0, 1, 0, 0], [0, 0, 1, DT], [0, 0, 0, 1]])
    H = np.eye(4)
    
    # Q Matrix
    q_var = q_std**2
    # Q pro pozici i rychlost (blokově)
    Q_block = np.array([[DT**3/3, DT**2/2], [DT**2/2, DT]]) * q_var
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = Q_block
    Q[2:4, 2:4] = Q_block
    
    # R Matrix (Diagonální)
    # Zde aplikujeme faktory
    R_diag = np.array([
        var_gps_base * r_factor_gps, 
        var_odo_base * r_factor_odo,
        var_gps_base * r_factor_gps,
        var_odo_base * r_factor_odo
    ])
    # Pro rychlost výpočtu nepoužíváme plnou matici R, ale vektor diag
    # (v update kroku to ale musíme rozepsat nebo použít trik)
    R = np.diag(R_diag)

    x = gt[0].copy()
    P = np.eye(4) * 10.0 # Větší počáteční nejistota
    
    mse_sum = 0.0
    
    for k in range(n):
        y = meas[k]
        
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update
        # S = H P H.T + R
        S = P + R # Protože H je identita
        
        # K = P H.T S^-1
        # Protože H je identita, K = P @ S^-1
        # Použijeme solve pro stabilitu: K = P @ inv(S) -> S.T @ K.T = P.T
        try:
            K = np.linalg.solve(S.T, P.T).T
        except np.linalg.LinAlgError:
            return 1e9 # Penalizace za singularitu
            
        x = x + K @ (y - x) # y - Hx, H je identita
        
        # Joseph form update pro P: (I - KH)P(I - KH).T + KRK.T
        I_KH = np.eye(4) - K # H je identita
        P = I_KH @ P @ I_KH.T + K @ R @ K.T
        
        # Error (X, Y position only)
        err_x = x[0] - gt[k, 0]
        err_y = x[2] - gt[k, 2]
        mse_sum += (err_x**2 + err_y**2)
        
    return mse_sum / n

# 4. Objective Function
def objective(trial):
    # Logaritmické škály pro VŠECHNY parametry
    # To umožní najít řádově správné nastavení
    q_std = trial.suggest_float("q_std", 1e-4, 10.0, log=True)
    
    # R faktory: od "nevěřím ničemu" (100x) po "věřím absolutně" (0.001x)
    r_factor_gps = trial.suggest_float("r_factor_gps", 1e-3, 100.0, log=True)
    r_factor_odo = trial.suggest_float("r_factor_odo", 1e-3, 100.0, log=True)
    
    return run_kf_fast(q_std, r_factor_gps, r_factor_odo)

# 5. Spuštění
print(f"Spouštím {N_TRIALS} pokusů...")
# Použijeme TPESampler s fixním seedem pro reprodukovatelnost
sampler = optuna.samplers.TPESampler(seed=42) 
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# 6. Výsledky
best = study.best_params
print("\n" + "="*40)
print(f"FINÁLNÍ VÝSLEDEK (MSE: {study.best_value:.4f})")
print("="*40)
print(f"q_std: {best['q_std']:.6f}")
print(f"r_factor_gps: {best['r_factor_gps']:.6f}")
print(f"r_factor_odo: {best['r_factor_odo']:.6f}")

final_var_gps = var_gps_base * best['r_factor_gps']
final_var_odo = var_odo_base * best['r_factor_odo']
final_q_scale = best['q_std']**2

print("\n--- COPY TO CONFIG ---")
print(f"var_gps = {final_var_gps:.6f}")
print(f"var_odo = {final_var_odo:.6f}")
print(f"q_scale = {final_q_scale:.8f}")
print("----------------------")

# 7. Vizualizace (Contour Plot)
# Ukáže nám vztah mezi Q a R_GPS - uvidíš to "údolí"
try:
    fig = optuna.visualization.plot_contour(study, params=["q_std", "r_factor_gps"])
    fig.write_image("optimization_landscape.png")
    print("Graf 'optimization_landscape.png' uložen.")
except:
    print("Vizualizace se nepodařila (možná chybí plotly/kaleido), ale čísla máš.")