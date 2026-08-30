import torch
import numpy as np
import os
import optuna
import logging
import matplotlib.pyplot as plt

# Disable logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === KONFIGURACE ===
DATA_DIR = 'preprocessed_NCLT_trajectory-2012-01-22-angle-update'
DT = 1.0
SUBSET_LEN = 5000  
N_TRIALS = 4000     # We'll give it plenty of attempts just to be sure

print(f"--- ULTIMATE OPTIMIZATION OF Q and R ---")

# 1. Data loading
try:
    train_list = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    gt = train_list[0]['ground_truth'].float().numpy()[:SUBSET_LEN]
    meas = train_list[0]['filtered_gps'].float().numpy()[:SUBSET_LEN]
    print(f"Data loaded: {len(gt)} steps.")
except Exception as e:
    print(f"CHYBA: {e}")
    exit()

# 2. Baseline Statistiky
residuals = meas - gt
var_gps_base = np.var(residuals[:, [0, 2]])
var_odo_base = np.var(residuals[:, [1, 3]])
print(f"Base GPS Var: {var_gps_base:.4f} (Std: {np.sqrt(var_gps_base):.2f}m)")
print(f"Base ODO Var: {var_odo_base:.4f}")

# 2. Baseline Statistiky & Korelace
residuals = meas - gt
cov_matrix_raw = np.cov(residuals, rowvar=False)

# Diagonal variances (you already have that)
var_gps_base = np.var(residuals[:, [0, 2]]) # Average of X and Y variance for GPS
var_odo_base = np.var(residuals[:, [1, 3]]) # Average of Vx and Vy variance for ODO

# Korelace mezi Vx a Vy (odometrie)
# Indexy v residuals: 0=X, 1=Vx, 2=Y, 3=Vy
cov_vx_vy = cov_matrix_raw[1, 3]
corr_vx_vy = cov_vx_vy / np.sqrt(cov_matrix_raw[1,1] * cov_matrix_raw[3,3])

print(f"Korelace chyb rychlosti (Vx vs Vy): {corr_vx_vy:.4f}")

# 3. Fast KF implementation (Numpy optimized)
def run_kf_fast(q_std, r_factor_gps, r_factor_odo):
    n = len(meas)
    F = np.array([[1, DT, 0, 0], [0, 1, 0, 0], [0, 0, 1, DT], [0, 0, 0, 1]])
    H = np.eye(4)
    
    # Q Matrix
    q_var = q_std**2
    # Q for position and velocity (block-wise)
    Q_block = np.array([[DT**3/3, DT**2/2], [DT**2/2, DT]]) * q_var
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = Q_block
    Q[2:4, 2:4] = Q_block
    
    # R Matrix (Diagonal)
    # Zde aplikujeme faktory
    R_diag = np.array([
        var_gps_base * r_factor_gps, 
        var_odo_base * r_factor_odo,
        var_gps_base * r_factor_gps,
        var_odo_base * r_factor_odo
    ])
    # For computation speed we don't use the full R matrix, but a diag vector
    
    R = np.diag(R_diag)

    x = gt[0].copy()
    # P = np.eye(4) * 10.0 # Larger initial uncertainty
    P = np.diag([10,1,10,1])
    
    mse_sum = 0.0
    
    for k in range(n):
        y = meas[k]
        
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update
        # S = H P H.T + R
        S = P + R # Because H is identity
        
        # K = P H.T S^-1
        # Because H is identity, K = P @ S^-1
        # We use solve for stability: K = P @ inv(S) -> S.T @ K.T = P.T
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
    # Logarithmic scales for ALL parameters
    # This allows finding the order-of-magnitude correct setting
    q_std = trial.suggest_float("q_std", 1e-4, 10.0, log=True)
    
    # R factors: from "I trust nothing" (100x) to "I trust absolutely" (0.001x)
    r_factor_gps = trial.suggest_float("r_factor_gps", 1e-3, 100.0, log=True)
    r_factor_odo = trial.suggest_float("r_factor_odo", 1e-3, 100.0, log=True)
    
    return run_kf_fast(q_std, r_factor_gps, r_factor_odo)

# 5. Execution
print(f"Running {N_TRIALS} trials...")
# We use TPESampler with a fixed seed for reproducibility
sampler = optuna.samplers.TPESampler(seed=42) 
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# 6. Results
best = study.best_params
print("\n" + "="*40)
print(f"FINAL RESULT (MSE: {study.best_value:.4f})")
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
# It will show us the relationship between Q and R_GPS - you'll see the "valley"
try:
    fig = optuna.visualization.plot_contour(study, params=["q_std", "r_factor_gps"])
    fig.write_image("optimization_landscape.png")
    print("Plot 'optimization_landscape.png' saved.")
except:
    print("Visualization failed (maybe plotly/kaleido is missing), but you have the numbers.")