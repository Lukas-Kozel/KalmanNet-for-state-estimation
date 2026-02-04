import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# === KONFIGURACE ===
DATA_DIR = 'preprocessed_NCLT_trajectory-2012-01-22'
DT = 1.0
SUBSET_LEN = 3000  # Délka trajektorie pro vyhodnocení
NUM_TRIALS = 10000   # Počet pokusů v Random Search (čím víc, tím lépe)

print(f"--- ROBUSTNÍ OPTIMALIZACE Q a R (Random Search) ---")

# 1. Načtení dat
try:
    train_list = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    gt = train_list[0]['ground_truth'].float().numpy()    # [N, 4]
    meas = train_list[0]['filtered_gps'].float().numpy()  # [N, 4]
    
    # Zkrácení pro rychlost
    gt = gt[:SUBSET_LEN]
    meas = meas[:SUBSET_LEN]
    print(f"Data načtena: {len(gt)} kroků.")
except Exception as e:
    print(f"CHYBA NAČTENÍ: {e}")
    exit()

# 2. Analytický základ pro R
residuals = meas - gt
var_gps = np.var(residuals[:, [0, 2]]) # Průměr rozptylu pozice
var_odo = np.var(residuals[:, [1, 3]]) # Průměr rozptylu rychlosti
print(f"Base Stats -> GPS Std: {np.sqrt(var_gps):.3f} m, ODO Std: {np.sqrt(var_odo):.3f} m/s")

# 3. KF Funkce (optimalizovaná pro Numpy pro rychlost v cyklu)
def run_kf_numpy(q_std, r_factor, measurements, gt_data):
    dt = DT
    n = len(measurements)
    
    # A. Matice F (CV Model)
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    
    # B. Matice H
    H = np.eye(4)
    
    # C. Matice R (Ladíme faktor)
    # Zvyšujeme/snižujeme důvěru v měření globálně
    R = np.diag([var_gps, var_odo, var_gps, var_odo]) * r_factor
    
    # D. Matice Q (Discrete White Noise Acceleration)
    # q_std je "process noise spectral density" (sigma_a)
    # Q blok pro jednu osu (x nebo y)
    Q_axis = np.array([[dt**3/3, dt**2/2],
                       [dt**2/2, dt]]) * (q_std**2)
    
    # Celá Q (bloková diagonála)
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = Q_axis
    Q[2:4, 2:4] = Q_axis
    
    # Inicializace
    x = gt_data[0].copy() # Start na GT
    P = np.eye(4) * 1.0
    
    mse_accum = 0.0
    
    # KF Loop
    for k in range(n):
        y = measurements[k]
        
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (y - H @ x)
        I_KH = np.eye(4) - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T # Joseph form pro stabilitu
        
        # MSE akumulace (jen pozice X, Y)
        err_x = x[0] - gt_data[k, 0]
        err_y = x[2] - gt_data[k, 2]
        mse_accum += (err_x**2 + err_y**2)
        
    return mse_accum / n

# 4. RANDOM SEARCH LOOP
results = []
print(f"\nSpouštím {NUM_TRIALS} simulací...")

# Rozsahy pro hledání (Log-uniform pro Q, Uniform pro R factor)
q_search_space = np.logspace(-3, 0.5, NUM_TRIALS) # Q std od 0.001 do 3.0
r_search_space = np.random.uniform(0.5, 5.0, NUM_TRIALS) # R factor od 0.5x do 5x

# Pro jistotu přidáme pár fixních bodů
q_search_space[0] = 0.1
r_search_space[0] = 1.0

best_mse = float('inf')
best_params = None

for i in tqdm(range(NUM_TRIALS)):
    q_val = np.random.choice(q_search_space) # Náhodný výběr
    r_val = np.random.choice(r_search_space)
    
    loss = run_kf_numpy(q_val, r_val, meas, gt)
    
    results.append((q_val, r_val, loss))
    
    if loss < best_mse:
        best_mse = loss
        best_params = (q_val, r_val)

# 5. VÝSLEDKY A VIZUALIZACE
q_res = [r[0] for r in results]
r_res = [r[1] for r in results]
mse_res = [r[2] for r in results]

print(f"\n=== VÍTĚZNÉ NASTAVENÍ ===")
print(f"MSE: {best_mse:.4f} m^2 (RMSE: {np.sqrt(best_mse):.2f} m)")
print(f"Optimální q_std (sigma_a): {best_params[0]:.6f}")
print(f"Optimální r_factor: {best_params[1]:.4f}")
print(f" -> Znamená to, že R by mělo být {best_params[1]:.2f}x větší/menší než naměřený rozptyl.")

# Vykreslení mapy
plt.figure(figsize=(10, 6))
sc = plt.scatter(q_res, r_res, c=mse_res, cmap='viridis_r', s=50, edgecolors='k')
plt.colorbar(sc, label='MSE (m^2)')
plt.xscale('log')
plt.xlabel('Process Noise Std (q_std)')
plt.ylabel('Measurement Noise Scale (r_factor)')
plt.title('Hledání optima pro KF (Tmavší = Lepší)')
plt.plot(best_params[0], best_params[1], 'r*', markersize=20, label='Optimum')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig('kf_optimization_landscape.png')
print("Graf uložen jako 'kf_optimization_landscape.png'")

print("\n--- Kód pro tvůj config ---")
print(f"var_gps = {var_gps * best_params[1]:.4f}")
print(f"var_odo = {var_odo * best_params[1]:.4f}")
print(f"q_scale = {best_params[0]**2:.6f}  # Pozor: toto je variance (std^2)")