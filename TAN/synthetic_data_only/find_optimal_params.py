import sys
import os
from pathlib import Path
import scipy.io as sio

# ==============================================================================
# 0. NASTAVENÍ CEST (FIX IMPORTŮ)
# ==============================================================================
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent # skript -> synthetic_data -> debug -> root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"INFO: Přidán kořen projektu do sys.path: {project_root}")

import torch
import torch.nn.functional as func
import numpy as np
import optuna
from tqdm import tqdm

import Filters 
from Systems import DynamicSystemTAN 

# ==============================================================================
# 1. KONFIGURACE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cesty k datům
DATA_MAP_PATH = project_root / 'data' / 'data.mat'
DATA_TRAJECTORIES_PATH = script_path.parent / 'generated_data_synthetic_controlled' / 'len_300' / 'train.pt'

# Parametry optimalizace
N_TRIALS = 200           # Stačí méně, hledáme jen 1 parametr
N_TRAJECTORIES = 40      # Počet trajektorií pro robustní průměr
TIMEOUT_SEC = 3600 * 2   

print(f"--- OPTIMALIZACE Q PRO UKF (R je fixní) ---")
print(f"Zařízení: {DEVICE}")
print(f"Trajektorie: {DATA_TRAJECTORIES_PATH}")

# ==============================================================================
# 2. NAČTENÍ DAT
# ==============================================================================

# A) Mapa
if not DATA_MAP_PATH.exists():
    raise FileNotFoundError(f"Chybí mapa: {DATA_MAP_PATH}")

mat_data = sio.loadmat(DATA_MAP_PATH)

try:
    souradniceX_mapa = mat_data['souradniceX']
    souradniceY_mapa = mat_data['souradniceY'] 
    souradniceZ_mapa = mat_data['souradniceZ']
except KeyError as e:
    print(f"CHYBA: Klíč {e} nebyl v souboru nalezen. Dostupné klíče: {mat_data.keys()}")
    raise e

x_axis_unique = souradniceX_mapa[0, :]
y_axis_unique = souradniceY_mapa[:, 0]

# Tensor mapy
terMap_tensor = torch.from_numpy(souradniceZ_mapa).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
x_min, x_max = x_axis_unique.min(), x_axis_unique.max()
y_min, y_max = y_axis_unique.min(), y_axis_unique.max()

print(f"Mapa načtena: {souradniceZ_mapa.shape}")

# B) Trajektorie
if not DATA_TRAJECTORIES_PATH.exists():
    # Fallback
    fallback = script_path.parent / 'generated_data_synthetic_controlled' / 'len_100' / 'val.pt'
    print(f"⚠️ Varování: Dataset 300 nenalezen, zkouším fallback: {fallback}")
    DATA_TRAJECTORIES_PATH = fallback

data_dict = torch.load(DATA_TRAJECTORIES_PATH, map_location=DEVICE)
X_gt_all = data_dict['x']
Y_meas_all = data_dict['y']

if len(X_gt_all) < N_TRAJECTORIES:
    N_TRAJECTORIES = len(X_gt_all)

# Náhodný výběr podmnožiny
perm = torch.randperm(len(X_gt_all))
idx_subset = perm[:N_TRAJECTORIES]

X_subset = X_gt_all[idx_subset]
Y_subset = Y_meas_all[idx_subset]

print(f"Načteno {N_TRAJECTORIES} trajektorií pro optimalizaci.")

# ==============================================================================
# 3. FUNKCE MĚŘENÍ (h) - Vaše funkční verze
# ==============================================================================

def h_nl_differentiable(x: torch.Tensor, map_tensor, x_min, x_max, y_min, y_max) -> torch.Tensor:
    batch_size = x.shape[0]
    px = x[:, 0]
    py = x[:, 1]

    # Normalizace
    px_norm = 2.0 * (px - x_min) / (x_max - x_min) - 1.0
    py_norm = 2.0 * (py - y_min) / (y_max - y_min) - 1.0

    sampling_grid = torch.stack((px_norm, py_norm), dim=1).view(batch_size, 1, 1, 2)

    vyska_terenu_batch = func.grid_sample(
        map_tensor.expand(batch_size, -1, -1, -1),
        sampling_grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True 
    )
    vyska_terenu = vyska_terenu_batch.view(batch_size)

    # Transformace rychlostí - VAŠE FUNKČNÍ LOGIKA
    eps = 1e-12
    vx_w, vy_w = x[:, 2], x[:, 3]
    norm_v_w = torch.sqrt(vx_w**2 + vy_w**2).clamp(min=eps)
    cos_psi = vx_w / norm_v_w
    sin_psi = vy_w / norm_v_w

    # Zde používám znaménka, která jste potvrdil, že fungují:
    vx_b = cos_psi * vx_w - sin_psi * vy_w 
    vy_b = sin_psi * vx_w + cos_psi * vy_w 

    result = torch.stack([vyska_terenu, vx_b, vy_b], dim=1)
    return result

h_wrapper = lambda x: h_nl_differentiable(
    x, map_tensor=terMap_tensor, 
    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
)

# ==============================================================================
# 4. OBJECTIVE FUNCTION (Pouze Q)
# ==============================================================================

def objective(trial):
    # A) Parametry k optimalizaci
    # Hledáme pouze škálu procesního šumu
    q_scale = trial.suggest_float("q_scale", 1e-4, 100.0, log=True)
    
    # B) Známé parametry (FIXNÍ R)
    # Odpovídá: noise_std = torch.tensor([5.0, 1.0, 1.0])
    r_z_std = 5.0
    r_vx_std = 1.0
    r_vy_std = 1.0
    
    # C) Model
    state_dim = 4
    obs_dim = 3
    dt = 1.0
    
    F = torch.tensor([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], device=DEVICE)
    
    # Q Matice (Diskrétní CV model)
    Q = torch.zeros((4, 4), device=DEVICE)
    # Blok X
    Q[0, 0] = dt**3/3; Q[0, 2] = dt**2/2
    Q[2, 0] = dt**2/2; Q[2, 2] = dt
    # Blok Y
    Q[1, 1] = dt**3/3; Q[1, 3] = dt**2/2
    Q[3, 1] = dt**2/2; Q[3, 3] = dt
    
    # Aplikace škály
    Q = Q * q_scale
    
    # R Matice (FIXNÍ)
    R = torch.tensor([
        [r_z_std**2, 0, 0],
        [0, r_vx_std**2, 0],
        [0, 0, r_vy_std**2]
    ], device=DEVICE)
    
    P0 = torch.eye(4, device=DEVICE) * 25.0
    Ex0_dummy = torch.zeros(4, device=DEVICE)

    system_model = DynamicSystemTAN(
        state_dim=state_dim, obs_dim=obs_dim,
        Q=Q.float(), R=R.float(),
        Ex0=Ex0_dummy, P0=P0,
        F=F.float(), h=h_wrapper,
        x_axis_unique=torch.from_numpy(x_axis_unique), 
        y_axis_unique=torch.from_numpy(y_axis_unique),
        device=DEVICE
    )
    
    ukf = Filters.UnscentedKalmanFilter(system_model)
    
    # D) Evaluace
    total_rmse = 0.0
    valid_count = 0
    
    for i in range(N_TRAJECTORIES):
        try:
            x_gt = X_subset[i]
            y_obs = Y_subset[i]
            x_init = x_gt[0]
            
            res = ukf.process_sequence(y_obs, Ex0=x_init, P0=P0)
            x_est = res['x_filtered']
            
            # RMSE na posledních 80% trajektorie
            start_idx = 20
            min_len = min(len(x_est), len(x_gt))
            if min_len <= start_idx: continue

            diff = x_est[start_idx:min_len, :2] - x_gt[start_idx:min_len, :2]
            mse = torch.mean(torch.sum(diff**2, dim=1)).item()
            rmse = np.sqrt(mse)
            
            if np.isnan(rmse) or rmse > 5000.0:
                return 1e6 
                
            total_rmse += rmse
            valid_count += 1
            
            if (i + 1) % 10 == 0:
                trial.report(total_rmse / valid_count, i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
        except optuna.TrialPruned:
            raise
        except Exception:
            return 1e6

    if valid_count == 0: return 1e6
    return total_rmse / valid_count

# ==============================================================================
# 5. START
# ==============================================================================
if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    print(f"Spouštím optimalizaci pouze pro Q ({N_TRIALS} pokusů)...")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nPřerušeno.")
        
    print("\n" + "="*60)
    print(f"🏆 VÍTĚZNÉ PARAMETRY (RMSE: {study.best_value:.4f} m)")
    print("="*60)
    
    bp = study.best_params
    q_best = bp['q_scale']
    
    print(f"Optimální q_scale: {q_best:.6f}")
    
    print("\n>>> COPY-PASTE DO Systems.py <<<")
    print(f"q = {q_best:.6f}")
    print(f"self.Q = q * torch.tensor([")
    print(f"    [1/3, 0, 1/2, 0],")
    print(f"    [0, 1/3, 0, 1/2],")
    print(f"    [1/2, 0, 1, 0],")
    print(f"    [0, 1/2, 0, 1]")
    print(f"])")
    print(f"# Fixní R podle generátoru (std: [5, 1, 1])")
    print(f"self.R = torch.tensor([")
    print(f"    [25.0, 0, 0],")
    print(f"    [0, 1.0, 0],")
    print(f"    [0, 0, 1.0]")
    print(f"])")