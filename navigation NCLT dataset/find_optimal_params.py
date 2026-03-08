import torch
import torch.distributions as dist
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

# === KONFIGURACE ===
DATA_DIR = 'preprocessed_NCLT_trajectory-2012-01-22-angle-update' # Uprav si dle sebe
DT = 1.0
NUM_TRIALS = 50000  # Pro test zkráceno

# Detekce zařízení
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Běžím na: {device}")

# 1. Načtení dat
try:
    train_list = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    gt = train_list[0]['ground_truth'].float().to(device)
    meas = train_list[0]['filtered_gps'].float().to(device)
    FULL_LEN = gt.shape[0]
    print(f"✅ Data načtena. Celková délka: {FULL_LEN} kroků")
except Exception as e:
    print(f"CHYBA: {e}")
    exit()

# 2. DEFINICE PARAMETRŮ (Search Space)
print(f"Generuji {NUM_TRIALS} kombinací parametrů...")
q_std_vals = torch.pow(10, dist.Uniform(np.log10(0.0001), np.log10(5.0)).sample((NUM_TRIALS,)).to(device))
r_gps_vals = torch.pow(10, dist.Uniform(np.log10(1.0), np.log10(100.0)).sample((NUM_TRIALS,)).to(device))
r_odo_vals = torch.pow(10, dist.Uniform(np.log10(1e-5), np.log10(10)).sample((NUM_TRIALS,)).to(device))

# 3. PŘÍPRAVA MATIC
F_base = torch.tensor([[1.0, DT, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, DT],
                       [0.0, 0.0, 0.0, 1.0]], device=device)
F = F_base.unsqueeze(0).repeat(NUM_TRIALS, 1, 1)

H_base = torch.eye(4, device=device)
H = H_base.unsqueeze(0).repeat(NUM_TRIALS, 1, 1)

Q_base = torch.tensor([[DT**3/3, DT**2/2, 0, 0], 
                       [DT**2/2, DT, 0, 0],
                       [0, 0, DT**3/3, DT**2/2],
                       [0, 0, DT**2/2, DT]], device=device)

Q = torch.zeros((NUM_TRIALS, 4, 4), device=device)
q_var = q_std_vals ** 2  
Q[:, :, :] = Q_base.unsqueeze(0) * q_var.view(-1, 1, 1)

R = torch.zeros((NUM_TRIALS, 4, 4), device=device)
R[:, 0, 0] = r_gps_vals
R[:, 1, 1] = r_odo_vals
R[:, 2, 2] = r_gps_vals
R[:, 3, 3] = r_odo_vals

# 4. VEKTORIZOVANÝ KF LOOP
x = gt[0].unsqueeze(0).unsqueeze(2).repeat(NUM_TRIALS, 1, 1) 
p0_diag = torch.tensor([10.0, 1.0, 10.0, 1.0], device=device)
P = torch.diag(p0_diag).unsqueeze(0).repeat(NUM_TRIALS, 1, 1)
mse_accum = torch.zeros(NUM_TRIALS, device=device)

print("Spouštím vektorizovaný KF...")
for t in tqdm(range(len(meas)), desc="Simulace KF"):
    y = meas[t].unsqueeze(0).unsqueeze(2).repeat(NUM_TRIALS, 1, 1)
    
    x = torch.bmm(F, x)
    P = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q
    S = torch.bmm(torch.bmm(H, P), H.transpose(1, 2)) + R
    
    S_inv = torch.linalg.inv(S)
    K = torch.bmm(torch.bmm(P, H.transpose(1, 2)), S_inv)
    inov = y - torch.bmm(H, x)
    x = x + torch.bmm(K, inov)
    
    I = torch.eye(4, device=device).unsqueeze(0)
    I_KH = I - torch.bmm(K, H)
    P = torch.bmm(torch.bmm(I_KH, P), I_KH.transpose(1, 2)) + torch.bmm(torch.bmm(K, R), K.transpose(1, 2))
    
    diff = x.squeeze(2) - gt[t].unsqueeze(0) 
    dist_sq = diff[:, 0]**2 + diff[:, 2]**2 
    mse_accum += dist_sq

mse_final = mse_accum / len(meas)

best_idx = torch.argmin(mse_final).item()
best_mse = mse_final[best_idx].item()
best_q_scale = (q_std_vals[best_idx] ** 2).item()
best_var_gps = r_gps_vals[best_idx].item()
best_var_odo = r_odo_vals[best_idx].item()

print(f"\n=== VÍTĚZNÉ NASTAVENÍ (z {NUM_TRIALS} pokusů) ===")
print(f"MSE: {best_mse:.4f}")
print("-" * 30)
print(f"Grid-Search q_scale: {best_q_scale:.8f}")
print(f"Grid-Search var_gps: {best_var_gps:.6f}")
print(f"Grid-Search var_odo: {best_var_odo:.6f}")


# ==============================================================================
# 5. VÝPOČET EMPIRICKÉHO ŠUMU (Úkol od vedoucího)
# ==============================================================================
print("\n=== ANALÝZA EMPIRICKÉHO ŠUMU (GROUND TRUTH) ===")

# A) Šum měření: v_k = z_k - H * x_k
# Jelikož H je identita, je to z_k - x_k
v_k = meas - gt
v_k_var = torch.var(v_k, dim=0).cpu().numpy()

emp_var_gps = (v_k_var[0] + v_k_var[2]) / 2.0  # Průměr variance X a Y z GPS
emp_var_odo = (v_k_var[1] + v_k_var[3]) / 2.0  # Průměr variance Vx a Vy z Odometrie

# B) Šum procesu: w_k = x_{k+1} - F * x_k
gt_k = gt[:-1]
gt_k_plus_1 = gt[1:]
# Vynásobíme F * x_k (pomocí maticového násobení)
pred_gt = torch.matmul(gt_k, F_base.T)
w_k = gt_k_plus_1 - pred_gt
w_k_var = torch.var(w_k, dim=0).cpu().numpy()

# Extrakce q_scale: Z definice matice Q víme, že Q[1,1] a Q[3,3] se rovnají q_scale * DT.
# Protože DT = 1.0, empirická variance složek rychlosti ve vektoru w_k JE přímo q_scale.
emp_q_scale = (w_k_var[1] + w_k_var[3]) / 2.0

print(f"Empirický q_scale:   {emp_q_scale:.8f} (Grid-Search našel: {best_q_scale:.8f})")
print(f"Empirický var_gps:   {emp_var_gps:.6f} (Grid-Search našel: {best_var_gps:.6f})")
print(f"Empirický var_odo:   {emp_var_odo:.6f} (Grid-Search našel: {best_var_odo:.6f})")

# ==============================================================================
# 6. VIZUALIZACE (HISTOGRAMY vs. GAUSSIÁNY)
# ==============================================================================
# Převedeme data na CPU pro numpy a pyplot
v_k_np = v_k.cpu().numpy()
w_k_np = w_k.cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Analýza rozdělení šumu: Data vs. Parametry z Grid-Search", fontsize=16, fontweight='bold')

# Funkce pro plotování
def plot_dist(ax, data, theoretical_var, title, xlabel):
    ax.hist(data, bins=100, density=True, alpha=0.6, color='blue', label='Empirická data (Histogram)')
    
    # Teoretická Gaussova křivka s variancí nalezenou Grid-Searchem
    mu = 0
    sigma = np.sqrt(theoretical_var)
    x_val = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y_val = stats.norm.pdf(x_val, mu, sigma)
    
    ax.plot(x_val, y_val, 'r-', lw=2, label=f'GS Model\n(Var: {theoretical_var:.5f})')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Hustota pravděpodobnosti')
    ax.legend()
    ax.grid(alpha=0.3)

# 1. GPS Šum (Sloučíme X a Y odchylky pro jeden robustní histogram)
v_gps_all = np.concatenate([v_k_np[:, 0], v_k_np[:, 2]])
# Odřízneme extrémní outliery (např. 1 % dat) jen pro hezčí vizualizaci
v_gps_all = v_gps_all[np.abs(v_gps_all) < np.percentile(np.abs(v_gps_all), 99)]
plot_dist(axs[0], v_gps_all, best_var_gps, "Šum měření GPS ($v^{GPS}_k$)", "Chyba pozice [m]")

# 2. Odometrie Šum
v_odo_all = np.concatenate([v_k_np[:, 1], v_k_np[:, 3]])
v_odo_all = v_odo_all[np.abs(v_odo_all) < np.percentile(np.abs(v_odo_all), 99)]
plot_dist(axs[1], v_odo_all, best_var_odo, "Šum měření Odometrie ($v^{Odo}_k$)", "Chyba rychlosti [m/s]")

# 3. Procesní Šum (Nejistota modelu, q_scale se projeví na rychlosti)
w_vel_all = np.concatenate([w_k_np[:, 1], w_k_np[:, 3]])
w_vel_all = w_vel_all[np.abs(w_vel_all) < np.percentile(np.abs(w_vel_all), 99)]
# Teoretická variance pro rychlostní stav je q_scale * DT. Vzhledem k tomu, že DT=1, je to q_scale.
plot_dist(axs[2], w_vel_all, best_q_scale, "Šum procesu na rychlosti ($w^{vel}_k$)", "Nejistota modelu [m/s]")

plt.tight_layout()
plt.show()