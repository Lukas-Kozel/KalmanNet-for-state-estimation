import torch
import torch.distributions as dist
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# === KONFIGURACE ===
DATA_DIR = 'preprocessed_NCLT_trajectory-2012-01-22-angle-update'
DT = 1.0
# SUBSET_LEN = 3000
NUM_TRIALS = 300000  # Díky vektorizaci můžeš zkusit klidně 50 000

# Detekce zařízení (GPU je ideální)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Běžím na: {device}")

# 1. Načtení dat
try:
    train_list = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    # Převedeme na torch rovnou a pošleme na device
    gt = train_list[0]['ground_truth'].float().to(device)
    meas = train_list[0]['filtered_gps'].float().to(device)
    # Zjistíme skutečnou délku
    FULL_LEN = gt.shape[0]
    
    print(f"✅ Data načtena.")
    print(f"   Celková dostupná délka trajektorie: {FULL_LEN} kroků")
except Exception as e:
    print(f"CHYBA: {e}")
    exit()

# 2. DEFINICE PARAMETRŮ (Search Space)
# Abychom pokryli tvé hodnoty, hledáme logaritmicky v širokém rozsahu
print(f"Generuji {NUM_TRIALS} kombinací parametrů...")

# A) q_scale (variance procesu)
# Hledáme q_scale cca 0.0001 -> q_std cca 0.01
# Rozsah q_std: 0.001 až 1.0 (odpovídá q_scale 1e-6 až 1.0)
q_std_vals = torch.pow(10, dist.Uniform(np.log10(0.0001), np.log10(5.0)).sample((NUM_TRIALS,)).to(device))

# B) var_gps (variance měření pozice)
# Cíl je cca 39.0. Rozsah: 1.0 až 100.0
r_gps_vals = torch.pow(10, dist.Uniform(np.log10(1.0), np.log10(100.0)).sample((NUM_TRIALS,)).to(device))

# C) var_odo (variance měření rychlosti)
# Cíl je cca 0.00057. Rozsah: 0.00001 až 0.01
r_odo_vals = torch.pow(10, dist.Uniform(np.log10(1e-5), np.log10(10)).sample((NUM_TRIALS,)).to(device))

# 3. PŘÍPRAVA MATIC (Vektorizace)
# Vytvoříme [NUM_TRIALS, 4, 4] matice pro Q a R
# Batch size = NUM_TRIALS

# --- Matice F (nemění se) ---
F = torch.eye(4, device=device).unsqueeze(0).repeat(NUM_TRIALS, 1, 1)
F[:, 0, 1] = DT
F[:, 2, 3] = DT

# --- Matice H (nemění se) ---
H = torch.eye(4, device=device).unsqueeze(0).repeat(NUM_TRIALS, 1, 1)

# --- Matice Q (závisí na q_std) ---
# Discrete White Noise Acceleration model
Q_base = torch.tensor([[DT**3/3, DT**2/2], [DT**2/2, DT]], device=device)
Q = torch.zeros((NUM_TRIALS, 4, 4), device=device)
# q_std_vals je [N], musíme broadcastovat
q_var = q_std_vals ** 2  # q_scale
Q[:, 0:2, 0:2] = Q_base.unsqueeze(0) * q_var.view(-1, 1, 1)
Q[:, 2:4, 2:4] = Q_base.unsqueeze(0) * q_var.view(-1, 1, 1)

# --- Matice R (závisí na r_gps a r_odo) ---
R = torch.zeros((NUM_TRIALS, 4, 4), device=device)
R[:, 0, 0] = r_gps_vals
R[:, 1, 1] = r_odo_vals
R[:, 2, 2] = r_gps_vals
R[:, 3, 3] = r_odo_vals

# 4. VEKTORIZOVANÝ KF LOOP
# Stav: [NUM_TRIALS, 4, 1]
x = gt[0].unsqueeze(0).unsqueeze(2).repeat(NUM_TRIALS, 1, 1) # Start na GT
p0_diag = torch.tensor([10.0, 1.0, 10.0, 1.0], device=device)
P = torch.diag(p0_diag).unsqueeze(0).repeat(NUM_TRIALS, 1, 1)

# Akumulátor chyby [NUM_TRIALS]
mse_accum = torch.zeros(NUM_TRIALS, device=device)

print("Spouštím vektorizovaný KF...")
# Smyčka přes čas (nelze vektorizovat, KF je rekurentní)
for t in tqdm(range(len(meas)), desc="Simulace KF"):
    # Měření v čase t: [4] -> [NUM_TRIALS, 4, 1]
    y = meas[t].unsqueeze(0).unsqueeze(2).repeat(NUM_TRIALS, 1, 1)
    
    # 1. Predict
    # x = F @ x
    x = torch.bmm(F, x)
    # P = F @ P @ F.T + Q
    P = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q
    
    # 2. Update
    # S = H @ P @ H.T + R
    S = torch.bmm(torch.bmm(H, P), H.transpose(1, 2)) + R
    
    # K = P @ H.T @ inv(S)
    # Použijeme torch.linalg.solve pro stabilitu místo inv
    # Chceme K = P H^T S^-1 -> K S = P H^T
    # solve(A, B) řeší AX = B. Tady je to trochu jinak, použijeme batch inverse, je to rychlejší pro malé matice
    S_inv = torch.linalg.inv(S)
    K = torch.bmm(torch.bmm(P, H.transpose(1, 2)), S_inv)
    
    # Innovation: y - H x
    inov = y - torch.bmm(H, x)
    
    # x = x + K @ inov
    x = x + torch.bmm(K, inov)
    
    # P = (I - KH) P (I - KH)^T + K R K^T (Joseph form)
    I = torch.eye(4, device=device).unsqueeze(0)
    KH = torch.bmm(K, H)
    I_KH = I - KH
    P = torch.bmm(torch.bmm(I_KH, P), I_KH.transpose(1, 2)) + \
        torch.bmm(torch.bmm(K, R), K.transpose(1, 2))
    
    # 3. MSE Calculation (Batch)
    # gt[t]: [4]
    diff = x.squeeze(2) - gt[t].unsqueeze(0) # [N, 4]
    dist_sq = diff[:, 0]**2 + diff[:, 2]**2  # Jen X a Y
    mse_accum += dist_sq

# Průměr přes čas
mse_final = mse_accum / len(meas)

# 5. VYHODNOCENÍ
best_idx = torch.argmin(mse_final).item()
best_mse = mse_final[best_idx].item()

# Získání původních parametrů
best_q_scale = (q_std_vals[best_idx] ** 2).item()
best_var_gps = r_gps_vals[best_idx].item()
best_var_odo = r_odo_vals[best_idx].item()

print(f"\n=== VÍTĚZNÉ NASTAVENÍ (z {NUM_TRIALS} pokusů) ===")
print(f"MSE: {best_mse:.4f}")
print(f"RMSE: {np.sqrt(best_mse):.4f} m")
print("-" * 30)
print(f"q_scale = {best_q_scale:.8f}")
print(f"var_gps = {best_var_gps:.6f}")
print(f"var_odo = {best_var_odo:.6f}")

# Vykreslení (převedeme na CPU)
q_cpu = (q_std_vals**2).cpu().numpy()
mse_cpu = mse_final.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(q_cpu, mse_cpu, alpha=0.5, s=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('q_scale')
plt.ylabel('MSE')
plt.title('Závislost MSE na q_scale')
plt.axvline(best_q_scale, color='r', linestyle='--', label='Best Q')
plt.legend()
plt.show()