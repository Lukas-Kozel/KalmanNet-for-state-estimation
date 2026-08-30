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
NUM_TRIALS = 50000  # Shortened for testing

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# 1. Data loading
try:
    train_list = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    gt = train_list[0]['ground_truth'].float().to(device)
    meas = train_list[0]['filtered_gps'].float().to(device)
    FULL_LEN = gt.shape[0]
    print(f"✅ Data loaded. Total length: {FULL_LEN} steps")
except Exception as e:
    print(f"CHYBA: {e}")
    exit()

# 2. PARAMETER DEFINITIONS (Search Space)
print(f"Generating {NUM_TRIALS} parameter combinations...")
q_std_vals = torch.pow(10, dist.Uniform(np.log10(0.0001), np.log10(5.0)).sample((NUM_TRIALS,)).to(device))
r_gps_vals = torch.pow(10, dist.Uniform(np.log10(1.0), np.log10(100.0)).sample((NUM_TRIALS,)).to(device))
r_odo_vals = torch.pow(10, dist.Uniform(np.log10(1e-5), np.log10(10)).sample((NUM_TRIALS,)).to(device))

# 3. MATRIX PREPARATION
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

# 4. VECTORIZED KF LOOP
x = gt[0].unsqueeze(0).unsqueeze(2).repeat(NUM_TRIALS, 1, 1) 
p0_diag = torch.tensor([10.0, 1.0, 10.0, 1.0], device=device)
P = torch.diag(p0_diag).unsqueeze(0).repeat(NUM_TRIALS, 1, 1)
mse_accum = torch.zeros(NUM_TRIALS, device=device)

print("Running vectorized KF...")
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

print(f"\n=== WINNING CONFIGURATION (out of {NUM_TRIALS} trials) ===")
print(f"MSE: {best_mse:.4f}")
print("-" * 30)
print(f"Grid-Search q_scale: {best_q_scale:.8f}")
print(f"Grid-Search var_gps: {best_var_gps:.6f}")
print(f"Grid-Search var_odo: {best_var_odo:.6f}")


# ==============================================================================
# 5. EMPIRICAL NOISE COMPUTATION (Task from the supervisor)
# ==============================================================================
print("\n=== EMPIRICAL NOISE ANALYSIS (GROUND TRUTH) ===")

# A) Measurement noise: v_k = z_k - H * x_k
# Since H is identity, this is z_k - x_k
v_k = meas - gt
v_k_var = torch.var(v_k, dim=0).cpu().numpy()

emp_var_gps = (v_k_var[0] + v_k_var[2]) / 2.0  # Average of X and Y variance from GPS
emp_var_odo = (v_k_var[1] + v_k_var[3]) / 2.0  # Average of Vx and Vy variance from Odometry

# B) Process noise: w_k = x_{k+1} - F * x_k
gt_k = gt[:-1]
gt_k_plus_1 = gt[1:]
# Multiply F * x_k (using matrix multiplication)
pred_gt = torch.matmul(gt_k, F_base.T)
w_k = gt_k_plus_1 - pred_gt
w_k_var = torch.var(w_k, dim=0).cpu().numpy()

# Extraction of q_scale: From the definition of matrix Q we know that Q[1,1] and Q[3,3] equal q_scale * DT.
# Since DT = 1.0, the empirical variance of the velocity components in vector w_k IS directly q_scale.
emp_q_scale = (w_k_var[1] + w_k_var[3]) / 2.0

print(f"Empirical q_scale:   {emp_q_scale:.8f} (Grid-Search found: {best_q_scale:.8f})")
print(f"Empirical var_gps:   {emp_var_gps:.6f} (Grid-Search found: {best_var_gps:.6f})")
print(f"Empirical var_odo:   {emp_var_odo:.6f} (Grid-Search found: {best_var_odo:.6f})")

# ==============================================================================
# 6. VISUALIZATION (HISTOGRAMS vs. GAUSSIANS)
# ==============================================================================
# Convert data to CPU for numpy and pyplot
v_k_np = v_k.cpu().numpy()
w_k_np = w_k.cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Noise distribution analysis: Data vs. Grid-Search parameters", fontsize=16, fontweight='bold')

# Plotting function
def plot_dist(ax, data, theoretical_var, title, xlabel):
    ax.hist(data, bins=100, density=True, alpha=0.6, color='blue', label='Empirical data (Histogram)')
    
    # Theoretical Gaussian curve with variance found by Grid-Search
    mu = 0
    sigma = np.sqrt(theoretical_var)
    x_val = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y_val = stats.norm.pdf(x_val, mu, sigma)
    
    ax.plot(x_val, y_val, 'r-', lw=2, label=f'GS Model\n(Var: {theoretical_var:.5f})')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.legend()
    ax.grid(alpha=0.3)

# 1. GPS Noise (Merge X and Y deviations into one robust histogram)
v_gps_all = np.concatenate([v_k_np[:, 0], v_k_np[:, 2]])
# Trim extreme outliers (e.g. 1% of data) just for nicer visualization
v_gps_all = v_gps_all[np.abs(v_gps_all) < np.percentile(np.abs(v_gps_all), 99)]
plot_dist(axs[0], v_gps_all, best_var_gps, "GPS measurement noise ($v^{GPS}_k$)", "Chyba pozice [m]")

# 2. Odometry Noise
v_odo_all = np.concatenate([v_k_np[:, 1], v_k_np[:, 3]])
v_odo_all = v_odo_all[np.abs(v_odo_all) < np.percentile(np.abs(v_odo_all), 99)]
plot_dist(axs[1], v_odo_all, best_var_odo, "Odometry measurement noise ($v^{Odo}_k$)", "Chyba rychlosti [m/s]")

# 3. Process Noise (model uncertainty, q_scale affects velocity)
w_vel_all = np.concatenate([w_k_np[:, 1], w_k_np[:, 3]])
w_vel_all = w_vel_all[np.abs(w_vel_all) < np.percentile(np.abs(w_vel_all), 99)]
# Theoretical variance for the velocity state is q_scale * DT. Since DT=1, it equals q_scale.
plot_dist(axs[2], w_vel_all, best_q_scale, "Process noise on velocity ($w^{vel}_k$)", "Nejistota modelu [m/s]")

plt.tight_layout()
plt.show()