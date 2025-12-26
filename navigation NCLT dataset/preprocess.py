import os
import os.path as osp
import numpy as np
import pandas as pd
import scipy.interpolate
import torch
from tqdm import tqdm

# === KONFIGURACE CEST ===
DATA_ROOT = 'data' 
OUTPUT_DIR = os.path.join(DATA_ROOT, 'processed')

# === POMOCNÉ FUNKCE ===

def _compute_gps_conversion_params():
    LAT_0 = 0.738167915410646
    LON_0 = -1.46098650670922
    re = 6378135
    rp = 6356750
    r_ns = pow(re * rp, 2) / pow(
        pow(re * np.cos(LAT_0), 2) + pow(rp * np.sin(LAT_0), 2), 3 / 2)
    r_ew = pow(re, 2) / pow(
        pow(re * np.cos(LAT_0), 2) + pow(rp * np.sin(LAT_0), 2), 1 / 2)
    return (r_ns, r_ew, LAT_0, LON_0)

def gps_to_local_coord(lat, lon):
    r_ns, r_ew, LAT_0, LON_0 = _compute_gps_conversion_params()
    x = np.sin(lat - LAT_0) * r_ns
    y = np.sin(lon - LON_0) * r_ew * np.cos(LAT_0)
    return (x, y)

def find_nearest_index(array, time):
    diff_arr = array - time
    idx = np.where(diff_arr <= 0, diff_arr, -np.inf).argmax()
    return idx

def get_sensor_file_path(data_dir, data_date, filename):
    """
    Chytrá funkce pro nalezení souboru.
    Řeší problém 'double nesting' (např. sensor/2012-01-08/2012-01-08/gps.csv).
    """
    # 1. Zkusíme standardní cestu: data/sensor/DATUM/soubor
    path1 = osp.join(data_dir, 'sensor', data_date, filename)
    if os.path.exists(path1):
        return path1
    
    # 2. Zkusíme vnořenou cestu: data/sensor/DATUM/DATUM/soubor
    path2 = osp.join(data_dir, 'sensor', data_date, data_date, filename)
    if os.path.exists(path2):
        return path2
        
    # Pokud nenajdeme, vrátíme tu první (pro výpis chyby)
    return path1

# === ČTECÍ FUNKCE (Upravené pro robustní cesty) ===

def read_gps(data_dir, data_date, use_rtk=False):
    filename = 'gps_rtk.csv' if use_rtk else 'gps.csv'
    filepath = get_sensor_file_path(data_dir, data_date, filename) # <--- ZMĚNA
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GPS file not found: {filepath}")

    gps = np.loadtxt(filepath, delimiter=',')
    gps = np.delete(gps, np.where((gps[:, 1] < 3))[0], axis=0)
    
    t = gps[:, 0] / 1e6
    lat = gps[:, 3]
    lng = gps[:, 4]
    
    x, y = gps_to_local_coord(lat, lng)
    x = x + 76.50582406697139
    y = y + 108.31373031919006
    
    mask = (x >= -350) & (x <= 120) & (y >= -750) & (y <= 150)
    gps_data = np.vstack((t, x, y)).T
    gps_data = gps_data[mask]
    
    return gps_data

def read_imu(data_dir, data_date):
    filepath = get_sensor_file_path(data_dir, data_date, 'ms25.csv') # <--- ZMĚNA
    ms25 = np.loadtxt(filepath, delimiter=',')
    
    t = ms25[:, 0] / 1e6
    accel_x = ms25[:, 4]
    accel_y = ms25[:, 5]
    rot_h = ms25[:, 9]

    df_ax = pd.DataFrame(accel_x).rolling(50, min_periods=1).mean().to_numpy().flatten()
    df_ay = pd.DataFrame(accel_y).rolling(50, min_periods=1).mean().to_numpy().flatten()
    
    return np.vstack((t, df_ax, df_ay, rot_h)).T

def read_euler(data_dir, data_date):
    filepath = get_sensor_file_path(data_dir, data_date, 'ms25_euler.csv') # <--- ZMĚNA
    euler = np.loadtxt(filepath, delimiter=',')
    t = euler[:, 0] / 1e6
    h_OG = euler[:, 3]
    return np.vstack((t, h_OG)).T

def read_wheels(data_dir, data_date):
    filepath = get_sensor_file_path(data_dir, data_date, 'wheels.csv') # <--- ZMĚNA
    wheels = np.loadtxt(filepath, delimiter=',')
    t = wheels[:, 0] / 1e6
    v_left = wheels[:, 1]
    v_right = wheels[:, 2]
    return np.vstack((t, v_left, v_right)).T

def read_ground_truth(data_dir, data_date):
    # GT je obvykle v rootu 'ground_truth', tam vnořování nebývá
    filepath_gt = osp.join(data_dir, 'ground_truth', f'groundtruth_{data_date}.csv')
    
    # Ale kovariance je v senzorech, tam vnořování být může
    filepath_cov = get_sensor_file_path(data_dir, data_date, 'odometry_cov_100hz.csv') # <--- ZMĚNA
    
    if not os.path.exists(filepath_cov):
        # Fallback na GT časy
        gt = np.loadtxt(filepath_gt, delimiter=',')
        t = gt[:, 0] / 1e6
        x = gt[:, 1]
        y = gt[:, 2]
        yaw = gt[:, 5]
        return np.vstack((t, x, y, yaw)).T

    gt = np.loadtxt(filepath_gt, delimiter=',')
    cov = np.loadtxt(filepath_cov, delimiter=',')
    
    gt = gt[2:, :]
    t_cov = cov[:, 0]
    
    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0, fill_value='extrapolate')
    pose_gt = interp(t_cov)
    
    t = t_cov / 1e6
    x = pose_gt[:, 0]
    y = pose_gt[:, 1]
    yaw = pose_gt[:, 5]
    
    return np.vstack((t, x, y, yaw)).T
def preprocess_session(data_dir, data_date, use_rtk=False):
    print(f"Processing {data_date}...")
    
    try:
        ground_truth = read_ground_truth(data_dir, data_date)
        gps_data = read_gps(data_dir, data_date, use_rtk)
        imu_data = read_imu(data_dir, data_date)
        euler_data = read_euler(data_dir, data_date)
        wheel_data = read_wheels(data_dir, data_date)
    except Exception as e:
        print(f"Skipping {data_date}: {e}")
        return None

    x_true = ground_truth[:, 1]
    y_true = ground_truth[:, 2]
    theta_true = ground_truth[:, 3]
    
    KALMAN_FILTER_RATE = 1.0 
    dt = 1.0 / KALMAN_FILTER_RATE
    
    t_start = ground_truth[0, 0]
    t_end = ground_truth[-1, 0]
    t_grid = np.arange(t_start, t_end, dt)
    
    # Seznamy pro sběr dat
    filtered_gps_list = [] # S NaNs (Tvoje původní - pro EKF)
    gps_filled_list = []   # <--- NOVÉ: Bez NaNs (Nearest Neighbor - pro metriku autorů)
    imu_sensor_list = []
    filtered_wheel_list = []
    gt_list = []
    
    prev_gps_idx = -1
    prev_wheel_idx = -1
    
    # Rozbalení dat pro rychlý přístup
    gps_t, gps_x, gps_y = gps_data[:, 0], gps_data[:, 1], gps_data[:, 2]
    imu_t, imu_ax, imu_ay, imu_w = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2], imu_data[:, 3]
    eul_t, eul_h = euler_data[:, 0], euler_data[:, 1]
    whl_t, whl_vl, whl_vr = wheel_data[:, 0], wheel_data[:, 1], wheel_data[:, 2]
    gt_t = ground_truth[:, 0]
    
    x_est_init = np.array([x_true[0], y_true[0], 0, 0, theta_true[0], 0])
    
    # Hlavní smyčka přes časovou mřížku
    for k in range(len(t_grid)):
        curr_t = t_grid[k]
        
        # Najdi indexy nejbližších minulých měření
        idx_imu = find_nearest_index(imu_t, curr_t)
        idx_eul = find_nearest_index(eul_t, curr_t)
        idx_gps = find_nearest_index(gps_t, curr_t)
        idx_whl = find_nearest_index(whl_t, curr_t)
        idx_gt = find_nearest_index(gt_t, curr_t)
        
        ax = imu_ax[idx_imu]
        ay = imu_ay[idx_imu]
        omega = imu_w[idx_imu]
        theta = eul_h[idx_eul] 
        
        # 1. IMU
        imu_sensor_list.append([ax, ay, theta, omega])
        
        # 2. GPS - Dvě verze
        
        # Verze A: "Filtered" (s NaN) - pokud se index nezměnil, nemáme nová data
        if idx_gps != prev_gps_idx:
            filtered_gps_list.append([gps_x[idx_gps], gps_y[idx_gps]])
            prev_gps_idx = idx_gps
        else:
            filtered_gps_list.append([np.nan, np.nan])

        # Verze B: "Authors" (Filled) - <--- NOVÉ
        # Vždy vezmeme hodnotu na aktuálním indexu. Pokud data nepřišla,
        # find_nearest_index vrátí stejný index jako minule -> držíme starou hodnotu.
        gps_filled_list.append([gps_x[idx_gps], gps_y[idx_gps]])
            
        # 3. Odometrie
        if idx_whl != prev_wheel_idx:
            filtered_wheel_list.append([whl_vl[idx_whl], whl_vr[idx_whl]])
            prev_wheel_idx = idx_whl
        else:
            filtered_wheel_list.append([np.nan, np.nan])
            
        # 4. GT
        gt_list.append([x_true[idx_gt], y_true[idx_gt], theta_true[idx_gt]])

    processed_data = {
        'filtered_gps': torch.tensor(filtered_gps_list, dtype=torch.float32),
        'gps': torch.tensor(gps_filled_list, dtype=torch.float32), # <--- NOVÝ KLÍČ
        'imu': torch.tensor(imu_sensor_list, dtype=torch.float32),
        'filtered_wheel': torch.tensor(filtered_wheel_list, dtype=torch.float32),
        'ground_truth': torch.tensor(gt_list, dtype=torch.float32),
        'initial_state': torch.tensor(x_est_init, dtype=torch.float32),
        'data_date': data_date,
        't': torch.tensor(t_grid, dtype=torch.float32)
    }
    
    return processed_data

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    sensor_dir = os.path.join(DATA_ROOT, 'sensor')
    if not os.path.exists(sensor_dir):
        print(f"Error: Directory {sensor_dir} does not exist.")
        exit(1)
        
    available_sessions = sorted(os.listdir(sensor_dir))
    print(f"Found {len(available_sessions)} sessions in {sensor_dir}")
    
    processed_all = []
    
    for date in tqdm(available_sessions):
        if not os.path.isdir(os.path.join(sensor_dir, date)):
            continue
            
        data = preprocess_session(DATA_ROOT, date, use_rtk=False)
        if data is not None:
            processed_all.append(data)
            
    # Hardcoded split from paper
    train_dates = [
        '2012-08-20', '2012-02-19', '2012-08-04', '2012-03-25',
        '2012-06-15', '2012-01-22', '2012-01-15', '2012-01-08',
        '2012-02-04', '2012-02-02', '2012-05-11', '2012-04-29',
        '2012-02-18', '2012-03-31', '2012-12-01', '2012-09-28',
        '2012-02-05', '2012-11-17', '2012-02-12', '2012-05-26',
        '2012-03-17', '2012-10-28'
    ]
    val_dates = ['2013-01-10', '2013-02-23']
    test_dates = ['2012-11-16', '2013-04-05', '2012-11-04']
    
    train_set = []
    val_set = []
    test_set = []
    
    print("\nSorting sessions into splits...")
    for data in processed_all:
        d = data['data_date']
        if d in test_dates:
            test_set.append(data)
        elif d in val_dates:
            val_set.append(data)
        else:
            train_set.append(data)
            
    print(f"Train: {len(train_set)} trajectories")
    print(f"Val:   {len(val_set)} trajectories")
    print(f"Test:  {len(test_set)} trajectories")
    
    print(f"Saving to {OUTPUT_DIR}...")
    torch.save(train_set, os.path.join(OUTPUT_DIR, 'train.pt'))
    torch.save(val_set, os.path.join(OUTPUT_DIR, 'val.pt'))
    torch.save(test_set, os.path.join(OUTPUT_DIR, 'test.pt'))
    
    print("✅ Done!")