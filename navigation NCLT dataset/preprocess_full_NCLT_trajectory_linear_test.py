import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ==============================================================================
# 1. KONFIGURACE
# ==============================================================================
SESSION_DATE = '2012-01-22'
DT = 1.0  # Časový krok (1 Hz)

BASE_DIR = os.getcwd()
# Cesty si případně uprav dle své struktury
GT_FILE = os.path.join(BASE_DIR, 'ground_truth', f'groundtruth_{SESSION_DATE}.csv')
SENSOR_DIR = os.path.join(BASE_DIR, 'data', 'sensor', SESSION_DATE)
OUTPUT_DIR = os.path.join(BASE_DIR, f'preprocessed_NCLT_FULL') # Nový výstupní adresář

def gps_to_local_coord(lat, lon):
    """Převod Lat/Lon na lokální metry dle NCLT specifikace."""
    LAT_0 = 0.738167915410646
    LON_0 = -1.46098650670922
    re = 6378135
    rp = 6356750
    r_ns = pow(re * rp, 2) / pow(
        pow(re * np.cos(LAT_0), 2) + pow(rp * np.sin(LAT_0), 2), 3 / 2)
    r_ew = pow(re, 2) / pow(
        pow(re * np.cos(LAT_0), 2) + pow(rp * np.sin(LAT_0), 2), 1 / 2)
    
    x = np.sin(lat - LAT_0) * r_ns
    y = np.sin(lon - LON_0) * r_ew * np.cos(LAT_0)
    return x, y

def load_csv_as_df(path, cols=None, names=None):
    if not os.path.exists(path):
        # Fallback cesty
        alt_path = os.path.join(os.path.dirname(path), SESSION_DATE, os.path.basename(path))
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Soubor nenalezen: {path}")
    
    print(f"  -> Načítám: {os.path.basename(path)}")
    df = pd.read_csv(path, header=None, usecols=cols, names=names, 
                     on_bad_lines='skip', low_memory=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['t'] = df['t'] / 1e6
    df = df.set_index('t').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

def process_data():
    print(f"=== ZPRACOVÁNÍ CELÉ TRAJEKTORIE {SESSION_DATE} ===")
    
    # 1. Ground Truth
    df_gt_raw = load_csv_as_df(GT_FILE, cols=[0, 1, 2], names=['t', 'gt_px', 'gt_py'])
    
    # 2. GPS
    df_gps_raw = load_csv_as_df(os.path.join(SENSOR_DIR, 'gps.csv'), 
                                cols=[0, 3, 4], names=['t', 'lat', 'lon'])
    x, y = gps_to_local_coord(df_gps_raw['lat'].values, df_gps_raw['lon'].values)
    df_gps_raw['gps_px'] = x + 76.50582406697139
    df_gps_raw['gps_py'] = y + 108.31373031919006
    df_gps_raw = df_gps_raw[['gps_px', 'gps_py']] 

    # 3. Wheels
    df_whl_raw = load_csv_as_df(os.path.join(SENSOR_DIR, 'wheels.csv'), 
                                cols=[0, 1, 2], names=['t', 'vl', 'vr'])
    
    # 4. Euler (Heading)
    df_eul_raw = load_csv_as_df(os.path.join(SENSOR_DIR, 'ms25_euler.csv'), 
                                cols=[0, 3], names=['t', 'yaw'])

    # --- SYNCHRONIZACE ---
    print("  -> Synchronizace (1 Hz)...")
    t_start = df_gt_raw.index[0]
    t_end = df_gt_raw.index[-1]
    t_grid = np.arange(t_start, t_end, DT)
    
    def reindex_and_interp(df, target_index, angular_cols=None):
        if angular_cols is None: angular_cols = []
        combined = df.index.union(target_index).sort_values().drop_duplicates()
        df_re = df.reindex(combined)
        # Unwrap angles
        for col in angular_cols:
            if col in df_re.columns:
                valid = df_re[col].notna()
                df_re.loc[valid, col] = np.unwrap(df_re.loc[valid, col].values)
        
        df_int = df_re.interpolate(method='index').dropna()
        # Wrap angles back
        for col in angular_cols:
            if col in df_int.columns:
                df_int[col] = (df_int[col] + np.pi) % (2 * np.pi) - np.pi
        return df_int.reindex(target_index).dropna()

    df_gt = reindex_and_interp(df_gt_raw, t_grid)
    df_gps = reindex_and_interp(df_gps_raw, t_grid)
    df_whl = reindex_and_interp(df_whl_raw, t_grid)
    df_eul = reindex_and_interp(df_eul_raw, t_grid, angular_cols=['yaw'])

    # --- VÝPOČTY ---
    df = pd.concat([df_gt, df_gps, df_whl, df_eul], axis=1).dropna()
    
    df['gt_vx'] = df['gt_px'].diff() / DT
    df['gt_vy'] = df['gt_py'].diff() / DT
    
    df['v_lin'] = (df['vl'] + df['vr']) / 2.0
    df['odo_vx'] = df['v_lin'] * np.cos(df['yaw'])
    df['odo_vy'] = df['v_lin'] * np.sin(df['yaw'])

    # --- FINALIZACE ---
    cols = ['gt_px', 'gt_vx', 'gt_py', 'gt_vy', 'gps_px', 'odo_vx', 'gps_py', 'odo_vy']
    df_final = df[cols].dropna()
    
    print(f"  -> Finální délka trajektorie: {len(df_final)} kroků")

    X_np = df_final[['gt_px', 'gt_vx', 'gt_py', 'gt_vy']].values.astype(np.float32)
    Y_np = df_final[['gps_px', 'odo_vx', 'gps_py', 'odo_vy']].values.astype(np.float32)

    return torch.tensor(X_np), torch.tensor(Y_np)

def save_full_trajectory(X, Y):
    """Uloží celou trajektorii jako jeden dataset."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Ukládáme jako seznam s jedním prvkem (jedna dlouhá trajektorie)
    # To zajistí kompatibilitu s Dataloaderem, který čeká list sekvencí
    full_dataset = [{'ground_truth': X, 'filtered_gps': Y}]
    
    save_path = os.path.join(OUTPUT_DIR, 'test_full_trajectory.pt')
    torch.save(full_dataset, save_path)
    
    print(f"✅ Uložena CELÁ trajektorie do: {save_path}")

if __name__ == "__main__":
    try:
        X, Y = process_data()
        save_full_trajectory(X, Y)
    except Exception as e:
        print(f"\n❌ CHYBA: {e}")