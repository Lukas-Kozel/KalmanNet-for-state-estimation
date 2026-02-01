import os
import numpy as np
import pandas as pd
import torch

# ==============================================================================
# 1. KONFIGURACE
# ==============================================================================
SESSION_DATE = '2012-01-22'
DT = 1.0  # Časový krok (1 Hz)

# Cesty
BASE_DIR = os.getcwd()
GT_FILE = os.path.join(BASE_DIR, 'ground_truth', f'groundtruth_{SESSION_DATE}.csv')
SENSOR_DIR = os.path.join(BASE_DIR, 'data', 'sensor', SESSION_DATE)
OUTPUT_DIR = os.path.join(BASE_DIR, f'preprocessed_NCLT_trajectory-{SESSION_DATE}')

# ==============================================================================
# 2. POMOCNÉ FUNKCE (Geometrie)
# ==============================================================================
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
    """
    Načte CSV, vynutí numerický typ a nastaví první sloupec (čas) jako index.
    Robustní vůči hlavičkám a komentářům.
    """
    if not os.path.exists(path):
        # Zkusíme alternativní cestu (vnořené složky v NCLT)
        alt_path = os.path.join(os.path.dirname(path), SESSION_DATE, os.path.basename(path))
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Soubor nenalezen: {path}")
    
    print(f"  -> Načítám: {os.path.basename(path)}")
    
    # Načteme CSV (ignorujeme chyby na řádcích s jiným počtem sloupců)
    df = pd.read_csv(path, header=None, usecols=cols, names=names, on_bad_lines='skip')
    
    # --- OPRAVA CHYBY "Mixed Types" ---
    # Vynutíme konverzi všech sloupců na čísla. 
    # 'coerce' změní texty (hlavičky) na NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Odstraníme řádky, kde vznikly NaN (to jsou ty hlavičky/chyby)
    df = df.dropna()
    
    # Nyní už máme jistotu, že 't' je číslo
    df['t'] = df['t'] / 1e6  # Konverze us -> sekundy
    df = df.set_index('t').sort_index()
    
    # Odstraníme duplicity v čase
    df = df[~df.index.duplicated(keep='first')]
    
    return df

# ==============================================================================
# 3. HLAVNÍ PROCESING
# ==============================================================================
def process_data():
    print(f"=== ZPRACOVÁNÍ TRAJEKTORIE {SESSION_DATE} ===")
    
    # --- A. NAČTENÍ DAT ---
    # 1. Ground Truth [t, x, y, z, r, p, h] -> bereme [t, x, y]
    df_gt_raw = load_csv_as_df(GT_FILE, cols=[0, 1, 2], names=['t', 'gt_px', 'gt_py'])
    
    # 2. GPS [t, mode, num_sat, lat, lng, alt, ...] -> bereme [t, lat, lng]
    df_gps_raw = load_csv_as_df(os.path.join(SENSOR_DIR, 'gps.csv'), 
                                cols=[0, 3, 4], names=['t', 'lat', 'lon'])
    
    # Převod GPS na lokální metry
    # Pozor: Lat/Lon musí být numpy array (float), což load_csv_as_df nyní zaručuje
    x, y = gps_to_local_coord(df_gps_raw['lat'].values, df_gps_raw['lon'].values)
    df_gps_raw['gps_px'] = x + 76.50582406697139
    df_gps_raw['gps_py'] = y + 108.31373031919006
    df_gps_raw = df_gps_raw[['gps_px', 'gps_py']] # Necháme jen X, Y

    # 3. Wheels (Odometrie rychlost) [t, vl, vr]
    df_whl_raw = load_csv_as_df(os.path.join(SENSOR_DIR, 'wheels.csv'), 
                                cols=[0, 1, 2], names=['t', 'vl', 'vr'])
    
    # 4. Euler (FOG Gyro) [t, r, p, h] -> bereme [t, h] (heading/yaw)
    df_eul_raw = load_csv_as_df(os.path.join(SENSOR_DIR, 'ms25_euler.csv'), 
                                cols=[0, 3], names=['t', 'yaw'])

    # --- B. SYNCHRONIZACE (Resampling na 1 Hz) ---
    print("  -> Synchronizace a interpolace dat...")
    
    t_start = df_gt_raw.index[0]
    t_end = df_gt_raw.index[-1]
    t_grid = np.arange(t_start, t_end, DT)
    
    def reindex_and_interp(df, target_index):
        # Robustní interpolace: Sjednotíme indexy, seřadíme, interpolujeme, ořízneme
        combined_index = df.index.union(target_index).sort_values()
        # Drop duplicates v indexu pro jistotu
        combined_index = combined_index.drop_duplicates()
        
        df_interp = df.reindex(combined_index).interpolate(method='index').dropna()
        
        # Vrátíme jen cílové časy (intersection)
        # Použijeme reindex s tolerancí, kdyby float aritmetika neseděla přesně, 
        # ale zde jsme unionovali, takže by to mělo sedět.
        return df_interp.reindex(target_index).dropna()

    # 1. Ground Truth
    df_gt = reindex_and_interp(df_gt_raw, t_grid)
    
    # 2. GPS
    df_gps = reindex_and_interp(df_gps_raw, t_grid)
    
    # 3. Wheels
    df_whl = reindex_and_interp(df_whl_raw, t_grid)
    
    # 4. Euler
    df_eul = reindex_and_interp(df_eul_raw, t_grid)

    # --- C. VÝPOČTY (Rychlosti) ---
    print("  -> Výpočet globálních rychlostí...")
    
    # Sloučení (Inner join přes index = čas)
    df = pd.concat([df_gt, df_gps, df_whl, df_eul], axis=1).dropna()
    
    # 1. GT Rychlost (Derivace pozice)
    df['gt_vx'] = df['gt_px'].diff() / DT
    df['gt_vy'] = df['gt_py'].diff() / DT
    
    # 2. ODO Rychlost (Wheels + Gyro)
    df['v_lin'] = (df['vl'] + df['vr']) / 2.0
    df['odo_vx'] = df['v_lin'] * np.cos(df['yaw'])
    df['odo_vy'] = df['v_lin'] * np.sin(df['yaw'])

    # --- D. ČIŠTĚNÍ ---
    cols_of_interest = ['gt_px', 'gt_vx', 'gt_py', 'gt_vy', 
                        'gps_px', 'odo_vx', 'gps_py', 'odo_vy']
    
    df_final = df[cols_of_interest].dropna()
    
    print(f"  -> Finální počet vzorků: {len(df_final)}")

    # Příprava Tensorů
    X_np = df_final[['gt_px', 'gt_vx', 'gt_py', 'gt_vy']].values.astype(np.float32)
    Y_np = df_final[['gps_px', 'odo_vx', 'gps_py', 'odo_vy']].values.astype(np.float32)

    return torch.tensor(X_np), torch.tensor(Y_np)

# ==============================================================================
# 4. ULOŽENÍ
# ==============================================================================
def save_splits(X, Y):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    N = len(X)
    n_train = int(0.70 * N)
    n_val = int(0.20 * N)
    n_test = N - n_train - n_val
    
    print(f"  -> Rozdělení: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # 1. Nejdříve bereme Test (od začátku do n_test)
    X_test, Y_test   = X[:n_test], Y[:n_test]
    
    # 2. Potom Train (začíná tam, kde skončil test)
    X_train, Y_train = X[n_test:n_test+n_train], Y[n_test:n_test+n_train]
    
    # 3. Zbytek je Val (začíná tam, kde skončil train)
    X_val, Y_val     = X[n_test+n_train:], Y[n_test+n_train:]
    
    train_list = [{'ground_truth': X_train, 'filtered_gps': Y_train}] 
    val_list   = [{'ground_truth': X_val,   'filtered_gps': Y_val}]
    test_list  = [{'ground_truth': X_test,  'filtered_gps': Y_test}]
    
    torch.save(train_list, os.path.join(OUTPUT_DIR, 'train.pt'))
    torch.save(val_list, os.path.join(OUTPUT_DIR, 'val.pt'))
    torch.save(test_list, os.path.join(OUTPUT_DIR, 'test.pt'))
    
    print(f"✅ Uloženo do: {OUTPUT_DIR}")
    print(f"   Formát měření (filtered_gps): [GPS_X, ODO_VX, GPS_Y, ODO_VY]")

if __name__ == "__main__":
    try:
        X, Y = process_data()
        save_splits(X, Y)
    except Exception as e:
        print(f"\n❌ CHYBA: {e}")