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

# Cesty
BASE_DIR = os.getcwd()
GT_FILE = os.path.join(BASE_DIR, 'ground_truth', f'groundtruth_{SESSION_DATE}.csv')
SENSOR_DIR = os.path.join(BASE_DIR, 'data', 'sensor', SESSION_DATE)
OUTPUT_DIR = os.path.join(BASE_DIR, f'preprocessed_NCLT_trajectory-{SESSION_DATE}-angle-update')

# ==============================================================================
# 2. POMOCNÉ FUNKCE (Geometrie a Načítání)
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
    
    # Načteme CSV (low_memory=False potlačí DtypeWarning, my si to stejně převedeme sami)
    df = pd.read_csv(path, header=None, usecols=cols, names=names, 
                     on_bad_lines='skip', low_memory=False)
    
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
# 3. NOVÁ FUNKCE PRO DIAGNOSTIKU (OPRAVENO)
# ==============================================================================
def plot_diagnostics(df, output_dir):
    """Vykreslí Yaw a Rychlosti pro kontrolu kvality dat."""
    print("  -> Generuji diagnostické grafy...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Používáme .values pro převod na numpy array, aby si Matplotlib nestěžoval
    t_values = df.index.values

    # 1. Graf: YAW (Heading)
    axes[0].plot(t_values, df['yaw'].values, label='Yaw (Odometrie)', color='purple', linewidth=1)
    axes[0].set_ylabel('Yaw [rad]')
    axes[0].set_title('Kontrola úhlu natočení (Yaw)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # 2. Graf: Rychlost v ose X (Porovnání GT a ODO)
    axes[1].plot(t_values, df['gt_vx'].values, label='Ground Truth Vx', color='green', alpha=0.7)
    axes[1].plot(t_values, df['odo_vx'].values, label='Odometrie Vx', color='orange', linestyle='--', alpha=0.8)
    axes[1].set_ylabel('Vx [m/s]')
    axes[1].set_title('Kontrola rychlosti v ose X')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    # 3. Graf: Trajektorie (XY) - jen pro orientaci
    axes[2].plot(t_values, df['gt_px'].values, label='GT Poloha X', color='blue')
    axes[2].plot(t_values, df['gps_px'].values, label='GPS Poloha X', color='red', alpha=0.5, linestyle=':')
    axes[2].set_ylabel('Poloha X [m]')
    axes[2].set_xlabel('Čas [s]')
    axes[2].set_title('Kontrola Polohy X (GT vs GPS)')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'diagnostics_plot.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ Graf uložen: {save_path}")
    # plt.show() # Odkomentuj, pokud chceš zobrazit okno při běhu

# ==============================================================================
# 4. HLAVNÍ PROCESING
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
    
    def reindex_and_interp(df, target_index, angular_cols=None):
        if angular_cols is None:
            angular_cols = []

        # 1. Sjednocení indexů a odstranění duplicit
        combined_index = df.index.union(target_index).sort_values()
        combined_index = combined_index.drop_duplicates()
        
        # Přeindexování (zatím s NaN tam, kde chybí data)
        df_reindexed = df.reindex(combined_index)

        # 2. Speciální zpracování pro úhly (Unwrap -> Interpolate -> Wrap)
        for col in angular_cols:
            if col in df_reindexed.columns:
                # Získáme validní data (nenulová) pro unwrap
                valid_mask = df_reindexed[col].notna()
                values = df_reindexed.loc[valid_mask, col].values
                
                # "Rozbalíme" úhly (odstraníme skoky přes 2pi)
                unwrapped_values = np.unwrap(values)
                
                # Vrátíme rozbalené hodnoty do DataFrame pro interpolaci
                df_reindexed.loc[valid_mask, col] = unwrapped_values

        # 3. Lineární interpolace všech sloupců (nyní funguje i pro 'unwrapped' úhly)
        df_interp = df_reindexed.interpolate(method='index').dropna()

        # 4. Zabalení úhlů zpět do intervalu [-pi, pi]
        for col in angular_cols:
            if col in df_interp.columns:
                df_interp[col] = (df_interp[col] + np.pi) % (2 * np.pi) - np.pi

        # 5. Vrátíme jen cílové časy
        return df_interp.reindex(target_index).dropna()

    # 1. Ground Truth
    df_gt = reindex_and_interp(df_gt_raw, t_grid)
    
    # 2. GPS
    df_gps = reindex_and_interp(df_gps_raw, t_grid)
    
    # 3. Wheels
    df_whl = reindex_and_interp(df_whl_raw, t_grid)
    
    # 4. Euler
    df_eul = reindex_and_interp(df_eul_raw, t_grid, angular_cols=['yaw'])

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

    # --- D. DIAGNOSTIKA (Vykreslení grafu) ---
    # Voláme ještě před finálním ořezáním sloupců, abychom viděli 'yaw'
    plot_diagnostics(df, OUTPUT_DIR)

    # --- E. ČIŠTĚNÍ ---
    cols_of_interest = ['gt_px', 'gt_vx', 'gt_py', 'gt_vy', 
                        'gps_px', 'odo_vx', 'gps_py', 'odo_vy']
    
    df_final = df[cols_of_interest].dropna()
    
    print(f"  -> Finální počet vzorků: {len(df_final)}")

    # Příprava Tensorů
    X_np = df_final[['gt_px', 'gt_vx', 'gt_py', 'gt_vy']].values.astype(np.float32)
    Y_np = df_final[['gps_px', 'odo_vx', 'gps_py', 'odo_vy']].values.astype(np.float32)

    return torch.tensor(X_np), torch.tensor(Y_np)

# ==============================================================================
# 5. ULOŽENÍ
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