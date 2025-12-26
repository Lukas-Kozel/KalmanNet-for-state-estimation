#!/usr/bin/env python3
import sys
import os
import subprocess
import tarfile
import shutil

# Konfigurace
BASE_URL = 'https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu'
DATA_ROOT = 'data' # Hlavní složka, kam se to uloží

# Seznam všech datumu v datasetu
DATES = [
    '2012-01-08', '2012-01-15', '2012-01-22', '2012-02-02', '2012-02-04',
    '2012-02-05', '2012-02-12', '2012-02-18', '2012-02-19', '2012-03-17',
    '2012-03-25', '2012-03-31', '2012-04-29', '2012-05-11', '2012-05-26',
    '2012-06-15', '2012-08-04', '2012-08-20', '2012-09-28', '2012-10-28',
    '2012-11-04', '2012-11-16', '2012-11-17', '2012-12-01', '2013-01-10',
    '2013-02-23', '2013-04-05'
]

def ensure_dir(path):
    """Vytvoří složku, pokud neexistuje."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def download_file(url, dest_folder, file_name):
    """Stáhne soubor pomocí wget (podporuje navázání přerušeného stahování)."""
    ensure_dir(dest_folder)
    dest_path = os.path.join(dest_folder, file_name)
    
    print(f"Downloading {file_name}...")
    # Používáme wget s -c (continue), -q (quiet/méně spamu), --show-progress
    cmd = ['wget', '-c', '--show-progress', url, '-O', dest_path]
    
    try:
        subprocess.check_call(cmd)
        return dest_path
    except subprocess.CalledProcessError:
        print(f"❌ Error downloading {url}")
        return None

def extract_tar_gz(archive_path, extract_to):
    """Rozbalí tar.gz archiv do cílové složky."""
    print(f"Extracting {os.path.basename(archive_path)} into {extract_to}...")
    ensure_dir(extract_to)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print("✅ Extraction complete.")
    except Exception as e:
        print(f"❌ Error extracting: {e}")

def main():
    print(f"=== NCLT Dataset Downloader & Organizer ===")
    print(f"Target directory: {os.path.abspath(DATA_ROOT)}")
    
    # 1. Příprava hlavních složek
    gt_dir = os.path.join(DATA_ROOT, 'ground_truth')
    sensor_root_dir = os.path.join(DATA_ROOT, 'sensor')
    
    ensure_dir(gt_dir)
    ensure_dir(sensor_root_dir)

    for date in DATES:
        print(f"\nProcessing session: {date} ==========================")
        
        # --- A. GROUND TRUTH ---
        gt_filename = f"groundtruth_{date}.csv"
        gt_path = os.path.join(gt_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            gt_url = f"{BASE_URL}/ground_truth/{gt_filename}"
            download_file(gt_url, gt_dir, gt_filename)
        else:
            print(f"Skipping GT (already exists): {gt_filename}")

        # --- B. SENSOR DATA (GPS, IMU, WHEELS) ---
        # Cílová složka pro tuto session: data/sensor/2012-01-22/
        session_sensor_dir = os.path.join(sensor_root_dir, date)
        
        # Kontrola, zda už jsou data rozbalena (zkusíme najít např. gps.csv)
        if os.path.exists(os.path.join(session_sensor_dir, 'gps.csv')):
            print(f"Skipping Sensors (already extracted): {date}")
            continue
            
        # 1. Stáhnout archiv
        sen_filename = f"{date}_sen.tar.gz"
        sen_url = f"{BASE_URL}/sensor_data/{sen_filename}"
        
        # Stahujeme dočasně do rootu sensor složky
        archive_path = download_file(sen_url, sensor_root_dir, sen_filename)
        
        if archive_path and os.path.exists(archive_path):
            # 2. Rozbalit
            # NCLT archivy jsou "flat" (nemají kořenovou složku), musíme ji vytvořit
            extract_tar_gz(archive_path, session_sensor_dir)
            
            # 3. Smazat archiv (volitelné, šetří místo)
            print(f"Cleaning up archive {sen_filename}...")
            os.remove(archive_path)

    print("\n\n✅ Vše hotovo! Dataset je připraven pro preprocess.py.")

if __name__ == '__main__':
    # Kontrola, zda máme wget
    if shutil.which('wget') is None:
        print("Error: 'wget' is not installed. Please install it (sudo apt install wget).")
        sys.exit(1)
        
    main()