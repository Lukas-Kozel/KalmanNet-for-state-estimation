#!/usr/bin/env python3
import sys
import os
import subprocess
import tarfile
import shutil

# Konfigurace
BASE_URL = 'https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu'
DATA_ROOT = 'data' # Main folder where everything will be saved

# List of all dates in the dataset
DATES = [
    '2012-01-08', '2012-01-15', '2012-01-22', '2012-02-02', '2012-02-04',
    '2012-02-05', '2012-02-12', '2012-02-18', '2012-02-19', '2012-03-17',
    '2012-03-25', '2012-03-31', '2012-04-29', '2012-05-11', '2012-05-26',
    '2012-06-15', '2012-08-04', '2012-08-20', '2012-09-28', '2012-10-28',
    '2012-11-04', '2012-11-16', '2012-11-17', '2012-12-01', '2013-01-10',
    '2013-02-23', '2013-04-05'
]

def ensure_dir(path):
    """Creates a folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def download_file(url, dest_folder, file_name):
    """Downloads a file using wget (supports resuming interrupted downloads)."""
    ensure_dir(dest_folder)
    dest_path = os.path.join(dest_folder, file_name)
    
    print(f"Downloading {file_name}...")
    # We use wget with -c (continue), -q (quiet/less spam), --show-progress
    cmd = ['wget', '-c', '--show-progress', url, '-O', dest_path]
    
    try:
        subprocess.check_call(cmd)
        return dest_path
    except subprocess.CalledProcessError:
        print(f"❌ Error downloading {url}")
        return None

def extract_tar_gz(archive_path, extract_to):
    """Extracts a tar.gz archive into the target folder."""
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
    
    # 1. Prepare the main folders
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
        # Target folder for this session: data/sensor/2012-01-22/
        session_sensor_dir = os.path.join(sensor_root_dir, date)
        
        # Check whether the data is already extracted (try to find e.g. gps.csv)
        if os.path.exists(os.path.join(session_sensor_dir, 'gps.csv')):
            print(f"Skipping Sensors (already extracted): {date}")
            continue
            
        # 1. Download the archive
        sen_filename = f"{date}_sen.tar.gz"
        sen_url = f"{BASE_URL}/sensor_data/{sen_filename}"
        
        # Download temporarily into the root of the sensor folder
        archive_path = download_file(sen_url, sensor_root_dir, sen_filename)
        
        if archive_path and os.path.exists(archive_path):
            # 2. Rozbalit
            # NCLT archives are "flat" (they have no root folder), so we must create it
            extract_tar_gz(archive_path, session_sensor_dir)
            
            # 3. Delete the archive (optional, saves space)
            print(f"Cleaning up archive {sen_filename}...")
            os.remove(archive_path)

    print("\n\n✅ All done! The dataset is ready for preprocess.py.")

if __name__ == '__main__':
    # Check whether we have wget
    if shutil.which('wget') is None:
        print("Error: 'wget' is not installed. Please install it (sudo apt install wget).")
        sys.exit(1)
        
    main()