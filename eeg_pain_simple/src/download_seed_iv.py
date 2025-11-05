#!/usr/bin/env python3
"""
Download script for SEED-IV dataset (includes SEED-Pain subset).
Attempts to download from official sources.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import urllib.request

SEED_IV_URL = "http://bcmi.sjtu.edu.cn/~seed/seed-iv.html"
BASE_URL = "http://bcmi.sjtu.edu.cn/~seed/"

def check_url(url):
    """Check if URL is accessible."""
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except:
        return False

def download_file(url, output_path, chunk_size=8192):
    """Download file with progress."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def main():
    print("="*70)
    print("SEED-IV Dataset Downloader")
    print("="*70)
    print(f"\nOfficial page: {SEED_IV_URL}")
    print("\nChecking download options...")
    
    # Common download patterns for SEED-IV
    download_urls = [
        "http://bcmi.sjtu.edu.cn/~seed/downloads/seed-iv.zip",
        "http://bcmi.sjtu.edu.cn/~seed/seed-iv.zip",
        "http://bcmi.sjtu.edu.cn/~seed/data/seed-iv.zip",
    ]
    
    data_dir = Path("data/seed_iv")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nAttempting to find download link...")
    found = False
    
    for url in download_urls:
        print(f"\nTrying: {url}")
        if check_url(url):
            print(f"✅ Found accessible URL: {url}")
            output_file = data_dir / "seed-iv.zip"
            print(f"Downloading to: {output_file}")
            
            if download_file(url, output_file):
                print(f"\n✅ Downloaded to: {output_file}")
                print("Extracting...")
                try:
                    with zipfile.ZipFile(output_file, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print(f"✅ Extracted to: {data_dir}")
                    found = True
                    break
                except Exception as e:
                    print(f"❌ Extraction failed: {e}")
        else:
            print(f"❌ URL not accessible")
    
    if not found:
        print("\n" + "="*70)
        print("AUTOMATIC DOWNLOAD NOT AVAILABLE")
        print("="*70)
        print("\nManual download required:")
        print(f"1. Visit: {SEED_IV_URL}")
        print("2. Register/login if required")
        print("3. Download SEED-IV dataset")
        print(f"4. Extract to: {data_dir}/")
        print("\nAfter downloading, the dataset structure should be:")
        print(f"  {data_dir}/")
        print("    ├── subject_1/")
        print("    │   └── *.mat files")
        print("    ├── subject_2/")
        print("    └── ...")
        print("\nThen run:")
        print("  python src/preprocess_multidataset.py --dataset seed_pain --data-dir data/seed_iv/")
        return False
    
    return True

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        os.system(f"{sys.executable} -m pip install requests")
        import requests
    
    main()

