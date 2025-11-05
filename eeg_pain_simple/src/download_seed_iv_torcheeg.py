#!/usr/bin/env python3
"""
Download SEED-IV dataset using TorchEEG library.
This library can automatically download SEED-IV with proper authentication.
"""

import sys
from pathlib import Path

def main():
    print("="*70)
    print("SEED-IV Dataset Download via TorchEEG")
    print("="*70)
    
    try:
        from torcheeg.datasets import SEEDIVDataset
        print("✅ TorchEEG library found")
    except ImportError:
        print("Installing TorchEEG...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torcheeg", "-q"])
        from torcheeg.datasets import SEEDIVDataset
        print("✅ TorchEEG installed")
    
    data_dir = Path("data/seed_iv")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading SEED-IV dataset to: {data_dir}")
    print("This may take a while and may require authentication...")
    print("")
    
    try:
        # Initialize dataset - this should trigger download
        dataset = SEEDIVDataset(
            root_path=str(data_dir),
            # online=True,  # Set to True to download
        )
        print(f"✅ Dataset initialized")
        print(f"Dataset location: {data_dir}")
        print(f"\nTo use the dataset:")
        print(f"  python src/preprocess_multidataset.py --dataset seed_pain --data-dir {data_dir}/")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: SEED-IV may require:")
        print("1. Manual registration at http://bcmi.sjtu.edu.cn/~seed/seed-iv.html")
        print("2. License agreement completion")
        print("3. Email to weilonglive@gmail.com with signed agreement")
        return False

if __name__ == "__main__":
    main()

