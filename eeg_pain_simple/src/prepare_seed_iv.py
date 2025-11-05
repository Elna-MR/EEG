#!/usr/bin/env python3
"""
Prepare SEED-IV dataset for processing.
Handles MAT file conversion and pain trial extraction.
"""

import sys
from pathlib import Path
import numpy as np

def check_seed_iv_structure(data_dir: Path):
    """Check if SEED-IV dataset is properly structured."""
    print(f"Checking SEED-IV structure in: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    # SEED-IV typically has .mat files
    mat_files = list(data_dir.rglob("*.mat"))
    if mat_files:
        print(f"✅ Found {len(mat_files)} MAT files")
        return True
    
    # Check for subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"✅ Found {len(subdirs)} subject directories")
        for subdir in subdirs[:3]:
            files = list(subdir.rglob("*.mat"))
            if files:
                print(f"   {subdir.name}: {len(files)} MAT files")
        return True
    
    print("❌ No MAT files or subject directories found")
    return False

def convert_mat_to_bdf(mat_file: Path, output_dir: Path):
    """Convert MAT file to BDF format (requires scipy.io)."""
    try:
        import scipy.io
        from mne import create_info
        from mne.io import RawArray
        
        print(f"Converting {mat_file.name}...")
        
        # Load MAT file
        mat_data = scipy.io.loadmat(str(mat_file))
        
        # SEED-IV structure: typically has 'data' or 'eeg' key
        # This is dataset-specific - may need adjustment
        if 'data' in mat_data:
            eeg_data = mat_data['data']
        elif 'eeg' in mat_data:
            eeg_data = mat_data['eeg']
        else:
            print(f"⚠️  Unknown MAT structure, keys: {list(mat_data.keys())}")
            return None
        
        # Create MNE Raw object
        # Note: SEED-IV has 62 channels, but we'll select 20 for 10-20 system
        n_channels, n_samples = eeg_data.shape[:2]
        sfreq = 200  # SEED-IV typically uses 200Hz
        
        # Create channel names (generic - will be renamed later)
        ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = RawArray(eeg_data, info)
        
        # Save as BDF
        output_file = output_dir / f"{mat_file.stem}.bdf"
        raw.save(str(output_file), overwrite=True)
        
        print(f"✅ Converted to: {output_file}")
        return output_file
        
    except ImportError:
        print("❌ scipy not available. Install: pip install scipy")
        return None
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SEED-IV dataset")
    parser.add_argument("--data-dir", type=str, default="data/seed_iv",
                       help="Path to SEED-IV dataset directory")
    parser.add_argument("--convert", action="store_true",
                       help="Convert MAT files to BDF (requires scipy)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("="*70)
    print("SEED-IV Dataset Preparation")
    print("="*70)
    
    if not data_dir.exists():
        print(f"\n❌ Dataset directory not found: {data_dir}")
        print("\n📥 MANUAL DOWNLOAD REQUIRED:")
        print("1. Visit: http://bcmi.sjtu.edu.cn/~seed/seed-iv.html")
        print("2. Download license agreement: https://bcmi.sjtu.edu.cn/~seed/resource/license/SEED-IV%20license.pdf")
        print("3. Fill out and sign the agreement")
        print("4. Email signed PDF to: weilonglive@gmail.com")
        print("5. Wait for download link and password")
        print(f"6. Extract dataset to: {data_dir}/")
        return
    
    # Check structure
    if not check_seed_iv_structure(data_dir):
        print("\n⚠️  Dataset structure may be incorrect")
        print("Expected structure:")
        print(f"  {data_dir}/")
        print("    ├── subject_1/")
        print("    │   └── *.mat files")
        print("    └── ...")
    
    # Convert MAT to BDF if requested
    if args.convert:
        print("\n" + "="*70)
        print("Converting MAT files to BDF...")
        print("="*70)
        
        mat_files = list(data_dir.rglob("*.mat"))
        if not mat_files:
            print("❌ No MAT files found to convert")
            return
        
        output_dir = data_dir / "bdf_converted"
        output_dir.mkdir(exist_ok=True)
        
        converted = 0
        for mat_file in mat_files[:5]:  # Convert first 5 as test
            if convert_mat_to_bdf(mat_file, output_dir):
                converted += 1
        
        print(f"\n✅ Converted {converted}/{len(mat_files)} files")
        print(f"Converted files in: {output_dir}")
    else:
        print("\n" + "="*70)
        print("DATASET READY")
        print("="*70)
        print(f"\nDataset location: {data_dir}")
        print("\nNote: SEED-IV uses MAT format. To process:")
        print("1. Convert MAT to BDF (if needed):")
        print("   python src/prepare_seed_iv.py --data-dir data/seed_iv/ --convert")
        print("2. Or use MATLAB/Octave to convert to BDF/EDF format")
        print("3. Then run preprocessing:")
        print("   python src/preprocess_multidataset.py --dataset seed_pain --data-dir data/seed_iv/")

if __name__ == "__main__":
    main()

