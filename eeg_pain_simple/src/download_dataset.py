#!/usr/bin/env python3
"""
Download script for publicly available EEG pain datasets.
Downloads and prepares data for preprocessing.
"""

import os
import sys
import subprocess
from pathlib import Path

DATASET_INFO = {
    "biovid": {
        "name": "BioVid Heat Pain Database",
        "subjects": 87,
        "channels": 30,
        "description": "Thermal stimulation at multiple intensities",
        "download_url": "https://www.aau.at/en/research/research-areas/affective-computing/",
        "notes": "Contact dataset authors or check BCI competition repositories",
    },
    "painmonit": {
        "name": "PainMonit Database",
        "subjects": 56,
        "channels": "Variable",
        "description": "Electrical stimulation, multimodal recordings",
        "download_url": "https://www.painmonit.com/",
        "notes": "May require registration or contact authors",
    },
    "seed_pain": {
        "name": "SEED-Pain (subset of SEED-IV)",
        "subjects": 15,
        "channels": 62,
        "description": "Electrical stimulation with emotional valence",
        "download_url": "http://bcmi.sjtu.edu.cn/~seed/seed-iv.html",
        "notes": "Part of SEED-IV dataset - download full dataset and extract pain subset",
    }
}

def print_dataset_info():
    """Print information about available datasets."""
    print("="*70)
    print("AVAILABLE EEG PAIN DATASETS")
    print("="*70)
    for key, info in DATASET_INFO.items():
        print(f"\n{key.upper()}:")
        print(f"  Name: {info['name']}")
        print(f"  Subjects: {info['subjects']}")
        print(f"  Channels: {info['channels']}")
        print(f"  Description: {info['description']}")
        print(f"  Download: {info['download_url']}")
        print(f"  Notes: {info['notes']}")
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("1. Download dataset from the provided URL")
    print("2. Extract to: eeg_pain_simple/data/{dataset_name}/")
    print("3. Run: python src/preprocess_multidataset.py --dataset {dataset_name} --data-dir data/{dataset_name}/")
    print("="*70)

def check_dataset_structure(dataset_name: str, data_dir: Path):
    """Check if downloaded dataset has correct structure."""
    print(f"\nChecking dataset structure for {dataset_name}...")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Check for common file patterns
    bdf_files = list(data_dir.rglob("*.bdf"))
    edf_files = list(data_dir.rglob("*.edf"))
    mat_files = list(data_dir.rglob("*.mat"))
    
    if bdf_files:
        print(f"✅ Found {len(bdf_files)} BDF files")
        return True
    elif edf_files:
        print(f"✅ Found {len(edf_files)} EDF files")
        return True
    elif mat_files:
        print(f"✅ Found {len(mat_files)} MAT files (need conversion)")
        return True
    else:
        print(f"❌ No EEG files found (.bdf, .edf, .mat)")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "info":
        print_dataset_info()
    else:
        print_dataset_info()
        print("\nTo check if you have a dataset downloaded:")
        print("  python src/download_dataset.py check <dataset_name> <data_dir>")

