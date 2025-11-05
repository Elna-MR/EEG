#!/usr/bin/env python3
"""
Multi-dataset preprocessing for EEG pain detection:
- Supports ds005284 (current), BioVid, PainMonit, SEED-Pain datasets
- Same approach: 20-channel 10-20 system, Riemannian features, DANN training
- Adapts to different file formats and event structures
"""

import os
import json
import glob
import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------
# Dataset Configurations
# -----------------
DATASET_CONFIGS = {
    "ds005284": {
        "name": "ds005284",
        "file_pattern": "sub-*/eeg/*_task-26ByBiosemi_eeg.bdf",
        "file_format": "bdf",
        "event_code": 54,  # Condition 54 = pain
        "baseline_tmin": -1.5,
        "baseline_tmax": 0.0,
        "pain_tmin": 0.0,
        "pain_tmax": 2.0,
        "needs_channel_rename": True,
    },
    "biovid": {
        "name": "biovid",
        "file_pattern": "**/*.bdf",  # BioVid typically uses BDF/EDF
        "file_format": "bdf",
        "event_code": None,  # Will need to be determined from events file
        "baseline_tmin": -1.0,
        "baseline_tmax": 0.0,
        "pain_tmin": 0.0,
        "pain_tmax": 2.0,
        "needs_channel_rename": False,
    },
    "painmonit": {
        "name": "painmonit",
        "file_pattern": "**/*.bdf",
        "file_format": "bdf",
        "event_code": None,
        "baseline_tmin": -1.0,
        "baseline_tmax": 0.0,
        "pain_tmin": 0.0,
        "pain_tmax": 2.0,
        "needs_channel_rename": False,
    },
    "seed_pain": {
        "name": "seed_pain",
        "file_pattern": "**/*.mat",  # SEED typically uses MATLAB format
        "file_format": "mat",
        "event_code": None,
        "baseline_tmin": -1.0,
        "baseline_tmax": 0.0,
        "pain_tmin": 0.0,
        "pain_tmax": 2.0,
        "needs_channel_rename": False,
    }
}

# -----------------
# Global Config
# -----------------
LOWCUT, HIGHCUT = 1.0, 45.0
SFREQ_TARGET = 256.0
USE_10_20_SYSTEM = True
NUM_10_20_CHANNELS = 20
BALANCE_PER_SUBJECT = False
RANDOM_STATE = 42

OUT_DIR = Path("packed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rename_channels_from_biosemi(raw, sub_id):
    """Rename channels from Biosemi format to 10-20 system."""
    channel_mapping = {
        'A1': 'FP1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7',
        'A8': 'FT7', 'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5',
        'A15': 'T7', 'A16': 'TP7', 'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3',
        'A22': 'P5', 'A23': 'P7', 'A24': 'P9', 'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'LZ',
        'A29': 'OZ', 'A30': 'POZ', 'A31': 'PZ', 'A32': 'CPZ',
        'B1': 'FPZ', 'B2': 'FP2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFZ', 'B6': 'FZ', 'B7': 'F2',
        'B8': 'F4', 'B9': 'F6', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2',
        'B15': 'FCZ', 'B16': 'CZ', 'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8',
        'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2', 'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8',
        'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'
    }
    
    if any(ch.startswith('A') or ch.startswith('B') for ch in raw.ch_names):
        rename_dict = {}
        for ch in raw.ch_names:
            if ch in channel_mapping:
                rename_dict[ch] = channel_mapping[ch]
        if rename_dict:
            raw.rename_channels(rename_dict)
            print(f"[INFO] {sub_id}: Renamed {len(rename_dict)} channels from Biosemi format")
    
    return raw


def select_10_20_channels(raw, num_channels=20):
    """Select 10-20 system channels."""
    channels_19 = [
        'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ',
        'C3', 'C4', 'CZ',
        'P3', 'P4', 'PZ',
        'T7', 'T8',
        'P7', 'P8',
        'O1', 'O2'
    ]
    channels_20 = channels_19 + ['OZ']
    target_channels = channels_20 if num_channels == 20 else channels_19
    
    available_chs = [ch.upper() for ch in raw.ch_names]
    selected_chs = []
    missing_chs = []
    
    for target in target_channels:
        target_upper = target.upper()
        matches = [ch for ch in raw.ch_names if ch.upper() == target_upper]
        if matches:
            selected_chs.append(matches[0])
        else:
            missing_chs.append(target)
    
    if missing_chs:
        print(f"[WARN] Missing channels: {missing_chs}")
    
    if len(selected_chs) < num_channels - 2:  # Allow 1-2 missing
        raise ValueError(f"Too few 10-20 channels found: {len(selected_chs)}/{num_channels}")
    
    raw_selected = raw.pick_channels(selected_chs, ordered=False)
    print(f"[INFO] Selected {len(selected_chs)} 10-20 system channels: {selected_chs}")
    return raw_selected


def load_dataset_file(file_path: Path, file_format: str) -> mne.io.BaseRaw:
    """Load EEG file in various formats."""
    if file_format == "bdf":
        return mne.io.read_raw_bdf(file_path, preload=True, verbose="ERROR")
    elif file_format == "edf":
        return mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    elif file_format == "mat":
        # SEED dataset uses MATLAB format - need special handling
        raise NotImplementedError("MAT format loading not yet implemented. Please convert to BDF/EDF first.")
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def find_pain_events(events: np.ndarray, config: Dict) -> np.ndarray:
    """Find pain events based on dataset configuration."""
    if len(events) == 0:
        return np.array([])
    
    if config["event_code"] is not None:
        # Direct event code (like ds005284)
        pain_events = events[events[:, 2] == config["event_code"]]
    else:
        # Need to infer from events structure
        # Common patterns: highest event code, or events marked in annotation
        # For now, use all events > threshold (will need dataset-specific tuning)
        unique_codes = np.unique(events[:, 2])
        if len(unique_codes) >= 2:
            # Assume highest code is pain (common pattern)
            pain_code = unique_codes[-1]
            pain_events = events[events[:, 2] == pain_code]
        else:
            # Single event type - use all
            pain_events = events
    
    return pain_events


def process_subject(file_path: Path, config: Dict, sub_id: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Process a single subject's data."""
    print(f"\n[INFO] Processing {sub_id}: {file_path}")
    
    # Load file
    raw = load_dataset_file(file_path, config["file_format"])
    
    # Resample if needed
    if abs(raw.info["sfreq"] - SFREQ_TARGET) > 1e-6:
        raw.resample(SFREQ_TARGET, verbose=False)
    
    # Find events
    stim_channel = None
    if 'Status' in raw.ch_names:
        stim_channel = 'Status'
    elif 'STATUS' in [ch.upper() for ch in raw.ch_names]:
        stim_channel = [ch for ch in raw.ch_names if ch.upper() == 'STATUS'][0]
    
    events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose=False)
    
    # Find pain events
    pain_events = find_pain_events(events, config)
    
    if len(pain_events) == 0:
        print(f"[WARN] {sub_id}: No pain events found; skipping")
        return None, None, None
    
    # Exclude non-EEG channels
    eeg_chs = [ch for ch in raw.ch_names if ch.upper() not in ['STATUS', 'TRIGGER']]
    if len(eeg_chs) < len(raw.ch_names):
        raw.pick_channels(eeg_chs)
    
    # Rename channels if needed
    if config["needs_channel_rename"]:
        raw = rename_channels_from_biosemi(raw, sub_id)
    
    # Select 10-20 system channels
    if USE_10_20_SYSTEM:
        raw = select_10_20_channels(raw, num_channels=NUM_10_20_CHANNELS)
    
    # Filter
    raw.filter(LOWCUT, HIGHCUT, verbose=False)
    
    # Create epochs
    epoch_tmin = config["baseline_tmin"]
    epoch_tmax = config["pain_tmax"]
    
    # Use the event code from pain_events (not hardcoded)
    if len(pain_events) > 0:
        pain_code = int(pain_events[0, 2])
        epochs = mne.Epochs(raw, pain_events, event_id={"pain": pain_code}, 
                           tmin=epoch_tmin, tmax=epoch_tmax, 
                           baseline=None, preload=True, verbose=False)
    else:
        print(f"[WARN] {sub_id}: No pain events found after filtering")
        return None, None, None
    
    if len(epochs) == 0:
        print(f"[WARN] {sub_id}: No epochs created; skipping")
        return None, None, None
    
    X_full = epochs.get_data()  # (trials, channels, samples)
    n_trials = X_full.shape[0]
    sfreq = epochs.info['sfreq']
    
    # Extract baseline and pain segments
    base_start = 0
    base_end = int((config["baseline_tmax"] - config["baseline_tmin"]) * sfreq)
    pain_start = int((config["pain_tmin"] - config["baseline_tmin"]) * sfreq)
    pain_end = int((config["pain_tmax"] - config["baseline_tmin"]) * sfreq)
    
    X_base = X_full[:, :, base_start:base_end]
    X_pain = X_full[:, :, pain_start:pain_end]
    
    # Pad baseline to match pain duration
    if X_base.shape[2] < X_pain.shape[2]:
        pad_width = X_pain.shape[2] - X_base.shape[2]
        X_base = np.pad(X_base, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
    
    # Combine
    X = np.concatenate([X_base, X_pain], axis=0)
    y = np.concatenate([
        np.zeros(n_trials, dtype=int),
        np.ones(n_trials, dtype=int)
    ])
    
    print(f"[DEBUG] {sub_id}: {n_trials} epochs → {n_trials} baseline + {n_trials} pain = {len(X)} total")
    
    return X, y, sub_id


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess EEG pain dataset")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), 
                       default="ds005284", help="Dataset to process")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename (default: {dataset}_pain.npz)")
    
    args = parser.parse_args()
    
    config = DATASET_CONFIGS[args.dataset]
    data_root = Path(args.data_dir)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Find files
    file_pattern = data_root / config["file_pattern"]
    files = sorted(glob.glob(str(file_pattern)))
    
    if not files:
        raise FileNotFoundError(f"No files found matching: {file_pattern}")
    
    print(f"\n{'='*70}")
    print(f"[INFO] Processing {config['name']} dataset")
    print(f"[INFO] Found {len(files)} files")
    print(f"{'='*70}")
    
    all_X = []
    all_y = []
    all_sub = []
    
    for file_path in files:
        file_path = Path(file_path)
        # Extract subject ID from path
        sub_id = file_path.parent.name if file_path.parent.name.startswith("sub-") else f"sub-{file_path.stem}"
        
        try:
            X, y, sub_id = process_subject(file_path, config, sub_id)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
                all_sub.append(np.array([sub_id] * len(y), dtype="U16"))
        except Exception as e:
            print(f"[ERROR] {sub_id}: {e}")
            continue
    
    if not all_X:
        raise RuntimeError("No epochs collected across subjects.")
    
    # Combine all subjects
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subs_all = np.concatenate(all_sub, axis=0)
    
    # Balance classes
    pain_idx = np.where(y_all == 1)[0]
    base_idx = np.where(y_all == 0)[0]
    
    min_size = min(len(pain_idx), len(base_idx))
    rng = np.random.default_rng(RANDOM_STATE)
    pain_idx_balanced = rng.choice(pain_idx, size=min_size, replace=False)
    base_idx_balanced = rng.choice(base_idx, size=min_size, replace=False)
    keep_idx = np.concatenate([pain_idx_balanced, base_idx_balanced])
    
    X_all = X_all[keep_idx]
    y_all = y_all[keep_idx]
    subs_all = subs_all[keep_idx]
    
    print(f"\n[CHECK] Final label counts: pain={int((y_all==1).sum())}, baseline={int((y_all==0).sum())}, total={len(y_all)}")
    print(f"[CHECK] X shape: {X_all.shape}  (trials, channels, samples)")
    print(f"[CHECK] sfreq: {SFREQ_TARGET}")
    
    # Save
    output_name = args.output or f"{config['name']}_pain.npz"
    out_npz = OUT_DIR / output_name
    out_sfreq_json = OUT_DIR / f"sfreq_{config['name']}.json"
    
    np.savez_compressed(out_npz, X=X_all, y=y_all, subject=subs_all, sfreq=SFREQ_TARGET)
    with open(out_sfreq_json, "w") as f:
        json.dump({"sfreq": float(SFREQ_TARGET), "num_channels": NUM_10_20_CHANNELS, "dataset": config['name']}, f)
    
    print(f"[INFO] Saved epochs to {out_npz}")
    print(f"[INFO] Saved sampling freq to {out_sfreq_json}")


if __name__ == "__main__":
    main()

