#!/usr/bin/env python3
"""
Preprocessing for osf-data:
- Loads all subjects' BrainVision .dat/.evt EEG files
- Filters 1–45 Hz
- Uses event codes 21/22 for pain events (based on .evt file structure)
- Event onset = pain start point (t=0.0s)
- Baseline: -1.5 to 0.0s (pre-stim, before pain)
- Pain: 0.0 to 2.0s (post-stim, after pain onset)
- Balances classes (undersample majority)
- Saves packed/osf-data_pain.npz with X (trials, channels, samples), y (0=baseline,1=pain), sfreq

Same preprocessing pipeline as ds005284 for comparison.
"""

import os
import json
import glob
import numpy as np
import mne
from pathlib import Path
import re

# -----------------
# Config (same as ds005284)
# -----------------
# DATA_ROOT: Relative to script location (scripts/ folder)
# Script is in osf-data/scripts/, so data/ is in osf-data/data/
_script_dir = Path(__file__).parent
DATA_ROOT = _script_dir.parent / "data"  # osf-data/data directory

LOWCUT, HIGHCUT = 1.0, 45.0       # Hz
SFREQ_TARGET = 256.0              # Hz (downsample to this, if needed)

# Event codes from .evt file (21/22 appear to be pain markers)
PAIN_CODES = [21, 22]  # Use both codes as pain events

# Epoch timing relative to event onset (where pain starts)
BASE_TMIN, BASE_TMAX = -1.5, 0.0  # Baseline: 1.5s BEFORE event onset (pre-stim)
PAIN_TMIN, PAIN_TMAX = 0.0, 2.0   # Pain: 0.0 to 2.0s AFTER event onset (post-stim)

BALANCE_PER_SUBJECT = False       # set True to balance inside each subject; False = balance globally
RANDOM_STATE = 42
MAX_TRIALS = 832                  # Maximum number of trials to keep (416 pain + 416 baseline, matching ds005284)

# 10-20 system channel selection
USE_10_20_SYSTEM = True           # Set True to select only 10-20 system channels
NUM_CHANNELS = 20                  # Number of 10-20 system channels (20 channels: FP1, FP2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, T7, T8, P7, P8, O1, O2, OZ)

# OUT_DIR: Relative to script location (scripts/ folder)
# Script is in osf-data/scripts/, so packed/ is in osf-data/packed/
_script_dir = Path(__file__).parent
OUT_DIR = _script_dir.parent / "packed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------
# Helpers
# -----------------
rng = np.random.default_rng(RANDOM_STATE)

def _debug_counts(tag, events):
    if events is None or len(events) == 0:
        print(f"[DEBUG] {tag}: no events")
        return
    vals, cnts = np.unique(events[:, 2], return_counts=True)
    print(f"[DEBUG] {tag}: " + ", ".join([f"{int(v)}={int(c)}" for v,c in zip(vals,cnts)]))

def make_epochs(raw, events, event_id, tmin, tmax):
    return mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=None,
        reject=None, flat=None,
        preload=True, verbose=False
    )

def balance_indices(y):
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return np.arange(len(y))  # nothing to balance
    n = min(len(idx_pos), len(idx_neg))
    keep = np.r_[rng.choice(idx_pos, n, replace=False),
                 rng.choice(idx_neg, n, replace=False)]
    keep.sort()
    return keep

def select_10_20_channels(raw):
    """
    Select 20 channels from 10-20 system EEG channels.
    Standard 10-20 system: FP1, FP2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, T7, T8, P7, P8, O1, O2, OZ
    """
    # 20 channels from 10-20 system
    target_channels = [
        'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ',
        'C3', 'C4', 'CZ',
        'P3', 'P4', 'PZ',
        'T7', 'T8',
        'P7', 'P8',
        'O1', 'O2', 'OZ'
    ]
    
    # Get available channel names (case-insensitive matching)
    available_chs = [ch.upper() for ch in raw.ch_names]
    
    # Find matching channels
    selected_chs = []
    missing_chs = []
    
    for target in target_channels:
        target_upper_check = target.upper()
        if target_upper_check in available_chs:
            idx = available_chs.index(target_upper_check)
            selected_chs.append(raw.ch_names[idx])
        else:
            missing_chs.append(target)
    
    if missing_chs:
        print(f"[WARN] Missing 10-20 channels: {missing_chs}")
        print(f"[INFO] Available channels: {raw.ch_names[:10]}... (showing first 10)")
    
    if len(selected_chs) < NUM_CHANNELS:
        print(f"[WARN] Only found {len(selected_chs)}/{NUM_CHANNELS} 10-20 channels")
    
    # Pick channels and reorder to match target order
    try:
        raw_selected = raw.pick_channels(selected_chs)
        ch_order = []
        for target in target_channels:
            target_upper_check = target.upper()
            for ch in raw_selected.ch_names:
                if ch.upper() == target_upper_check:
                    ch_order.append(ch)
                    break
        if ch_order:
            raw_selected = raw_selected.reorder_channels(ch_order)
        print(f"[INFO] Selected {len(selected_chs)} 10-20 system channels: {raw_selected.ch_names}")
        if len(selected_chs) != NUM_CHANNELS:
            print(f"[WARN] Expected {NUM_CHANNELS} channels but got {len(selected_chs)}")
        return raw_selected
    except ValueError as e:
        print(f"[ERROR] Failed to select channels: {e}")
        print(f"[INFO] Available channels: {raw.ch_names}")
        raise

def parse_evt_file(evt_path):
    """
    Parse BrainVision .evt file.
    Format: Tmu    Code    TriNo    Comnt
    Returns: events array (n_events, 3) with [sample, 0, code]
    """
    events = []
    sfreq = None
    
    with open(evt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header if present
    start_idx = 0
    for i, line in enumerate(lines):
        if 'Tmu' in line or 'Code' in line:
            start_idx = i + 1
            break
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            parts = line.split()
        
        if len(parts) >= 2:
            try:
                tmu = int(parts[0])  # Time in microseconds
                code = int(parts[1])  # Event code
                
                # Convert microseconds to sample (assuming 1000 Hz default, will resample later)
                # Common BrainVision sampling rates: 1000, 500, 256 Hz
                # We'll use 1000 Hz as default and resample to 256 Hz later
                if sfreq is None:
                    sfreq = 1000.0  # Default, will be adjusted based on file
                
                sample = int(tmu / 1e6 * sfreq)
                events.append([sample, 0, code])
            except (ValueError, IndexError):
                continue
    
    if len(events) == 0:
        return None, sfreq
    
    return np.array(events, dtype=int), sfreq

def load_brainvision_dat(dat_path, evt_path, n_channels=64, sfreq=1000.0):
    """
    Load BrainVision .dat file directly.
    Assumes IEEE_FLOAT_32 format, MULTIPLEXED orientation.
    """
    dat_path = Path(dat_path)
    evt_path = Path(evt_path)
    
    # Read events first to get timing info
    events_list = []
    times_microsec = []
    
    with open(evt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    start_idx = 0
    for i, line in enumerate(lines):
        if 'Tmu' in line or 'Code' in line:
            start_idx = i + 1
            break
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t') if '\t' in line else line.split()
        if len(parts) >= 2:
            try:
                tmu = int(parts[0])
                code = int(parts[1])
                times_microsec.append(tmu)
                events_list.append([tmu, 0, code])
            except (ValueError, IndexError):
                continue
    
    # Estimate sampling rate from file size and duration
    file_size = dat_path.stat().st_size
    dtype_size = np.dtype(np.float32).itemsize
    total_elements = file_size // dtype_size
    n_samples = total_elements // n_channels
    
    # Calculate sfreq from duration if we have events
    if len(times_microsec) > 1:
        duration_sec = (max(times_microsec) - min(times_microsec)) / 1e6
        if duration_sec > 0:
            estimated_sfreq = n_samples / duration_sec
            # Round to nearest common value
            if estimated_sfreq > 800:
                sfreq = 1000.0
            elif estimated_sfreq > 400:
                sfreq = 500.0
            else:
                sfreq = 256.0
    
    # Read binary data
    with open(dat_path, 'rb') as f:
        data_bytes = f.read()
    
    # Reshape data (MULTIPLEXED: ch1_sample1, ch2_sample1, ..., chN_sample1, ch1_sample2, ...)
    data = np.frombuffer(data_bytes, dtype=np.float32)
    n_samples_actual = len(data) // n_channels
    data = data[:n_samples_actual * n_channels].reshape(n_samples_actual, n_channels).T  # (channels, samples)
    
    # Create channel names (will be renamed later)
    ch_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    # Create MNE Raw object
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # Convert events to correct sample indices
    if events_list:
        events = []
        for tmu, _, code in events_list:
            # Convert microseconds to sample index
            sample = int(tmu / 1e6 * sfreq)
            events.append([sample, 0, code])
        events = np.array(events, dtype=int)
    else:
        events = None
    
    return raw, events

def map_64ch_to_1020():
    """
    Map 64-channel layout (Ch1-Ch64) to 10-20 system channel names.
    Based on standard 64-channel extended 10-20 system layout.
    """
    # Standard 64-channel to 10-20 mapping
    # This is a common mapping - adjust if your specific cap layout differs
    ch64_to_1020 = {
        # Frontal
        'Ch1': 'FP1', 'Ch2': 'FPZ', 'Ch3': 'FP2',
        'Ch4': 'AF7', 'Ch5': 'AF3', 'Ch6': 'AFZ', 'Ch7': 'AF4', 'Ch8': 'AF8',
        'Ch9': 'F7', 'Ch10': 'F5', 'Ch11': 'F3', 'Ch12': 'F1', 'Ch13': 'FZ', 
        'Ch14': 'F2', 'Ch15': 'F4', 'Ch16': 'F6', 'Ch17': 'F8',
        # Central
        'Ch18': 'FT7', 'Ch19': 'FC5', 'Ch20': 'FC3', 'Ch21': 'FC1', 'Ch22': 'FCZ',
        'Ch23': 'FC2', 'Ch24': 'FC4', 'Ch25': 'FC6', 'Ch26': 'FT8',
        'Ch27': 'T7', 'Ch28': 'C5', 'Ch29': 'C3', 'Ch30': 'C1', 'Ch31': 'CZ',
        'Ch32': 'C2', 'Ch33': 'C4', 'Ch34': 'C6', 'Ch35': 'T8',
        # Parietal
        'Ch36': 'TP7', 'Ch37': 'CP5', 'Ch38': 'CP3', 'Ch39': 'CP1', 'Ch40': 'CPZ',
        'Ch41': 'CP2', 'Ch42': 'CP4', 'Ch43': 'CP6', 'Ch44': 'TP8',
        'Ch45': 'P7', 'Ch46': 'P5', 'Ch47': 'P3', 'Ch48': 'P1', 'Ch49': 'PZ',
        'Ch50': 'P2', 'Ch51': 'P4', 'Ch52': 'P6', 'Ch53': 'P8',
        # Occipital
        'Ch54': 'PO7', 'Ch55': 'PO3', 'Ch56': 'POZ', 'Ch57': 'PO4', 'Ch58': 'PO8',
        'Ch59': 'O1', 'Ch60': 'OZ', 'Ch61': 'O2',
        # Additional channels (may not be in standard 10-20 system)
        'Ch62': 'IZ', 'Ch63': 'NZ', 'Ch64': 'M1'  # or other reference channels
    }
    return ch64_to_1020

def rename_channels_to_1020(raw):
    """
    Rename channels to standard 10-20 names.
    Handles both Ch1-Ch64 format and already-named channels.
    """
    rename_dict = {}
    ch_upper = [ch.upper() for ch in raw.ch_names]
    
    # Check if channels are in Ch1-Ch64 format (case-insensitive)
    if any(ch.startswith('CH') and ch[2:].isdigit() for ch in ch_upper):
        # Map Ch1-Ch64 to 10-20 system
        ch64_mapping = map_64ch_to_1020()
        # Create uppercase version of mapping for lookup
        ch64_mapping_upper = {k.upper(): v for k, v in ch64_mapping.items()}
        for i, ch in enumerate(raw.ch_names):
            ch_upper_name = ch.upper()
            if ch_upper_name in ch64_mapping_upper:
                rename_dict[ch] = ch64_mapping_upper[ch_upper_name]
        print(f"[INFO] Mapping {len(rename_dict)} channels from Ch1-Ch64 format to 10-20 system")
    else:
        # Try to match existing channel names to 10-20 system (20 channels)
        standard_names = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ',
                          'C3', 'C4', 'CZ', 'P3', 'P4', 'PZ',
                          'T7', 'T8', 'P7', 'P8', 'O1', 'O2', 'OZ']
        
        for std_name in standard_names:
            std_upper = std_name.upper()
            for i, ch in enumerate(ch_upper):
                if ch == std_upper or ch.replace(' ', '') == std_upper:
                    if raw.ch_names[i] not in rename_dict.values():
                        rename_dict[raw.ch_names[i]] = std_name
        
        if rename_dict:
            print(f"[INFO] Renamed {len(rename_dict)} channels to 10-20 system")
    
    if rename_dict:
        raw.rename_channels(rename_dict)
    
    return raw

# -----------------
# Main Processing Function
# -----------------
def process_subjects():
    """
    Process all subjects with 20 channels from 10-20 system.
    """
    all_X = []
    all_y = []
    all_sub = []
    
    # Find all .dat files
    dat_files = sorted(glob.glob(str(DATA_ROOT / "Study*" / "P_*" / "*.dat")))
    
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found under {DATA_ROOT}")
    
    print(f"\n{'='*70}")
    print(f"[INFO] Processing with {NUM_CHANNELS} channels from 10-20 system")
    print(f"{'='*70}")
    print(f"[INFO] Found {len(dat_files)} .dat files")
    
    # Group by subject
    subjects = {}
    for dat_path in dat_files:
        dat_path = Path(dat_path)
        sub_id = dat_path.parent.name  # P_XX
        study = dat_path.parent.parent.name  # Study One/Two
        
        # Create unique subject ID
        unique_sub_id = f"{study}_{sub_id}"
        
        if unique_sub_id not in subjects:
            subjects[unique_sub_id] = []
        subjects[unique_sub_id].append(dat_path)
    
    print(f"[INFO] Found {len(subjects)} unique subjects")
    
    for unique_sub_id, dat_files_subj in subjects.items():
        print(f"\n[INFO] Processing {unique_sub_id}: {len(dat_files_subj)} files")
        
        subj_X = []
        subj_y = []
        
        for dat_path in dat_files_subj:
            evt_path = dat_path.with_suffix('.evt')
            
            if not evt_path.exists():
                print(f"[WARN] {dat_path.name}: No .evt file found, skipping")
                continue
            
            try:
                # Auto-detect channel count by trying different values
                # Based on file size and duration, determine best fit
                file_size = dat_path.stat().st_size
                dtype_size = 4  # float32
                total_elements = file_size // dtype_size
                
                # Read event times to get duration
                times_microsec = []
                with open(evt_path, 'r') as f:
                    lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.split('\t') if '\t' in line else line.split()
                    if len(parts) >= 1:
                        try:
                            tmu = int(parts[0])
                            times_microsec.append(tmu)
                        except ValueError:
                            continue
                
                duration_sec = (max(times_microsec) - min(times_microsec)) / 1e6 if times_microsec else None
                
                # Try different channel counts and pick the one that gives reasonable sfreq
                raw = None
                events = None
                best_n_ch = None
                
                for n_ch in [64, 32, 128]:
                    try:
                        n_samples = total_elements // n_ch
                        if duration_sec and duration_sec > 0:
                            estimated_sfreq = n_samples / duration_sec
                            # Prefer sfreq close to common values (256, 500, 1000)
                            if 200 <= estimated_sfreq <= 1200:
                                test_raw, test_events = load_brainvision_dat(dat_path, evt_path, n_channels=n_ch, sfreq=1000.0)
                                if test_raw.n_times > 0:
                                    raw = test_raw
                                    events = test_events
                                    best_n_ch = n_ch
                                    break
                        else:
                            # If no duration info, just try loading
                            test_raw, test_events = load_brainvision_dat(dat_path, evt_path, n_channels=n_ch, sfreq=1000.0)
                            if test_raw.n_times > 0:
                                raw = test_raw
                                events = test_events
                                best_n_ch = n_ch
                                break
                    except Exception as e:
                        continue
                
                if raw is None or raw.n_times == 0:
                    print(f"[WARN] {dat_path.name}: Failed to load, skipping")
                    continue
                
                # Downsample if needed
                if abs(raw.info["sfreq"] - SFREQ_TARGET) > 1e-6:
                    original_sfreq = raw.info["sfreq"]
                    raw.resample(SFREQ_TARGET, verbose=False)
                    # Adjust events
                    if events is not None:
                        events[:, 0] = (events[:, 0] * SFREQ_TARGET / original_sfreq).astype(int)
                
                # Rename channels to 10-20 system
                raw = rename_channels_to_1020(raw)
                
                # Select 20 channels from 10-20 system
                if USE_10_20_SYSTEM:
                    raw = select_10_20_channels(raw)
                
                # Filter
                raw.filter(LOWCUT, HIGHCUT, verbose=False)
                
                # Filter events to pain codes only
                if events is not None and len(events) > 0:
                    pain_events = events[np.isin(events[:, 2], PAIN_CODES)]
                    _debug_counts(f"{unique_sub_id} {dat_path.name} pain events", pain_events)
                    
                    if len(pain_events) == 0:
                        print(f"[WARN] {dat_path.name}: No pain events found, skipping")
                        continue
                    
                    # Create epochs
                    epoch_tmin = BASE_TMIN
                    epoch_tmax = PAIN_TMAX
                    
                    event_id = {f"pain_{code}": code for code in PAIN_CODES}
                    epochs = make_epochs(raw, pain_events, event_id, epoch_tmin, epoch_tmax)
                    
                    if len(epochs) == 0:
                        print(f"[WARN] {dat_path.name}: No epochs created, skipping")
                        continue
                    
                    X_full = epochs.get_data()  # (trials, channels, samples)
                    n_trials, n_channels, n_samples = X_full.shape
                    
                    # Extract baseline and pain segments
                    sfreq = epochs.info['sfreq']
                    base_start = 0
                    base_end = int((BASE_TMAX - BASE_TMIN) * sfreq)
                    pain_start = int((PAIN_TMIN - BASE_TMIN) * sfreq)
                    pain_end = int((PAIN_TMAX - BASE_TMIN) * sfreq)
                    
                    X_base = X_full[:, :, base_start:base_end]
                    X_pain = X_full[:, :, pain_start:pain_end]
                    
                    # Pad baseline to match pain duration
                    n_samples_pain = X_pain.shape[2]
                    n_samples_base = X_base.shape[2]
                    if n_samples_base < n_samples_pain:
                        pad_width = n_samples_pain - n_samples_base
                        X_base = np.pad(X_base, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
                    
                    # Combine baseline and pain
                    X_file = np.concatenate([X_base, X_pain], axis=0)
                    y_file = np.concatenate([
                        np.zeros(n_trials, dtype=int),
                        np.ones(n_trials, dtype=int)
                    ])
                    
                    subj_X.append(X_file)
                    subj_y.append(y_file)
                    
                    print(f"[DEBUG] {dat_path.name}: {n_trials} epochs → {len(X_file)} trials")
                else:
                    print(f"[WARN] {dat_path.name}: No events found, skipping")
                    continue
                    
            except Exception as e:
                print(f"[ERROR] {dat_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(subj_X) == 0:
            print(f"[WARN] {unique_sub_id}: No data collected, skipping")
            continue
        
        # Concatenate all files for this subject
        X_subj = np.concatenate(subj_X, axis=0)
        y_subj = np.concatenate(subj_y, axis=0)
        
        # Balance within subject if requested
        if BALANCE_PER_SUBJECT:
            keep = balance_indices(y_subj)
            X_subj, y_subj = X_subj[keep], y_subj[keep]
        
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_sub.append(np.array([unique_sub_id] * len(y_subj), dtype="U32"))
        
        print(f"[DEBUG] {unique_sub_id} kept: pain={int((y_subj==1).sum())}, base={int((y_subj==0).sum())}, total={len(y_subj)}")
    
    if not all_X:
        raise RuntimeError("No epochs collected across subjects.")
    
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subs_all = np.concatenate(all_sub, axis=0)
    
    # Global balance (if not per-subject)
    if not BALANCE_PER_SUBJECT:
        keep = balance_indices(y_all)
        X_all, y_all, subs_all = X_all[keep], y_all[keep], subs_all[keep]
    
    # Limit to MAX_TRIALS if specified (sample down to match target dataset size)
    if MAX_TRIALS > 0 and len(y_all) > MAX_TRIALS:
        print(f"\n[INFO] Limiting trials from {len(y_all)} to {MAX_TRIALS}")
        # Sample down to MAX_TRIALS while maintaining balance
        idx_pos = np.where(y_all == 1)[0]
        idx_neg = np.where(y_all == 0)[0]
        n_per_class = MAX_TRIALS // 2
        if len(idx_pos) > n_per_class:
            idx_pos = rng.choice(idx_pos, n_per_class, replace=False)
        if len(idx_neg) > n_per_class:
            idx_neg = rng.choice(idx_neg, n_per_class, replace=False)
        keep = np.r_[idx_pos, idx_neg]
        keep.sort()
        X_all, y_all, subs_all = X_all[keep], y_all[keep], subs_all[keep]
    
    print(f"\n[CHECK] Final label counts: pain={int((y_all==1).sum())}, baseline={int((y_all==0).sum())}, total={len(y_all)}")
    print(f"[CHECK] X shape: {X_all.shape}  (trials, channels, samples)")
    print(f"[CHECK] sfreq: {SFREQ_TARGET}")
    
    # Save to file
    out_npz = OUT_DIR / "osf-data_pain.npz"
    out_sfreq_json = OUT_DIR / "sfreq.json"
    
    np.savez_compressed(out_npz, X=X_all, y=y_all, subject=subs_all, sfreq=SFREQ_TARGET)
    with open(out_sfreq_json, "w") as f:
        json.dump({"sfreq": float(SFREQ_TARGET), "num_channels": NUM_CHANNELS}, f)
    
    print(f"[INFO] Saved epochs to {out_npz}")
    print(f"[INFO] Saved sampling freq to {out_sfreq_json}")

# -----------------
# Main Entry Point
# -----------------
def main():
    import sys
    print("Starting preprocessing...", file=sys.stderr, flush=True)
    print(f"USE_10_20_SYSTEM={USE_10_20_SYSTEM}, NUM_CHANNELS={NUM_CHANNELS}", file=sys.stderr, flush=True)
    
    if USE_10_20_SYSTEM:
        print("\n" + "="*70)
        print(f"PROCESSING WITH {NUM_CHANNELS} CHANNELS FROM 10-20 SYSTEM")
        print("="*70)
        process_subjects()
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
    else:
        print("[ERROR] USE_10_20_SYSTEM must be True")
        sys.exit(1)


if __name__ == "__main__":
    main()

