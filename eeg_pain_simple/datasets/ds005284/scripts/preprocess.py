#!/usr/bin/env python3
"""
Preprocessing for ds005284:
- Loads all subjects' .bdf EEG
- Filters 1–45 Hz
- Uses ONLY condition 54 events (removes condition 64)
- Event 54 onset = pain start point (t=0.0s)
- Baseline: -1.5 to 0.0s (pre-stim, before pain)
- Pain: 0.0 to 2.0s (post-stim, after pain onset)
- Balances classes (undersample majority)
- Saves packed/ds005284_pain.npz with X (trials, channels, samples), y (0=baseline,1=pain), sfreq
"""

import os
import json
import glob
import numpy as np
import mne
from pathlib import Path

# -----------------
# Config (edit if needed)
# -----------------
# DATA_ROOT can be set via EEG_DATA_ROOT env var, or will try common paths
_default_paths = ["../data", "data", ".", ".."]
_env_root = os.environ.get("EEG_DATA_ROOT")
DATA_ROOT = Path(_env_root) if _env_root else None
if DATA_ROOT is None:
    for p in _default_paths:
        test = Path(p) / "sub-001" / "eeg"
        if test.exists():
            DATA_ROOT = Path(p)
            break
if DATA_ROOT is None:
    DATA_ROOT = Path(".")  # fallback

SUB_GLOB  = "sub-*/eeg/*_task-26ByBiosemi_eeg.bdf"

LOWCUT, HIGHCUT = 1.0, 45.0       # Hz
SFREQ_TARGET = 256.0              # Hz (downsample to this, if needed)

PAIN_CODE = 54  # Only use condition 54 (pain events)
# NOPAIN_CODE = 64  # Removed - no longer using condition 64

# Epoch timing relative to condition 54 onset (where pain starts)
BASE_TMIN, BASE_TMAX = -1.5, 0.0  # Baseline: 1.5s BEFORE event onset (pre-stim)
PAIN_TMIN, PAIN_TMAX = 0.0, 2.0   # Pain: 0.0 to 2.0s AFTER event onset (post-stim)

BALANCE_PER_SUBJECT = False       # set True to balance inside each subject; False = balance globally
RANDOM_STATE = 42

# 10-20 system channel selection
USE_10_20_SYSTEM = True           # Set True to select only 10-20 system channels
NUM_CHANNELS = 20                  # Number of 10-20 system channels (20 channels: FP1, FP2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, T7, T8, P7, P8, O1, O2, OZ)

OUT_DIR = Path("packed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NPZ = OUT_DIR / "ds005284_pain.npz"  # Default (will be overridden when processing)
OUT_SFREQ_JSON = OUT_DIR / "sfreq.json"

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

def rename_channels_from_biosemi(raw, sub_id):
    """
    Rename channels from Biosemi A1-A32, B1-B32 format to standard 10-20 names.
    Uses the mapping from channels.tsv file.
    """
    # Mapping from Biosemi channel names to standard 10-20 names
    # Based on the channels.tsv file structure
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
    
    # Rename channels if they are in Biosemi format
    if any(ch.startswith('A') or ch.startswith('B') for ch in raw.ch_names):
        rename_dict = {}
        for ch in raw.ch_names:
            if ch in channel_mapping:
                rename_dict[ch] = channel_mapping[ch]
        if rename_dict:
            raw.rename_channels(rename_dict)
            print(f"[INFO] {sub_id}: Renamed {len(rename_dict)} channels from Biosemi format")
    
    return raw

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
            # Find original case version
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
        # Reorder channels to match target order
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

    bdf_files = sorted(glob.glob(str(DATA_ROOT / SUB_GLOB)))
    if not bdf_files:
        raise FileNotFoundError(f"No .bdf files found under {DATA_ROOT} with pattern {SUB_GLOB}")

    print(f"\n{'='*70}")
    print(f"[INFO] Processing with {NUM_CHANNELS} channels from 10-20 system")
    print(f"{'='*70}")
    print(f"[INFO] Found {len(bdf_files)} subjects")

    for bdf_path in bdf_files:
        sub_id = Path(bdf_path).parts[-3]  # sub-XXX from sub-XXX/eeg/file
        print(f"\n[INFO] Processing {sub_id}: {bdf_path}")

        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        
        # Downsample FIRST if needed (before finding events)
        if abs(raw.info["sfreq"] - SFREQ_TARGET) > 1e-6:
            raw.resample(SFREQ_TARGET, verbose=False)
        
        # Find events (Status channel still available)
        stim_channel = None
        if 'Status' in raw.ch_names:
            stim_channel = 'Status'
        elif 'STATUS' in [ch.upper() for ch in raw.ch_names]:
            stim_channel = [ch for ch in raw.ch_names if ch.upper() == 'STATUS'][0]
        
        events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose=False)
        _debug_counts(f"{sub_id} raw events (all)", events)
        
        # Filter to only condition 54 (pain events) - remove condition 64
        if len(events) > 0:
            events_54 = events[events[:, 2] == PAIN_CODE]
            events = events_54
            _debug_counts(f"{sub_id} events (condition 54 only)", events)
            if len(events) == 0:
                print(f"[WARN] {sub_id}: No condition 54 events found; skipping subject")
                continue
        else:
            print(f"[WARN] {sub_id}: No events found; skipping subject")
            continue
        
        # Exclude non-EEG channels (like 'Status') after finding events
        eeg_chs = [ch for ch in raw.ch_names if ch.upper() not in ['STATUS', 'TRIGGER']]
        if len(eeg_chs) < len(raw.ch_names):
            raw.pick_channels(eeg_chs)

        # Rename channels from Biosemi format (A1-A32, B1-B32) to standard 10-20 names
        raw = rename_channels_from_biosemi(raw, sub_id)

        # Select 20 channels from 10-20 system
        if USE_10_20_SYSTEM:
            print(f"[INFO] {sub_id}: Selecting {NUM_CHANNELS} 10-20 system channels")
            raw = select_10_20_channels(raw)
            print(f"[INFO] {sub_id}: Channels after selection: {len(raw.ch_names)}")

        # Filter
        raw.filter(LOWCUT, HIGHCUT, verbose=False)

        # Create epochs from condition 54 events
        # Epoch window: -1.5s to 2.0s (baseline: -1.5 to 0.0, pain: 0.0 to 2.0)
        # Event onset (0.0s) is where pain starts
        epoch_tmin = BASE_TMIN  # -1.5s (baseline starts)
        epoch_tmax = PAIN_TMAX  # 2.0s (pain ends)
        
        pain_epochs = make_epochs(raw, events, { "pain": PAIN_CODE }, epoch_tmin, epoch_tmax)
        if len(pain_epochs) == 0:
            print(f"[WARN] {sub_id}: no epochs found; skipping subject")
            continue
        
        X_full = pain_epochs.get_data()  # (trials, channels, samples)
        n_trials, n_channels, n_samples = X_full.shape
        
        # Calculate sample indices relative to epoch start
        # Epoch goes from -1.5s to 2.0s, so total duration is 3.5s
        sfreq = pain_epochs.info['sfreq']
        epoch_duration = epoch_tmax - epoch_tmin  # 3.5s
        
        # Baseline: from -1.5s to 0.0s (relative to epoch start at -1.5s)
        base_start = 0  # Start of epoch (corresponds to -1.5s)
        base_end = int((BASE_TMAX - BASE_TMIN) * sfreq)  # 1.5s into epoch (corresponds to 0.0s)
        
        # Pain: from 0.0s to 2.0s (relative to epoch start at -1.5s)
        pain_start = int((PAIN_TMIN - BASE_TMIN) * sfreq)  # 1.5s into epoch (corresponds to 0.0s)
        pain_end = int((PAIN_TMAX - BASE_TMIN) * sfreq)    # 3.5s into epoch (corresponds to 2.0s)
        
        # Extract baseline segments (-1.5 to 0.0, 1.5 seconds)
        X_base = X_full[:, :, base_start:base_end]
        # Extract pain segments (0.0 to 2.0, 2.0 seconds)  
        X_pain = X_full[:, :, pain_start:pain_end]
        
        # Pad baseline to match pain duration (1.5s -> 2.0s) for consistent array shapes
        # Note: Padding won't affect Riemannian features much since covariance summarizes temporal info
        n_samples_pain = X_pain.shape[2]
        n_samples_base = X_base.shape[2]
        if n_samples_base < n_samples_pain:
            pad_width = n_samples_pain - n_samples_base
            X_base = np.pad(X_base, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        # Combine: each original epoch gives one baseline and one pain trial
        X = np.concatenate([X_base, X_pain], axis=0)  # (2*n_trials, channels, samples)
        y = np.concatenate([
            np.zeros(n_trials, dtype=int),  # baseline trials
            np.ones(n_trials, dtype=int)    # pain trials
        ])
        
        print(f"[DEBUG] {sub_id}: {n_trials} original epochs → {n_trials} baseline + {n_trials} pain = {len(X)} total")

        # Optional: balance within subject
        if BALANCE_PER_SUBJECT:
            keep = balance_indices(y)
            X, y = X[keep], y[keep]

        all_X.append(X)
        all_y.append(y)
        all_sub.append(np.array([sub_id] * len(y), dtype="U16"))

        print(f"[DEBUG] {sub_id} kept: pain={int((y==1).sum())}, base={int((y==0).sum())}, total={len(y)}")

    if not all_X:
        raise RuntimeError("No epochs collected across subjects.")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subs_all = np.concatenate(all_sub, axis=0)

    # Global balance (if not per-subject)
    if not BALANCE_PER_SUBJECT:
        keep = balance_indices(y_all)
        X_all, y_all, subs_all = X_all[keep], y_all[keep], subs_all[keep]

    print(f"\n[CHECK] Final label counts: pain={int((y_all==1).sum())}, baseline={int((y_all==0).sum())}, total={len(y_all)}")
    print(f"[CHECK] X shape: {X_all.shape}  (trials, channels, samples)")
    print(f"[CHECK] sfreq: {SFREQ_TARGET}")

    # Save to file
    out_npz = OUT_DIR / "ds005284_pain.npz"
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




