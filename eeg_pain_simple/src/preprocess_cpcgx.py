#!/usr/bin/env python3
"""
Preprocessing for Chronic Pain Resting-State EEG (cpCGX_BIDS)

- Loads all subjects' BrainVision EEG (.vhdr) using manual parser
- Filters 1–45 Hz, downsamples to 256 Hz
- Renames channels to standard 10-20 names
- ICA artifact removal (blink + EOG)
- Splits continuous data into fixed-length epochs:
    EO = eyes open (label = 0)
    EC = eyes closed (label = 1)
- Saves packed dataset: X, y, subject, sfreq
"""

import argparse
import json
import numpy as np
from pathlib import Path
import struct
import mne

# -----------------------
# CONFIG
# -----------------------
LOWCUT = 1.0
HIGHCUT = 45.0
SFREQ_TARGET = 256
EPOCH_LEN = 4.0
EPOCH_OVERLAP = 2.0

# -----------------------
# Manual BrainVision Parser
# -----------------------
def parse_vhdr(vhdr_path):
    """Parse .vhdr file manually."""
    config = {}
    ch_names = []
    
    with open(vhdr_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if current_section == 'Common Infos':
                config[key] = value
            elif current_section == 'Channel Infos':
                if key.startswith('Ch'):
                    # Format: Ch1=Name,Ref,Resolution,Unit
                    parts = value.split(',')
                    ch_name = parts[0] if parts else key
                    ch_names.append(ch_name)
    
    return config, ch_names

def load_brainvision_manual(vhdr_path):
    """Load BrainVision file manually (bypasses MNE parser)."""
    vhdr_path = Path(vhdr_path)
    
    # Parse header
    config, ch_names = parse_vhdr(vhdr_path)
    
    # Get paths
    data_file = vhdr_path.parent / config.get('DataFile', vhdr_path.stem + '.eeg')
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Get sampling rate
    sampling_interval_us = float(config.get('SamplingInterval', '2000'))
    sfreq = 1e6 / sampling_interval_us  # Convert microseconds to Hz
    
    # Get number of channels
    n_channels = int(config.get('NumberOfChannels', len(ch_names)))
    
    # Read binary data
    binary_format = config.get('BinaryFormat', 'IEEE_FLOAT_32')
    data_orientation = config.get('DataOrientation', 'MULTIPLEXED')
    
    with open(data_file, 'rb') as f:
        if binary_format == 'IEEE_FLOAT_32':
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported binary format: {binary_format}")
        
        # Read all data
        data_bytes = f.read()
        dtype_size = np.dtype(dtype).itemsize
        total_elements = len(data_bytes) // dtype_size
        n_samples = total_elements // n_channels
        
        # Reshape based on orientation
        if data_orientation == 'MULTIPLEXED':
            # Format: ch1_sample1, ch2_sample1, ..., chN_sample1, ch1_sample2, ...
            data = np.frombuffer(data_bytes[:n_samples * n_channels * dtype_size], dtype=dtype)
            data = data.reshape(n_samples, n_channels).T  # Transpose to (channels, samples)
        else:
            # VECTORIZED: all samples for ch1, then all for ch2, etc.
            data = np.frombuffer(data_bytes[:n_samples * n_channels * dtype_size], dtype=dtype)
            data = data.reshape(n_channels, n_samples)
    
    # Ensure channel names match
    if len(ch_names) != n_channels:
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    # Create MNE Raw object
    info = mne.create_info(ch_names[:n_channels], sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    return raw

def rename_to_1020(raw):
    """Rename channels to standard 10-20."""
    mapping = {
        "Fp1": "FP1", "Fp2": "FP2",
        "F3": "F3", "F4": "F4",
        "C3": "C3", "C4": "C4",
        "P3": "P3", "P4": "P4",
        "O1": "O1", "O2": "O2",
        "F7": "F7", "F8": "F8",
        "T7": "T7", "T8": "T8",
        "P7": "P7", "P8": "P8",
        "Fz": "FZ", "Cz": "CZ", "Pz": "PZ", "Oz": "OZ",
    }
    rename_dict = {k: v for k, v in mapping.items() if k in raw.ch_names}
    if rename_dict:
        raw.rename_channels(rename_dict)
    return raw

def sliding_epochs(raw, length, overlap):
    """Generate sliding windows."""
    step = length - overlap
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    win = int(length * sfreq)
    step_samp = int(step * sfreq)
    X = []
    for start in range(0, n_samples - win + 1, step_samp):
        stop = start + win
        if stop <= n_samples:
            X.append(raw.get_data(start=start, stop=stop))
    if len(X) == 0:
        return np.array([]).reshape(0, raw.n_channels, win)
    return np.stack(X, axis=0)

# -----------------------
# MAIN
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing sub-*/eeg/*.vhdr files")
    ap.add_argument("--out", required=True, help="Output NPZ")
    args = ap.parse_args()
    
    root = Path(args.root)
    files = sorted(root.glob("sub-*/eeg/*.vhdr"))
    
    if not files:
        raise RuntimeError(f"No .vhdr files found in {root}")
    
    print(f"\n{'='*70}")
    print(f"[INFO] Found {len(files)} BrainVision files")
    print(f"[INFO] Using manual BrainVision parser (most accurate)")
    print(f"{'='*70}\n")
    
    all_X, all_y, all_subj = [], [], []
    
    for f in files:
        print(f"\n[INFO] Processing {f.name}")
        sub_id = f.parent.parent.name
        
        try:
            raw = load_brainvision_manual(f)
            raw.load_data(verbose="ERROR")
        except Exception as e:
            print(f"[ERROR] Failed to load {f.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if raw.n_times == 0:
            print(f"[WARN] {sub_id}: Empty file, skipping")
            continue
        
        raw = rename_to_1020(raw)
        
        eeg_ch_names = [ch for ch in raw.ch_names 
                        if ch.upper() not in ['EOG', 'ECG', 'EMG', 'STIM', 'STATUS', 'TRIGGER', 'MISC']]
        
        if len(eeg_ch_names) == 0:
            print(f"[WARN] {sub_id}: No EEG channels found")
            continue
        
        try:
            raw.set_channel_types({ch: 'eeg' for ch in eeg_ch_names}, on_unit_change='ignore')
        except Exception:
            for ch in eeg_ch_names:
                if ch in raw.ch_names:
                    idx = raw.ch_names.index(ch)
                    raw.info['chs'][idx]['kind'] = mne.io.constants.FIFF.FIFFV_EEG_CH
                    raw.info['chs'][idx]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_EEG
        
        picks_eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks_eeg) == 0:
            continue
        
        raw.filter(LOWCUT, HIGHCUT, picks=picks_eeg, verbose=False)
        
        if abs(raw.info["sfreq"] - SFREQ_TARGET) > 1e-6:
            raw.resample(SFREQ_TARGET, verbose=False)
        
        try:
            ica = mne.preprocessing.ICA(n_components=20, random_state=42, verbose="ERROR")
            ica.fit(raw, verbose="ERROR")
            eog_indices, eog_scores = ica.find_bads_eog(raw, verbose="ERROR")
            ica.exclude = eog_indices[:2] if len(eog_indices) > 0 else []
            raw = ica.apply(raw, verbose="ERROR")
            print(f"[INFO] {sub_id}: Applied ICA")
        except Exception:
            pass
        
        if "task-EO" in f.name or "EO" in f.name.upper():
            label = 0
        elif "task-EC" in f.name or "EC" in f.name.upper():
            label = 1
        else:
            continue
        
        X_seg = sliding_epochs(raw, EPOCH_LEN, EPOCH_OVERLAP)
        if len(X_seg) == 0:
            continue
        
        y_seg = np.ones(len(X_seg), dtype=int) * label
        subj_seg = np.array([sub_id] * len(X_seg), dtype="U32")
        
        all_X.append(X_seg)
        all_y.append(y_seg)
        all_subj.append(subj_seg)
        
        print(f"[INFO] {sub_id}: Created {len(X_seg)} epochs (label={label})")
    
    if not all_X:
        raise RuntimeError("No epochs collected")
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y).astype(int)
    subject = np.concatenate(all_subj)
    
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]
    min_size = min(len(class_0_idx), len(class_1_idx))
    if min_size > 0:
        rng = np.random.default_rng(42)
        class_0_balanced = rng.choice(class_0_idx, size=min_size, replace=False)
        class_1_balanced = rng.choice(class_1_idx, size=min_size, replace=False)
        keep_idx = np.concatenate([class_0_balanced, class_1_balanced])
        X = X[keep_idx]
        y = y[keep_idx]
        subject = subject[keep_idx]
    
    print(f"\n{'='*70}")
    print(f"[INFO] Final shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  subjects: {len(np.unique(subject))}")
    print(f"  Labels: EO={int((y==0).sum())}, EC={int((y==1).sum())}")
    print(f"{'='*70}\n")
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, subject=subject, sfreq=SFREQ_TARGET)
    
    print(f"[INFO] Saved {args.out}")

if __name__ == "__main__":
    main()

