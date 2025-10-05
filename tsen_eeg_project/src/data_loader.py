import os
from typing import List, Tuple
import numpy as np
import mne

def list_edf_files(data_dir: str) -> List[str]:
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith((".edf", ".gdf"))]

def load_eeg_edf(path: str, resample_hz: int = None) -> Tuple[np.ndarray, int, list]:
    """Load EDF/GDF via MNE. Returns (data[n_channels, n_samples], fs, ch_names)."""
    if path.lower().endswith('.gdf'):
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    if resample_hz is not None and abs(raw.info["sfreq"] - resample_hz) > 1e-3:
        raw.resample(resample_hz)
    data = raw.get_data()  # n_channels x n_samples
    fs = int(raw.info["sfreq"])
    ch_names = raw.info["ch_names"]
    # Z-score per channel
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
    return data, fs, ch_names
