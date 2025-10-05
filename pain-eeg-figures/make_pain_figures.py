#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate EEG figures illustrating pain-related spectral changes.
- Figure 1: Band power (theta/alpha/gamma) low vs high
- Figure 2: Topomaps for alpha & gamma (low/high/delta)
- Figure 3: Time–frequency (example)

Usage:
    python make_pain_figures.py --data data --labels data/labels.csv --lpf 70 --gamma-high 70
"""

import argparse, os, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import spectrogram, welch

warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Folder with EEG files (.edf/.bdf/.vhdr/.fif/.gdf)")
    ap.add_argument("--labels", type=str, default=None, help="Optional CSV with columns: file, condition or pain_score")
    ap.add_argument("--out", type=str, default="figures", help="Output folder for figures")
    ap.add_argument("--montage", type=str, default="standard_1020", help="Montage name (e.g., standard_1020)")
    ap.add_argument("--hpf", type=float, default=1.0, help="High-pass filter (Hz)")
    ap.add_argument("--lpf", type=float, default=45.0, help="Low-pass filter (Hz)")
    ap.add_argument("--notch", type=float, default=50.0, help="Notch frequency (50 or 60)")
    ap.add_argument("--psd-seg", type=float, default=4.0, help="Welch segment length (s)")
    ap.add_argument("--gamma-high", type=float, default=45.0, help="Upper bound for gamma band (Hz)")
    ap.add_argument("--save-spectrogram", action="store_true", help="Save time–frequency figure")
    return ap.parse_args()

def band_power(psd, freqs, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if idx.sum() == 0:
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, idx], freqs[idx], axis=1)

def main():
    args = parse_args()
    DATA_DIR = Path(args.data)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # bands (theta/alpha always fixed; gamma configurable up to 45–100 Hz)
    BANDS = {"theta": (4, 8), "alpha": (8, 12), "gamma": (30, float(args.gamma_high))}

    # Load labels if present
    labels = None
    if args.labels and Path(args.labels).exists():
        labels = pd.read_csv(args.labels)

    # Gather files
    patterns = ("*.edf","*.bdf","*.vhdr","*.fif","*.gdf")
    files = sorted(sum([glob.glob(str(DATA_DIR / p)) for p in patterns], []))
    if not files:
        raise SystemExit(f"No EEG files found in {DATA_DIR}. Supported: {', '.join(patterns)}")

    rows = []
    last_info = None

    # Precompute montage
    montage = None
    if args.montage:
        try:
            montage = mne.channels.make_standard_montage(args.montage)
        except Exception:
            montage = None

    for f in files:
        # Read raw data
        if f.endswith(".vhdr"):
            raw = mne.io.read_raw_brainvision(f, preload=True, verbose=False)
        elif f.endswith(".gdf"):
            raw = mne.io.read_raw_gdf(f, preload=True, verbose=False)
        else:
            raw = mne.io.read_raw(f, preload=True, verbose=False)

        # Set montage if available
        if montage is not None:
            try:
                raw.set_montage(montage, on_missing="ignore")
            except Exception:
                pass

        # Basic filtering + notch
        raw.filter(args.hpf, args.lpf, verbose=False)
        try:
            raw.notch_filter(args.notch, verbose=False)
        except Exception:
            pass

        # Average reference (safe default)
        try:
            raw.set_eeg_reference("average", projection=False)
        except Exception:
            pass

        sfreq = raw.info["sfreq"]
        last_info = raw.info  # keep most recent info for topomap

        # PSD via Welch
        n_per_seg = max(int(args.psd_seg * sfreq), 256)
        picks = mne.pick_types(raw.info, eeg=True)
        data = raw.get_data(picks=picks)
        
        # Compute PSD for each channel
        psd_list = []
        for ch_data in data:
            freqs, psd_ch = welch(ch_data, fs=sfreq, nperseg=n_per_seg)
            psd_list.append(psd_ch)
        
        psd = np.array(psd_list)
        
        # Filter frequencies
        freq_mask = (freqs >= args.hpf) & (freqs <= args.lpf)
        freqs = freqs[freq_mask]
        psd = psd[:, freq_mask]

        ch_names = np.array(raw.info["ch_names"])
        band_vals = {b: band_power(psd, freqs, lo, hi) for b, (lo, hi) in BANDS.items()}

        subj = os.path.basename(f)
        condition = None
        pain_score = None
        if labels is not None:
            row = labels.loc[labels['file'] == os.path.basename(f)]
            if not row.empty:
                if 'condition' in row.columns and pd.notna(row['condition'].values[0]):
                    condition = str(row['condition'].values[0]).strip().lower()
                if 'pain_score' in row.columns and pd.notna(row['pain_score'].values[0]):
                    try:
                        pain_score = float(row['pain_score'].values[0])
                    except Exception:
                        pain_score = None

        for i, ch in enumerate(ch_names):
            d = {"file": subj, "channel": ch, "condition": condition, "pain_score": pain_score}
            for b in BANDS:
                d[b] = float(band_vals[b][i]) if i < len(band_vals[b]) else np.nan
            rows.append(d)

    df = pd.DataFrame(rows)

    # Determine groups: low/high
    if "condition" in df.columns and df["condition"].notna().any():
        df["group"] = df["condition"].fillna("low")
    elif "pain_score" in df.columns and df["pain_score"].notna().any():
        med = df.groupby('file')['pain_score'].transform('median')
        df["group"] = np.where(df["pain_score"] >= med, "high", "low")
    else:
        # fallback: alpha median per file as proxy
        ap = df.groupby('file')['alpha'].transform('median')
        thr = ap.median()
        df["group"] = np.where(ap >= thr, "high", "low")

    # ---------- Figure 1: spectral bars ----------
    fig1 = plt.figure(figsize=(6, 4))
    bands = list(BANDS.keys())
    means = []
    sems = []
    for g in ["low", "high"]:
        gdf = df[df["group"] == g].groupby("file")[bands].mean()
        means.append(gdf.mean(0).values)
        sems.append(gdf.sem(0).values)
    means = np.vstack(means); sems = np.vstack(sems)
    x = np.arange(len(bands)); w = 0.35
    for i, g in enumerate(["low", "high"]):
        plt.bar(x + i*w, means[i], w, yerr=sems[i], label=g, capsize=3)
    plt.xticks(x + w/2, bands)
    plt.ylabel("Band power (a.u.)")
    plt.title("Spectral power by pain group")
    plt.legend()
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "fig1_spectral_power.png", dpi=300)

    # ---------- Figure 2: topomaps ----------
    # Need channel positions in last_info; if missing, skip gracefully
    have_pos = last_info is not None and len(mne.pick_types(last_info, eeg=True)) > 0
    if have_pos:
        for band in ["alpha", "gamma"]:
            try:
                maps = {}
                for g in ["low", "high"]:
                    maps[g] = df[df["group"] == g].groupby("channel")[band].mean()
                chans = last_info["ch_names"]
                vals_low = np.array([maps["low"].get(ch, np.nan) for ch in chans])
                vals_high = np.array([maps["high"].get(ch, np.nan) for ch in chans])
                delta = vals_high - vals_low

                for name, vals in [("low", vals_low), ("high", vals_high), ("delta", delta)]:
                    fig = plt.figure(figsize=(3.8, 3.4))
                    mne.viz.plot_topomap(vals, last_info, show=False)
                    plt.title(f"{band} – {name}")
                    fig.tight_layout()
                    fig.savefig(OUT_DIR / f"fig2_topo_{band}_{name}.png", dpi=300)
            except Exception:
                pass

    # ---------- Figure 3: time–frequency ----------
    if args.save_spectrogram:
        # pick first file again
        f0 = files[0]
        if f0.endswith(".vhdr"):
            raw0 = mne.io.read_raw_brainvision(f0, preload=True, verbose=False)
        elif f0.endswith(".gdf"):
            raw0 = mne.io.read_raw_gdf(f0, preload=True, verbose=False)
        else:
            raw0 = mne.io.read_raw(f0, preload=True, verbose=False)
        raw0.filter(args.hpf, args.lpf, verbose=False)
        picks = mne.pick_types(raw0.info, eeg=True)
        data = raw0.get_data(picks=picks).mean(0)
        fs = raw0.info["sfreq"]
        f, t, Sxx = spectrogram(data, fs=fs, nperseg=1024, noverlap=512)
        plt.figure(figsize=(6, 3.2))
        plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading="auto")
        plt.ylim(0, args.lpf)
        plt.xlabel("Time (s)"); plt.ylabel("Hz"); plt.title("Time–frequency (example)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig3_timefreq_example.png", dpi=300)

    print(f"Done. Figures saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
