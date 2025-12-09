#!/usr/bin/env python3
"""
Riemannian Feature Extraction for ds005284
Enhancements:
1. Multiple covariance estimators (SCM, LWF, OAS, Ledoit-Wolf)
2. Multiple metrics (Riemann, LogEuclid, Euclidean)
3. More frequency bands (Delta, Theta, Alpha, Beta, Gamma)
4. Time-windowed covariance matrices
5. Geodesic distance features
6. Regularized covariance estimation
7. Multi-scale features
"""

import argparse
import json
from pathlib import Path
import numpy as np
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import logm, expm
import warnings
warnings.filterwarnings('ignore')


def riemann_fb_features_basic(X: np.ndarray, sfreq: float,
                              bands=((4, 8), (8, 12), (13, 30), (30, 45))) -> np.ndarray:
    """Original basic Riemannian features (for comparison)"""
    feats = []
    for l, h in bands:
        Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") 
                       for tr in X], axis=0)
        C = Covariances(estimator='lwf').fit_transform(Xb)
        ts = TangentSpace(metric='logeuclid').fit_transform(C)
        feats.append(ts)
    return np.concatenate(feats, axis=1)


def riemann_fb_features_improved(X: np.ndarray, sfreq: float,
                                 bands=((1, 4), (4, 8), (8, 12), (12, 30), (30, 45)),
                                 estimators=['scm', 'lwf', 'oas'],
                                 metrics=['riemann', 'logeuclid'],
                                 use_time_windows=True,
                                 use_geodesic_distances=True) -> np.ndarray:
    """
    Improved Riemannian features with multiple estimators, metrics, and advanced techniques.
    
    Parameters:
    -----------
    X : np.ndarray
        EEG data (n_trials, n_channels, n_samples)
    sfreq : float
        Sampling frequency
    bands : tuple
        Frequency bands (default: Delta, Theta, Alpha, Beta, Gamma)
    estimators : list
        Covariance estimators: 'scm', 'lwf', 'oas', 'ledoit_wolf'
    metrics : list
        Tangent space metrics: 'riemann', 'logeuclid', 'euclid'
    use_time_windows : bool
        Compute covariance for time windows (multi-scale)
    use_geodesic_distances : bool
        Add geodesic distance features
    """
    all_feats = []
    
    # Compute reference covariance (Riemannian mean across all trials, all bands)
    # This will be used for geodesic distances
    ref_covs = []
    for l, h in bands:
        Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") 
                       for tr in X], axis=0)
        C = Covariances(estimator='scm').fit_transform(Xb)
        ref_covs.append(mean_covariance(C, metric='riemann'))
    
    # Process each frequency band
    for band_idx, (l, h) in enumerate(bands):
        # Filter data for this band
        Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") 
                       for tr in X], axis=0)
        
        band_feats = []
        
        # Multiple covariance estimators
        for estimator in estimators:
            try:
                # Compute covariance matrices
                C = Covariances(estimator=estimator).fit_transform(Xb)
                
                # Multiple tangent space metrics
                for metric in metrics:
                    ts = TangentSpace(metric=metric).fit_transform(C)
                    band_feats.append(ts)
                
                # Geodesic distances to reference
                if use_geodesic_distances and estimator == 'scm':
                    ref_cov = ref_covs[band_idx]
                    distances = np.array([distance_riemann(c, ref_cov) for c in C])
                    distances = distances.reshape(-1, 1)  # (n_trials, 1)
                    band_feats.append(distances)
                
            except Exception as e:
                print(f"[WARN] Failed estimator {estimator} for band ({l}, {h}): {e}")
                continue
        
        # Time-windowed covariance (multi-scale)
        if use_time_windows:
            n_samples = Xb.shape[2]
            window_sizes = [n_samples // 4, n_samples // 2, n_samples]  # 25%, 50%, 100%
            
            for window_size in window_sizes:
                if window_size < 10:  # Too small
                    continue
                
                # Compute covariance for each window
                window_covs = []
                for trial in Xb:
                    # Use sliding windows
                    n_windows = max(1, n_samples // window_size)
                    trial_covs = []
                    
                    for w in range(n_windows):
                        start = w * window_size
                        end = min(start + window_size, n_samples)
                        window_data = trial[:, start:end]
                        
                        # Compute covariance for this window
                        if window_data.shape[1] > window_data.shape[0]:  # More samples than channels
                            C_window = np.cov(window_data)
                            # Regularize
                            C_window += 1e-6 * np.eye(C_window.shape[0])
                            trial_covs.append(C_window)
                    
                    if trial_covs:
                        # Average covariance across windows
                        C_trial = np.mean(trial_covs, axis=0)
                        window_covs.append(C_trial)
                
                if window_covs:
                    window_covs = np.array(window_covs)
                    # Map to tangent space
                    try:
                        ts_window = TangentSpace(metric='logeuclid').fit_transform(window_covs)
                        band_feats.append(ts_window)
                    except:
                        pass
        
        if band_feats:
            all_feats.extend(band_feats)
    
    if not all_feats:
        raise ValueError("No features extracted!")
    
    return np.concatenate(all_feats, axis=1)


def riemann_fb_features_advanced(X: np.ndarray, sfreq: float,
                                 bands=((1, 4), (4, 8), (8, 12), (12, 30), (30, 45))) -> np.ndarray:
    """
    Most advanced Riemannian features:
    - Multiple estimators and metrics
    - Time-windowed covariance
    - Geodesic distances
    - Cross-band covariance features
    """
    # Standard improved features
    feats_improved = riemann_fb_features_improved(
        X, sfreq, bands=bands,
        estimators=['scm', 'lwf', 'oas'],
        metrics=['riemann', 'logeuclid'],
        use_time_windows=True,
        use_geodesic_distances=True
    )
    
    # Cross-band covariance features
    # Compute covariance for each band separately, then compute cross-band correlations
    band_covs = []
    for l, h in bands:
        Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") 
                       for tr in X], axis=0)
        C = Covariances(estimator='scm').fit_transform(Xb)
        # Extract diagonal (variance) and upper triangle (correlations)
        diag = np.array([np.diag(c) for c in C])  # (n_trials, n_channels)
        band_covs.append(diag)
    
    # Cross-band correlations
    if len(band_covs) > 1:
        cross_band_feats = []
        for i in range(len(band_covs)):
            for j in range(i+1, len(band_covs)):
                # Correlation between band i and band j variances
                cross_corr = np.array([np.corrcoef(band_covs[i][k], band_covs[j][k])[0, 1] 
                                      for k in range(len(X))])
                cross_corr = cross_corr.reshape(-1, 1)
                cross_band_feats.append(cross_corr)
        
        if cross_band_feats:
            cross_band_feats = np.concatenate(cross_band_feats, axis=1)
            feats_improved = np.hstack([feats_improved, cross_band_feats])
    
    return feats_improved


def main():
    ap = argparse.ArgumentParser(description="Improved Riemannian feature extraction")
    ap.add_argument("--npz", default="../packed/ds005284_pain.npz", 
                   help="Input preprocessed NPZ")
    ap.add_argument("--out", default="../packed/features_ds005284_riemannian_improved.npz", 
                   help="Output features NPZ")
    ap.add_argument("--mode", choices=['basic', 'improved', 'advanced'], default='advanced',
                   help="Feature extraction mode")
    ap.add_argument("--bands", nargs='+', type=float, default=None,
                   help="Frequency bands as pairs: --bands 1 4 4 8 8 12 12 30 30 45")
    args = ap.parse_args()

    # Load data
    print(f"[INFO] Loading data from {args.npz}...")
    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    sfreq = float(data["sfreq"])

    print(f"[INFO] Data shape: {X.shape}")
    print(f"[INFO] Sampling frequency: {sfreq} Hz")
    
    # Parse frequency bands
    if args.bands:
        bands = []
        for i in range(0, len(args.bands), 2):
            if i+1 < len(args.bands):
                bands.append((args.bands[i], args.bands[i+1]))
    else:
        # Default: Delta, Theta, Alpha, Beta, Gamma
        bands = ((1, 4), (4, 8), (8, 12), (12, 30), (30, 45))
    
    print(f"[INFO] Frequency bands: {bands}")
    print(f"[INFO] Mode: {args.mode}")

    # Extract features
    print(f"[INFO] Extracting {args.mode} Riemannian features...")
    if args.mode == 'basic':
        Z = riemann_fb_features_basic(X, sfreq, bands=bands)
    elif args.mode == 'improved':
        Z = riemann_fb_features_improved(X, sfreq, bands=bands)
    else:  # advanced
        Z = riemann_fb_features_advanced(X, sfreq, bands=bands)

    print(f"[INFO] Extracted features shape: {Z.shape}")
    print(f"[INFO] Feature dimension: {Z.shape[1]}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=Z, y=y, subject=subject)
    
    # Save metadata
    metadata = {
        "n_features": int(Z.shape[1]),
        "n_trials": int(Z.shape[0]),
        "mode": args.mode,
        "bands": bands,
        "sfreq": float(sfreq)
    }
    metadata_file = Path(args.out).with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Features saved to {args.out}")
    print(f"[INFO] Metadata saved to {metadata_file}")
    print(f"[INFO] Feature extraction complete!")


if __name__ == "__main__":
    main()

