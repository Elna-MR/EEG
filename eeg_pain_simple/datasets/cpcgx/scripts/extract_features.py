import argparse
import json
from pathlib import Path
import numpy as np
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def riemann_fb_features(X: np.ndarray, sfreq: float,
                        bands=((4, 8), (8, 12), (13, 30), (30, 45))) -> np.ndarray:
    """
    Extract Riemannian features using frequency-band covariance matrices.
    Optimized to process all samples at once instead of one-by-one.
    """
    feats = []
    print(f"[INFO] Processing {len(X)} samples through {len(bands)} frequency bands...")
    
    for band_idx, (l, h) in enumerate(bands):
        print(f"[INFO] Band {band_idx+1}/{len(bands)}: {l}-{h} Hz...", end=" ", flush=True)
        
        # Process all samples at once (much faster than loop)
        # Reshape to (n_samples * n_channels, n_times) for batch filtering
        n_samples, n_channels, n_times = X.shape
        X_reshaped = X.reshape(-1, n_times).astype(np.float64)
        
        # Filter all at once
        Xb_filtered = filter_data(X_reshaped, sfreq=sfreq, l_freq=l, h_freq=h, 
                                  verbose="ERROR", n_jobs=1)
        
        # Reshape back to (n_samples, n_channels, n_times)
        Xb = Xb_filtered.reshape(n_samples, n_channels, -1)
        
        # Compute covariance matrices
        C = Covariances(estimator='lwf').fit_transform(Xb)
        
        # Map to tangent space
        ts = TangentSpace(metric='logeuclid').fit_transform(C)
        feats.append(ts)
        
        print(f"Done ({ts.shape[1]} features)")
    
    return np.concatenate(feats, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../packed/cpcgx_pain.npz", help="Input preprocessed NPZ (default: ../packed/cpcgx_pain.npz)")
    ap.add_argument("--out", default="../packed/features_cpcgx.npz", help="Output features NPZ (default: ../packed/features_cpcgx.npz)")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    sfreq = float(data["sfreq"])

    Z = riemann_fb_features(X, sfreq)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=Z, y=y, subject=subject)
    print(f"[INFO] Features {Z.shape} saved to {args.out}")


if __name__ == "__main__":
    main()


