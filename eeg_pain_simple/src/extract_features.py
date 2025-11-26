import argparse
import json
from pathlib import Path
import numpy as np
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def riemann_fb_features(X: np.ndarray, sfreq: float,
                        bands=((4, 8), (8, 12), (13, 30), (30, 45))) -> np.ndarray:
    feats = []
    for l, h in bands:
        Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") for tr in X], axis=0)
        C = Covariances(estimator='lwf').fit_transform(Xb)
        ts = TangentSpace(metric='logeuclid').fit_transform(C)
        feats.append(ts)
    return np.concatenate(feats, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Input preprocessed NPZ")
    ap.add_argument("--out", required=True, help="Output features NPZ")
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


