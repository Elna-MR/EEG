import os, json, argparse
import numpy as np
import pandas as pd

from .config import TsEnConfig, TrainConfig
from .data_loader import list_edf_files, load_eeg_edf
from .features import compute_tsen_features
from .utils import read_labels
from .model import train_eval_knn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to directory with EDF files")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--labels", type=str, default="", help="CSV with columns [filename,label]")
    ap.add_argument("--no-train", action="store_true", help="Only extract features, skip training")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = TsEnConfig()
    tcfg = TrainConfig()

    edf_paths = list_edf_files(args.data)
    if not edf_paths:
        print(f"No EDF files found in {args.data}")
        return

    rows = []
    for path in edf_paths:
        try:
            data, fs, ch_names = load_eeg_edf(path, resample_hz=cfg.fs_target)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        feats = compute_tsen_features(
            data=data,
            fs=fs,
            bands=cfg.bands,
            qs=cfg.qs,
            window_sec=cfg.window_sec,
            step_sec=cfg.step_sec,
            bins=cfg.hist_bins
        )
        row = {"filename": os.path.basename(path)}
        row.update(feats)
        rows.append(row)

    feat_df = pd.DataFrame(rows).fillna(0.0)
    feat_csv = os.path.join(args.out, "features.csv")
    feat_df.to_csv(feat_csv, index=False)
    print(f"Wrote features to {feat_csv} (shape={feat_df.shape})")

    if args.no_train or not args.labels:
        print("Skipping training (no labels or --no-train).")
        return

    labels_df = read_labels(args.labels)
    merged = feat_df.merge(labels_df, on="filename", how="inner")
    if merged.empty:
        print("No overlap between feature filenames and labels.csv")
        return

    # Prepare X, y
    y = merged["label"].astype("category").cat.codes.values
    X = merged.drop(columns=["filename", "label"]).values

    acc, f1 = train_eval_knn(X, y, k=tcfg.k_neighbors, n_splits=tcfg.n_splits, random_state=tcfg.random_state)
    metrics = {"accuracy_mean_cv": acc, "f1_mean_cv": f1}
    with open(os.path.join(args.out, "knn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("CV metrics:", metrics)

if __name__ == "__main__":
    main()
