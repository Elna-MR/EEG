#!/usr/bin/env python3
"""
Sanity check script to verify the preprocessed dataset.
"""

import numpy as np
from collections import Counter


def main():
    D = np.load("packed/ds005284_pain.npz", allow_pickle=True)
    X, y, subj = D["X"], D["y"], D["subject"]
    
    print("=" * 60)
    print("DATASET SANITY CHECK")
    print("=" * 60)
    print(f"\nEpochs shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Subjects shape: {subj.shape}")
    
    print(f"\nClass counts: {Counter(y.tolist())}")
    print(f"Unique subjects: {len(np.unique(subj))}")
    
    print(f"\n{'Subject':<12} {'Total':<8} {'Pain':<8} {'Baseline':<8}")
    print("-" * 40)
    
    # Per-subject label counts
    for s in np.unique(subj):
        idx = (subj == s)
        c = Counter(y[idx].tolist())
        n_total = idx.sum()
        n_pain = c.get(1, 0)
        n_base = c.get(0, 0)
        print(f"{s:<12} {n_total:<8} {n_pain:<8} {n_base:<8}")
    
    print("\n" + "=" * 60)
    print(f"Total trials: {len(y)}")
    print(f"Total Pain: {Counter(y.tolist())[1]}")
    print(f"Total Baseline: {Counter(y.tolist())[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()

