#!/usr/bin/env python3
"""
Subject-wise cross-validation SVM training with GroupKFold.
This ensures no data leakage between subjects.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    # Load features
    F = np.load("packed/features.npz", allow_pickle=True)
    X = F["X"]           # (n_trials, n_features) - fixed: was "X_feat", should be "X"
    y = F["y"]           # 0/1
    groups = F["subject"]  # subject IDs per trial (str or int)
    
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    print(f"Unique subjects: {len(np.unique(groups))}")
    print(f"Class distribution: {np.bincount(y)}")
    print()
    
    # Subject-wise 5-fold CV
    gkf = GroupKFold(n_splits=5)
    
    accs, aucs = [], []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        
        print(f"Fold {fold}: Train={len(Xtr)} ({np.sum(ytr==0)} baseline, {np.sum(ytr==1)} pain), "
              f"Val={len(Xva)} ({np.sum(yva==0)} baseline, {np.sum(yva==1)} pain)")
        
        clf = make_pipeline(
            StandardScaler(), 
            SVC(kernel="rbf", class_weight="balanced", probability=True)
        )
        clf.fit(Xtr, ytr)
        
        p = clf.predict(Xva)
        proba = clf.predict_proba(Xva)[:, 1]
        
        acc = accuracy_score(yva, p)
        auc = roc_auc_score(yva, proba)
        
        accs.append(acc)
        aucs.append(auc)
        fold_results.append({
            "fold": fold,
            "train_size": len(Xtr),
            "val_size": len(Xva),
            "accuracy": float(acc),
            "auroc": float(auc)
        })
        
        print(f"  Accuracy: {acc:.3f}, AUROC: {auc:.3f}")
    
    print()
    print("=" * 70)
    print("Subject-wise CV Results:")
    print("=" * 70)
    print(f"ACC: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"AUROC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print("=" * 70)
    
    # Save results
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "cv_method": "GroupKFold (subject-wise)",
        "n_splits": 5,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_auroc": float(np.mean(aucs)),
        "std_auroc": float(np.std(aucs)),
        "fold_results": fold_results
    }
    
    with open(report_dir / "svm_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to {report_dir}/svm_cv_results.json")


if __name__ == "__main__":
    main()

