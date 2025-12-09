#!/usr/bin/env python3
"""
Test the trained Ensemble model (from ds005284) on osf-data
- Loads the trained ensemble models (RF, GBM, LR, SVM)
- Loads osf-data improved Riemannian features
- Applies same PCA and scaling transformations
- Tests the ensemble and reports accuracy
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def extract_psd_features(X_raw, sfreq):
    """Extract PSD features"""
    from scipy import signal
    psd_features = []
    for trial in X_raw:
        freqs, psd = signal.welch(trial, sfreq, nperseg=min(256, trial.shape[1]))
        delta = np.mean(psd[:, (freqs >= 1) & (freqs < 4)], axis=1)
        theta = np.mean(psd[:, (freqs >= 4) & (freqs < 8)], axis=1)
        alpha = np.mean(psd[:, (freqs >= 8) & (freqs < 12)], axis=1)
        beta = np.mean(psd[:, (freqs >= 12) & (freqs < 30)], axis=1)
        gamma = np.mean(psd[:, (freqs >= 30) & (freqs < 45)], axis=1)
        psd_features.append(np.concatenate([delta, theta, alpha, beta, gamma]))
    return np.array(psd_features)


def extract_temporal_features(X_raw):
    """Extract temporal features"""
    temp_features = []
    for trial in X_raw:
        feat = []
        for ch in trial:
            feat.extend([
                np.mean(ch), np.std(ch), np.median(ch),
                np.percentile(ch, 25), np.percentile(ch, 75),
                np.max(ch) - np.min(ch), np.var(ch)
            ])
        temp_features.append(feat)
    return np.array(temp_features)


def extract_cross_channel_features(X_raw):
    """Extract cross-channel correlation features"""
    cross_features = []
    for trial in X_raw:
        corr = np.corrcoef(trial)
        upper_tri = corr[np.triu_indices_from(corr, k=1)]
        cross_features.append(upper_tri)
    return np.array(cross_features)


def main():
    ap = argparse.ArgumentParser(description="Test ensemble model on osf-data")
    ap.add_argument("--models-dir", default="../../ds005284/reports/ensemble_models",
                   help="Directory containing saved ensemble models")
    ap.add_argument("--features", default="../packed/features_osf-data_riemannian_improved.npz",
                   help="osf-data improved Riemannian features")
    ap.add_argument("--preprocessed", default="../packed/osf-data_pain.npz",
                   help="Preprocessed data for additional features (optional)")
    ap.add_argument("--use-additional-features", action="store_true",
                   help="Add PSD, temporal, and cross-channel features")
    ap.add_argument("--output", default="../reports/ensemble_test_on_osf_data.json",
                   help="Output JSON file for results")
    args = ap.parse_args()

    print("="*70)
    print("TESTING ENSEMBLE MODEL ON osf-data")
    print("="*70)

    # Load osf-data features
    print("\n[1] Loading osf-data features...")
    feat_data = np.load(args.features, allow_pickle=True)
    X_osf = feat_data['X']
    y_osf = feat_data['y']
    subject_osf = feat_data['subject']
    
    print(f"   osf-data features: {X_osf.shape}")
    print(f"   osf-data samples: {len(X_osf)}")
    print(f"   osf-data classes: Baseline={np.sum(y_osf==0)}, Pain={np.sum(y_osf==1)}")
    print(f"   osf-data subjects: {len(np.unique(subject_osf))}")

    # Add additional features if requested
    if args.use_additional_features and Path(args.preprocessed).exists():
        print("\n[2] Extracting additional features...")
        prep_data = np.load(args.preprocessed, allow_pickle=True)
        X_raw = prep_data['X']
        sfreq = float(prep_data['sfreq'])
        
        psd_features = extract_psd_features(X_raw, sfreq)
        temp_features = extract_temporal_features(X_raw)
        cross_features = extract_cross_channel_features(X_raw)
        
        X_enhanced = np.hstack([X_osf, psd_features, temp_features, cross_features])
        print(f"   Combined features: {X_enhanced.shape}")
    else:
        X_enhanced = X_osf
        print(f"\n[2] Using only Riemannian features: {X_enhanced.shape}")

    # Load PCA and Scaler from training
    print("\n[3] Loading PCA and Scaler from training...")
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    with open(models_dir / "pca_model.pkl", 'rb') as f:
        pca = pickle.load(f)
    with open(models_dir / "scaler_model.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"   PCA components: {pca.n_components_}")
    
    # Apply transformations
    print("\n[4] Applying PCA and normalization...")
    # Handle feature dimension mismatch
    expected_features = pca.n_features_in_
    actual_features = X_enhanced.shape[1]
    if actual_features != expected_features:
        print(f"   [WARN] Feature dimension mismatch: expected {expected_features}, got {actual_features}")
        if actual_features > expected_features:
            # Truncate extra features
            X_enhanced = X_enhanced[:, :expected_features]
            print(f"   Truncated to {expected_features} features")
        else:
            # Pad with zeros
            padding = np.zeros((X_enhanced.shape[0], expected_features - actual_features))
            X_enhanced = np.hstack([X_enhanced, padding])
            print(f"   Padded to {expected_features} features")
    
    X_osf_pca = pca.transform(X_enhanced)
    X_osf_scaled = scaler.transform(X_osf_pca)
    print(f"   Transformed features: {X_osf_scaled.shape}")

    # Load ensemble models
    print("\n[5] Loading ensemble models...")
    model_names = ['rf1', 'rf2', 'gbm', 'lr', 'svm']
    models = {}
    for name in model_names:
        model_path = models_dir / f"{name}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"   Loaded {name}")

    # Load training results to get ensemble weights
    training_results_path = models_dir.parent / "ensemble_training_results.json"
    if training_results_path.exists():
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        ensemble_weights = training_results.get('ensemble_weights', {})
        print(f"\n   Ensemble weights: {ensemble_weights}")
    else:
        # Default equal weights if results not found
        ensemble_weights = {m: 0.2 for m in model_names}
        print(f"\n   Using equal weights (training results not found)")

    # Test individual models
    print("\n[6] Testing individual models...")
    individual_scores = {}
    all_probs = []
    
    for name in model_names:
        model = models[name]
        pred = model.predict(X_osf_scaled)
        prob = model.predict_proba(X_osf_scaled)[:, 1]
        acc = accuracy_score(y_osf, pred)
        individual_scores[name] = float(acc)
        all_probs.append(prob)
        print(f"   {name}: {acc:.4f} ({acc*100:.2f}%)")

    # Ensemble predictions
    print("\n[7] Computing ensemble predictions...")
    weights_array = np.array([ensemble_weights.get(m, 0.2) for m in model_names])
    weights_array = weights_array / weights_array.sum()  # Normalize
    
    # Weighted ensemble
    ensemble_weighted_probs = np.average(all_probs, axis=0, weights=weights_array)
    ensemble_weighted_pred = (ensemble_weighted_probs > 0.5).astype(int)
    ensemble_weighted_acc = accuracy_score(y_osf, ensemble_weighted_pred)
    
    # Voting ensemble
    all_preds = np.array([models[m].predict(X_osf_scaled) for m in model_names])
    ensemble_voting_pred = (all_preds.mean(axis=0) > 0.5).astype(int)
    ensemble_voting_acc = accuracy_score(y_osf, ensemble_voting_pred)
    
    # Per-class accuracy
    baseline_mask = y_osf == 0
    pain_mask = y_osf == 1
    baseline_acc = accuracy_score(y_osf[baseline_mask], ensemble_weighted_pred[baseline_mask]) if baseline_mask.sum() > 0 else 0.0
    pain_acc = accuracy_score(y_osf[pain_mask], ensemble_weighted_pred[pain_mask]) if pain_mask.sum() > 0 else 0.0
    
    # Per-subject accuracy
    subject_accs = {}
    for subj in np.unique(subject_osf):
        subj_mask = subject_osf == subj
        if subj_mask.sum() > 0:
            subj_acc = accuracy_score(y_osf[subj_mask], ensemble_weighted_pred[subj_mask])
            subject_accs[str(subj)] = float(subj_acc)

    print(f"\n   ============================================================")
    print(f"   ENSEMBLE TEST RESULTS ON osf-data:")
    print(f"   ============================================================")
    print(f"   Weighted Ensemble Accuracy: {ensemble_weighted_acc:.4f} ({ensemble_weighted_acc*100:.2f}%)")
    print(f"   Voting Ensemble Accuracy: {ensemble_voting_acc:.4f} ({ensemble_voting_acc*100:.2f}%)")
    print(f"   Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"   Pain Accuracy: {pain_acc:.4f} ({pain_acc*100:.2f}%)")
    print(f"   Test Samples: {len(y_osf)}")
    print(f"   Test Subjects: {len(np.unique(subject_osf))}")
    print(f"   ============================================================")

    # Save results
    results = {
        "model_source": "ds005284",
        "test_dataset": "osf-data",
        "overall_accuracy_weighted": float(ensemble_weighted_acc),
        "overall_accuracy_voting": float(ensemble_voting_acc),
        "baseline_accuracy": float(baseline_acc),
        "pain_accuracy": float(pain_acc),
        "test_samples": int(len(y_osf)),
        "test_subjects": int(len(np.unique(subject_osf))),
        "n_features": int(X_osf_scaled.shape[1]),
        "pca_components": int(pca.n_components_),
        "individual_scores": individual_scores,
        "per_subject_accuracy": subject_accs
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[8] Results saved to: {args.output}")
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

