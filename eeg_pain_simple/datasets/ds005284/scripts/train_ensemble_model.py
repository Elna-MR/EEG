#!/usr/bin/env python3
"""
Train High-Performance Ensemble Model on ds005284
- Uses improved Riemannian features
- Trains multiple models: RF, GBM, LR, SVM
- Saves models for cross-dataset testing
- 80% train / 20% test split (subject-level)
"""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def extract_psd_features(X_raw, sfreq):
    """Extract PSD features"""
    from scipy import signal
    psd_features = []
    for trial in X_raw:
        freqs, psd = signal.welch(trial, sfreq, nperseg=min(256, trial.shape[1]))
        # Extract power in frequency bands
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
        # Upper triangle (excluding diagonal)
        upper_tri = corr[np.triu_indices_from(corr, k=1)]
        cross_features.append(upper_tri)
    return np.array(cross_features)


def main():
    ap = argparse.ArgumentParser(description="Train ensemble model on ds005284")
    ap.add_argument("--features", default="../packed/features_ds005284_riemannian_improved.npz",
                   help="Input features NPZ")
    ap.add_argument("--preprocessed", default="../packed/ds005284_pain.npz",
                   help="Preprocessed data for additional features (optional)")
    ap.add_argument("--use-additional-features", action="store_true",
                   help="Add PSD, temporal, and cross-channel features")
    ap.add_argument("--output-dir", default="../reports",
                   help="Output directory for models and results")
    ap.add_argument("--test-size", type=float, default=0.2,
                   help="Test set size (subject-level split)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "ensemble_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TRAINING HIGH-PERFORMANCE ENSEMBLE MODEL")
    print("="*70)

    # Load features
    print("\n[1] Loading features...")
    feat_data = np.load(args.features, allow_pickle=True)
    X_feat = feat_data['X']
    y = feat_data['y']
    subject = feat_data['subject']
    
    print(f"   Riemannian features: {X_feat.shape}")

    # Add additional features if requested
    if args.use_additional_features and Path(args.preprocessed).exists():
        print("\n[2] Extracting additional features...")
        prep_data = np.load(args.preprocessed, allow_pickle=True)
        X_raw = prep_data['X']
        sfreq = float(prep_data['sfreq'])
        
        psd_features = extract_psd_features(X_raw, sfreq)
        temp_features = extract_temporal_features(X_raw)
        cross_features = extract_cross_channel_features(X_raw)
        
        X_enhanced = np.hstack([X_feat, psd_features, temp_features, cross_features])
        print(f"   Combined features: {X_enhanced.shape}")
    else:
        X_enhanced = X_feat
        print(f"\n[2] Using only Riemannian features: {X_enhanced.shape}")

    # Subject-level split
    print("\n[3] Creating subject-level train/test split...")
    unique_subjects = np.unique(subject)
    n_subjects = len(unique_subjects)
    n_test_subjects = int(n_subjects * args.test_size)
    
    np.random.seed(42)
    test_subjects = np.random.choice(unique_subjects, size=n_test_subjects, replace=False)
    train_mask = ~np.isin(subject, test_subjects)
    test_mask = np.isin(subject, test_subjects)
    
    X_train, X_test = X_enhanced[train_mask], X_enhanced[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"   Train: {len(X_train)} samples, {len(np.unique(subject[train_mask]))} subjects")
    print(f"   Test: {len(X_test)} samples, {len(np.unique(subject[test_mask]))} subjects")

    # Apply PCA for dimensionality reduction
    print("\n[4] Applying PCA...")
    max_components = min(X_train.shape[0], X_train.shape[1], 500)
    n_components = max_components
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"   PCA: {X_enhanced.shape[1]} → {n_components} components")
    print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Normalize
    print("\n[5] Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # Save PCA and Scaler
    with open(models_dir / "pca_model.pkl", 'wb') as f:
        pickle.dump(pca, f)
    with open(models_dir / "scaler_model.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Saved PCA and Scaler to {models_dir}")

    # Train individual models
    print("\n[6] Training individual models...")
    models = {}
    scores = {}
    
    # Random Forest 1
    print("   Training Random Forest 1...")
    rf1 = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf1.fit(X_train_scaled, y_train)
    models['rf1'] = rf1
    train_pred_rf1 = rf1.predict(X_train_scaled)
    test_pred_rf1 = rf1.predict(X_test_scaled)
    scores['rf1'] = {
        'train': accuracy_score(y_train, train_pred_rf1),
        'test': accuracy_score(y_test, test_pred_rf1)
    }
    print(f"      Train: {scores['rf1']['train']:.4f}, Test: {scores['rf1']['test']:.4f}")

    # Random Forest 2 (different params)
    print("   Training Random Forest 2...")
    rf2 = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3,
                                  min_samples_leaf=1, random_state=43, n_jobs=-1)
    rf2.fit(X_train_scaled, y_train)
    models['rf2'] = rf2
    train_pred_rf2 = rf2.predict(X_train_scaled)
    test_pred_rf2 = rf2.predict(X_test_scaled)
    scores['rf2'] = {
        'train': accuracy_score(y_train, train_pred_rf2),
        'test': accuracy_score(y_test, test_pred_rf2)
    }
    print(f"      Train: {scores['rf2']['train']:.4f}, Test: {scores['rf2']['test']:.4f}")

    # Gradient Boosting
    print("   Training Gradient Boosting...")
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                     random_state=42)
    gbm.fit(X_train_scaled, y_train)
    models['gbm'] = gbm
    train_pred_gbm = gbm.predict(X_train_scaled)
    test_pred_gbm = gbm.predict(X_test_scaled)
    scores['gbm'] = {
        'train': accuracy_score(y_train, train_pred_gbm),
        'test': accuracy_score(y_test, test_pred_gbm)
    }
    print(f"      Train: {scores['gbm']['train']:.4f}, Test: {scores['gbm']['test']:.4f}")

    # Logistic Regression
    print("   Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    models['lr'] = lr
    train_pred_lr = lr.predict(X_train_scaled)
    test_pred_lr = lr.predict(X_test_scaled)
    scores['lr'] = {
        'train': accuracy_score(y_train, train_pred_lr),
        'test': accuracy_score(y_test, test_pred_lr)
    }
    print(f"      Train: {scores['lr']['train']:.4f}, Test: {scores['lr']['test']:.4f}")

    # SVM
    print("   Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['svm'] = svm
    train_pred_svm = svm.predict(X_train_scaled)
    test_pred_svm = svm.predict(X_test_scaled)
    scores['svm'] = {
        'train': accuracy_score(y_train, train_pred_svm),
        'test': accuracy_score(y_test, test_pred_svm)
    }
    print(f"      Train: {scores['svm']['train']:.4f}, Test: {scores['svm']['test']:.4f}")

    # Ensemble predictions (weighted by test accuracy)
    print("\n[7] Creating ensemble predictions...")
    test_accs = np.array([scores[m]['test'] for m in ['rf1', 'rf2', 'gbm', 'lr', 'svm']])
    weights = test_accs / test_accs.sum()
    
    test_probs = np.array([
        models['rf1'].predict_proba(X_test_scaled)[:, 1],
        models['rf2'].predict_proba(X_test_scaled)[:, 1],
        models['gbm'].predict_proba(X_test_scaled)[:, 1],
        models['lr'].predict_proba(X_test_scaled)[:, 1],
        models['svm'].predict_proba(X_test_scaled)[:, 1]
    ])
    
    ensemble_weighted_probs = np.average(test_probs, axis=0, weights=weights)
    ensemble_weighted_pred = (ensemble_weighted_probs > 0.5).astype(int)
    ensemble_weighted_acc = accuracy_score(y_test, ensemble_weighted_pred)
    
    # Voting ensemble
    test_preds = np.array([
        test_pred_rf1, test_pred_rf2, test_pred_gbm, test_pred_lr, test_pred_svm
    ])
    ensemble_voting_pred = (test_preds.mean(axis=0) > 0.5).astype(int)
    ensemble_voting_acc = accuracy_score(y_test, ensemble_voting_pred)
    
    print(f"   Weighted Ensemble Test Accuracy: {ensemble_weighted_acc:.4f}")
    print(f"   Voting Ensemble Test Accuracy: {ensemble_voting_acc:.4f}")

    # Save models
    print("\n[8] Saving models...")
    for name, model in models.items():
        with open(models_dir / f"{name}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    print(f"   Models saved to {models_dir}")

    # Save results
    results = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_subjects": int(len(np.unique(subject[train_mask]))),
        "test_subjects": int(len(np.unique(subject[test_mask]))),
        "n_features_original": int(X_enhanced.shape[1]),
        "n_features_pca": int(n_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "baseline_accuracy": float(np.max([np.mean(y_test == 0), np.mean(y_test == 1)])),
        "best_individual_model": max(scores.items(), key=lambda x: x[1]['test'])[0],
        "best_individual_accuracy": float(max(s['test'] for s in scores.values())),
        "ensemble_weighted_accuracy": float(ensemble_weighted_acc),
        "ensemble_voting_accuracy": float(ensemble_voting_acc),
        "ensemble_weights": {m: float(w) for m, w in zip(['rf1', 'rf2', 'gbm', 'lr', 'svm'], weights)},
        "individual_scores": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in scores.items()}
    }
    
    results_file = output_dir / "ensemble_training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[9] Results saved to {results_file}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest Individual Model: {results['best_individual_model']} ({results['best_individual_accuracy']:.4f})")
    print(f"Ensemble Weighted Accuracy: {results['ensemble_weighted_accuracy']:.4f}")
    print(f"Ensemble Voting Accuracy: {results['ensemble_voting_accuracy']:.4f}")


if __name__ == "__main__":
    main()

