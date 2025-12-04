#!/usr/bin/env python3
"""
Subject-level performance analysis and visualization:
- Per-subject accuracy (3-5 subjects shown)
- Subject-specific bias reduction demonstration
- Example covariance matrices
- Riemannian mean trajectories
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
import warnings
warnings.filterwarnings('ignore')

# Import DANN model from train_dann.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_dann import DANN, GradientReversal


def load_trained_model(model_path, input_dim, num_domains, device='cpu'):
    """Load a trained DANN model from checkpoint."""
    model = DANN(input_dim=input_dim, num_domains=num_domains)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def compute_per_subject_metrics(model, X, y, subject, scaler, device='cpu'):
    """Compute accuracy per subject."""
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model.classifier(model.feature(X_tensor)).squeeze(1)
        preds_tensor = (torch.sigmoid(logits) > 0.5).long().cpu()
        preds = np.array(preds_tensor.tolist())
    
    subject_accs = {}
    unique_subjects = np.unique(subject)
    for sub in unique_subjects:
        mask = subject == sub
        if mask.sum() > 0:
            acc = accuracy_score(y[mask], preds[mask])
            subject_accs[sub] = {
                'accuracy': acc,
                'n_samples': mask.sum(),
                'n_pain': int(y[mask].sum()),
                'n_baseline': int((~y[mask].astype(bool)).sum())
            }
    return subject_accs, preds


def compute_domain_confusion(model, X, subject, scaler, device='cpu'):
    """Compute domain confusion matrix to show bias reduction."""
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    unique_subjects = np.unique(subject)
    subject_to_domain = {sub: idx for idx, sub in enumerate(unique_subjects)}
    domains_true = np.array([subject_to_domain[sub] for sub in subject])
    
    with torch.no_grad():
        z = model.feature(X_tensor)
        dom_logits = model.domain_disc(z)
        dom_preds_tensor = torch.argmax(dom_logits, dim=1).cpu()
        dom_preds = np.array(dom_preds_tensor.tolist())
    
    return domains_true, dom_preds, subject_to_domain


def extract_covariance_matrices(X_raw, sfreq, bands=((4, 8), (8, 12), (13, 30), (30, 45))):
    """Extract covariance matrices for visualization."""
    all_covs = []
    for l, h in bands:
        Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") 
                       for tr in X_raw], axis=0)
        C = Covariances(estimator='lwf').fit_transform(Xb)
        all_covs.append(C)
    return all_covs


def compute_riemannian_mean_trajectory(X_raw, y, subject, sfreq, selected_subjects=None, 
                                      bands=((4, 8), (8, 12), (13, 30), (30, 45))):
    """Compute Riemannian mean covariance for each subject and class."""
    if selected_subjects is None:
        selected_subjects = np.unique(subject)[:5]  # Show first 5 subjects
    
    trajectories = {}
    for sub in selected_subjects:
        mask = subject == sub
        if not mask.any():
            continue
        
        trajectories[sub] = {}
        for class_label in [0, 1]:
            class_mask = mask & (y == class_label)
            if not class_mask.any():
                continue
            
            X_sub_class = X_raw[class_mask]
            covs_per_band = []
            
            for l, h in bands:
                Xb = np.stack([filter_data(tr.astype(np.float64), sfreq=sfreq, l_freq=l, h_freq=h, verbose="ERROR") 
                              for tr in X_sub_class], axis=0)
                C = Covariances(estimator='lwf').fit_transform(Xb)
                # Compute Riemannian mean
                C_mean = mean_covariance(C, metric='riemann')
                covs_per_band.append(C_mean)
            
            trajectories[sub][class_label] = covs_per_band
    
    return trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../packed/features_osf-data.npz", 
                   help="Features NPZ file")
    ap.add_argument("--preprocessed", default="../packed/osf-data_pain.npz",
                   help="Preprocessed NPZ file (for covariance matrices)")
    ap.add_argument("--report-dir", default="../reports",
                   help="Directory to save reports")
    ap.add_argument("--n-subjects", type=int, default=5,
                   help="Number of subjects to show in detail (default: 5)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Device to use for inference")
    args = ap.parse_args()
    
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading data...")
    # Load features
    feat_data = np.load(args.features, allow_pickle=True)
    X_feat = feat_data['X']
    y_feat = feat_data['y']
    subject_feat = feat_data['subject']
    
    # Load preprocessed data for covariance matrices
    prep_data = np.load(args.preprocessed, allow_pickle=True)
    X_raw = prep_data['X']
    y_raw = prep_data['y']
    subject_raw = prep_data['subject']
    sfreq = float(prep_data['sfreq'])
    
    print(f"[INFO] Features shape: {X_feat.shape}")
    print(f"[INFO] Raw data shape: {X_raw.shape}")
    print(f"[INFO] Unique subjects: {len(np.unique(subject_feat))}")
    
    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    
    # Prepare scaler (same as training)
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)
    
    # Split data same way as training
    unique_subjs = np.unique(subject_feat)
    train_subjs, val_subjs = train_test_split(unique_subjs, test_size=0.2, random_state=42)
    val_mask = np.isin(subject_feat, val_subjs)
    
    # Load or retrain model (we'll use the training script's logic)
    print("[INFO] Setting up model...")
    unique_subjects = np.unique(subject_feat)
    subject_to_domain = {sub: idx for idx, sub in enumerate(unique_subjects)}
    domains = np.array([subject_to_domain[sub] for sub in subject_feat])
    
    model = DANN(input_dim=X_feat.shape[1], num_domains=int(domains.max() + 1))
    model = model.to(device)
    
    # Train model (since checkpoints aren't saved, we retrain)
    # This matches the training procedure from train_dann.py
    print("[INFO] Training model for analysis (this may take a few minutes)...")
    train_mask = np.isin(subject_feat, train_subjs)
    X_train = X_feat_scaled[train_mask]
    y_train = y_feat[train_mask]
    d_train = domains[train_mask]
    
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(d_train, dtype=torch.long)
    )
    ds_val = TensorDataset(
        torch.tensor(X_feat_scaled[val_mask], dtype=torch.float32),
        torch.tensor(y_feat[val_mask], dtype=torch.long),
        torch.tensor(domains[val_mask], dtype=torch.long)
    )
    
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=0)
    
    # Use same trainer config as train_dann.py
    if device.type == "cuda":
        trainer = pl.Trainer(max_epochs=30, enable_checkpointing=False, logger=False, accelerator="gpu", devices=1)
    elif device.type == "mps":
        trainer = pl.Trainer(max_epochs=30, enable_checkpointing=False, logger=False, accelerator="mps", devices=1)
    else:
        trainer = pl.Trainer(max_epochs=30, enable_checkpointing=False, logger=False, accelerator="cpu")
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)
    
    model.eval()
    
    # 1. Per-subject performance
    print("[INFO] Computing per-subject metrics...")
    subject_accs, all_preds = compute_per_subject_metrics(
        model, X_feat, y_feat, subject_feat, scaler, device
    )
    
    # Select top N subjects to show (mix of train/val)
    selected_subjects = []
    val_subject_accs = {k: v for k, v in subject_accs.items() if k in val_subjs}
    train_subject_accs = {k: v for k, v in subject_accs.items() if k in train_subjs}
    
    # Show some validation subjects (most important)
    n_val_show = min(args.n_subjects // 2 + 1, len(val_subject_accs))
    selected_subjects.extend(sorted(val_subject_accs.items(), key=lambda x: x[1]['accuracy'])[:n_val_show])
    
    # Fill remaining with train subjects
    n_train_show = args.n_subjects - len(selected_subjects)
    if n_train_show > 0:
        selected_subjects.extend(sorted(train_subject_accs.items(), key=lambda x: x[1]['accuracy'])[:n_train_show])
    
    selected_subjects = [s[0] for s in selected_subjects[:args.n_subjects]]
    
    # 2. Domain confusion (bias reduction)
    print("[INFO] Computing domain confusion...")
    domains_true, domains_pred, subject_to_domain = compute_domain_confusion(
        model, X_feat, subject_feat, scaler, device
    )
    
    # 3. Covariance matrices
    print("[INFO] Extracting covariance matrices...")
    # Select a few example trials
    n_examples = 5
    example_indices = np.random.choice(len(X_raw), n_examples, replace=False)
    X_examples = X_raw[example_indices]
    y_examples = y_raw[example_indices]
    
    bands = ((4, 8), (8, 12), (13, 30), (30, 45))
    covs_examples = extract_covariance_matrices(X_examples, sfreq, bands)
    
    # 4. Riemannian mean trajectories
    print("[INFO] Computing Riemannian mean trajectories...")
    trajectories = compute_riemannian_mean_trajectory(
        X_raw, y_raw, subject_raw, sfreq, selected_subjects, bands
    )
    
    # Create visualizations
    print("[INFO] Creating visualizations...")
    
    # Figure 1: Per-subject performance
    fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
    
    # Bar plot of selected subjects
    selected_data = [(sub, subject_accs[sub]) for sub in selected_subjects if sub in subject_accs]
    subjects_sorted = sorted(selected_data, key=lambda x: x[1]['accuracy'])
    sub_names = [s[0] for s in subjects_sorted]
    accs = [s[1]['accuracy'] for s in subjects_sorted]
    colors = ['#2ecc71' if s[0] in val_subjs else '#3498db' for s in subjects_sorted]
    
    bars = axes1[0].barh(range(len(sub_names)), accs, color=colors, alpha=0.7)
    axes1[0].set_yticks(range(len(sub_names)))
    axes1[0].set_yticklabels(sub_names)
    axes1[0].set_xlabel('Accuracy', fontsize=12)
    axes1[0].set_title(f'Per-Subject Classification Accuracy (showing {len(sub_names)} subjects)', fontsize=14, fontweight='bold')
    axes1[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    axes1[0].legend(['Chance', 'Validation', 'Training'])
    axes1[0].grid(True, alpha=0.3, axis='x')
    axes1[0].set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        axes1[0].text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=10)
    
    # Summary statistics
    all_accs = [v['accuracy'] for v in subject_accs.values()]
    val_accs = [subject_accs[s]['accuracy'] for s in val_subjs if s in subject_accs]
    train_accs = [subject_accs[s]['accuracy'] for s in train_subjs if s in subject_accs]
    
    summary_text = f"""
    Overall Statistics:
    - Mean Accuracy: {np.mean(all_accs):.3f} ± {np.std(all_accs):.3f}
    - Validation Mean: {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}
    - Training Mean: {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f}
    - Min: {np.min(all_accs):.3f} | Max: {np.max(all_accs):.3f}
    """
    
    axes1[1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                  verticalalignment='center', transform=axes1[1].transAxes)
    axes1[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(report_dir / "subject_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Domain confusion matrix (bias reduction)
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix for domain prediction
    n_domains = len(unique_subjects)
    # Sample for visualization if too many subjects
    if n_domains > 20:
        # Show confusion for selected subjects only
        selected_domains = [subject_to_domain[s] for s in selected_subjects]
        mask = np.isin(domains_true, selected_domains)
        cm = confusion_matrix(domains_true[mask], domains_pred[mask], labels=selected_domains)
        domain_labels = selected_subjects
    else:
        cm = confusion_matrix(domains_true, domains_pred)
        domain_labels = [f"Sub{i}" for i in range(n_domains)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0],
                xticklabels=domain_labels[:len(cm)], yticklabels=domain_labels[:len(cm)])
    axes2[0].set_xlabel('Predicted Domain', fontsize=12)
    axes2[0].set_ylabel('True Domain', fontsize=12)
    axes2[0].set_title('Domain Confusion Matrix\n(Lower diagonal = better domain adaptation)', fontsize=12, fontweight='bold')
    
    # Domain prediction accuracy per subject
    domain_accs = {}
    for sub in unique_subjects:
        mask = subject_feat == sub
        if mask.sum() > 0:
            sub_domain = subject_to_domain[sub]
            sub_mask = (subject_feat == sub)
            correct = (domains_pred[sub_mask] == domains_true[sub_mask]).sum()
            total = sub_mask.sum()
            domain_accs[sub] = correct / total if total > 0 else 0
    
    selected_domain_accs = [(sub, domain_accs[sub]) for sub in selected_subjects if sub in domain_accs]
    selected_domain_accs.sort(key=lambda x: x[1])
    sub_names_dom = [s[0] for s in selected_domain_accs]
    dom_accs = [s[1] for s in selected_domain_accs]
    
    axes2[1].barh(range(len(sub_names_dom)), dom_accs, alpha=0.7, color='orange')
    axes2[1].set_yticks(range(len(sub_names_dom)))
    axes2[1].set_yticklabels(sub_names_dom)
    axes2[1].set_xlabel('Domain Prediction Accuracy', fontsize=12)
    axes2[1].set_title('Domain Discriminator Performance\n(Lower = better bias reduction)', fontsize=12, fontweight='bold')
    axes2[1].axvline(x=1.0/len(unique_subjects), color='green', linestyle='--', alpha=0.5, label='Chance')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(report_dir / "bias_reduction.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Example covariance matrices
    fig3, axes3 = plt.subplots(len(bands), n_examples, figsize=(3*n_examples, 3*len(bands)))
    if n_examples == 1:
        axes3 = axes3.reshape(-1, 1)
    
    band_names = ['Theta (4-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (13-30 Hz)', 'Gamma (30-45 Hz)']
    
    for band_idx, (band_covs, band_name) in enumerate(zip(covs_examples, band_names)):
        for ex_idx in range(n_examples):
            ax = axes3[band_idx, ex_idx]
            cov_matrix = band_covs[ex_idx]
            im = ax.imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'{band_name}\n{"Pain" if y_examples[ex_idx] == 1 else "Baseline"}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
            if ex_idx == 0:
                ax.set_ylabel('Channel', fontsize=10)
            if band_idx == len(bands) - 1:
                ax.set_xlabel('Channel', fontsize=10)
    
    plt.suptitle('Example Covariance Matrices Across Frequency Bands', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(report_dir / "covariance_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Riemannian mean trajectories
    n_subjects_show = min(len(trajectories), args.n_subjects)
    fig4, axes4 = plt.subplots(n_subjects_show, len(bands), 
                              figsize=(4*len(bands), 3*n_subjects_show))
    if n_subjects_show == 1:
        axes4 = axes4.reshape(1, -1)
    
    for sub_idx, (sub, traj_data) in enumerate(list(trajectories.items())[:n_subjects_show]):
        for band_idx, band_name in enumerate(band_names):
            ax = axes4[sub_idx, band_idx]
            
            # Plot mean covariance for baseline and pain
            for class_label, label_name in [(0, 'Baseline'), (1, 'Pain')]:
                if class_label in traj_data:
                    cov_mean = traj_data[class_label][band_idx]
                    im = ax.imshow(cov_mean, cmap='RdBu_r', aspect='auto', alpha=0.7)
                    if class_label == 1:
                        plt.colorbar(im, ax=ax, fraction=0.046)
            
            if sub_idx == 0:
                ax.set_title(band_name, fontsize=10, fontweight='bold')
            if band_idx == 0:
                ax.set_ylabel(f'Subject {sub}\nChannel', fontsize=9)
            if sub_idx == n_subjects_show - 1:
                ax.set_xlabel('Channel', fontsize=9)
    
    plt.suptitle('Riemannian Mean Covariance Trajectories\n(Baseline vs Pain per Subject)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(report_dir / "riemannian_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary JSON
    summary = {
        'n_subjects_total': len(unique_subjects),
        'n_subjects_shown': len(selected_subjects),
        'selected_subjects': selected_subjects,
        'per_subject_accuracy': {k: v['accuracy'] for k, v in subject_accs.items()},
        'overall_mean_accuracy': float(np.mean(all_accs)),
        'overall_std_accuracy': float(np.std(all_accs)),
        'validation_mean_accuracy': float(np.mean(val_accs)),
        'validation_std_accuracy': float(np.std(val_accs)),
        'domain_prediction_accuracy': {k: float(v) for k, v in domain_accs.items()},
        'mean_domain_accuracy': float(np.mean(list(domain_accs.values())))
    }
    
    with open(report_dir / "subject_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[INFO] Analysis complete! Results saved to {report_dir}/")
    print(f"  - subject_performance.png")
    print(f"  - bias_reduction.png")
    print(f"  - covariance_matrices.png")
    print(f"  - riemannian_trajectories.png")
    print(f"  - subject_analysis_summary.json")


if __name__ == "__main__":
    main()

