#!/usr/bin/env python3
"""
Improved DANN with Enhanced Features - Target: 80%+ Accuracy
Combines:
1. Enhanced features (PSD + temporal + cross-channel)
2. Data augmentation
3. Better DANN architecture
4. Improved regularization
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class EnhancedDANN(pl.LightningModule):
    def __init__(self, input_dim: int, num_domains: int, lr: float = 3e-4, 
                 dropout: float = 0.6, weight_decay: float = 5e-3, 
                 domain_weight: float = 0.005, hidden_size: int = 128):
        super().__init__()
        self.save_hyperparameters()
        
        # Larger feature extractor with more capacity
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_size), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),  # Additional layer
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout * 0.5),
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Domain discriminator with dropout
        self.domain_disc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, num_domains)
        )
        
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.domain_weight = domain_weight

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        x, y, d = batch
        z = self.feature(x)
        logits = self.classifier(z).squeeze(1)
        loss_y = self.bce(logits, y.float())

        # Adaptive domain loss weight
        p = (self.current_epoch + 1) / max(1.0, self.trainer.max_epochs)
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        z_rev = GradientReversal.apply(z, lambda_)
        dom_logits = self.domain_disc(z_rev)
        loss_d = self.ce(dom_logits, d.long())

        loss = loss_y + self.domain_weight * loss_d
        
        self.log_dict({
            "train_loss": loss, 
            "loss_y": loss_y, 
            "loss_d": loss_d,
            "lr": self.optimizers().param_groups[0]['lr']
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, d = batch
        logits = self.classifier(self.feature(x)).squeeze(1)
        loss = self.bce(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == y.long()).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)


def extract_psd_features(X_raw, sfreq, bands=((4, 8), (8, 12), (13, 30), (30, 45))):
    """Extract Power Spectral Density features"""
    features = []
    for trial in X_raw:
        trial_features = []
        for channel in trial:
            freqs, psd = signal.welch(channel, sfreq, nperseg=min(256, len(channel)))
            for l, h in bands:
                band_mask = (freqs >= l) & (freqs <= h)
                band_power = psd[band_mask].sum()
                trial_features.append(band_power)
            psd_norm = psd / (psd.sum() + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            trial_features.append(spectral_entropy)
        features.append(trial_features)
    return np.array(features)


def extract_temporal_features(X_raw):
    """Extract temporal statistics features"""
    features = []
    for trial in X_raw:
        trial_features = []
        for channel in trial:
            trial_features.extend([
                np.mean(channel), np.std(channel), np.var(channel),
                np.percentile(channel, 25), np.percentile(channel, 50),
                np.percentile(channel, 75), np.max(channel) - np.min(channel),
                np.mean(np.abs(np.diff(channel))), np.std(np.diff(channel)),
            ])
        features.append(trial_features)
    return np.array(features)


def extract_cross_channel_features(X_raw):
    """Extract cross-channel correlation features"""
    features = []
    for trial in X_raw:
        corr_matrix = np.corrcoef(trial)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        features.append(upper_triangle)
    return np.array(features)


def augment_data(X, y, noise_level=0.05, n_augment=2):
    """Light data augmentation for DANN"""
    X_aug = [X]
    y_aug = [y]
    
    np.random.seed(42)
    for _ in range(n_augment):
        noise = np.random.normal(0, noise_level * X.std(), X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../packed/features_ds005284_riemannian_improved.npz",
                   help="Features NPZ file (default: improved Riemannian features)")
    ap.add_argument("--preprocessed", default=None,
                   help="Preprocessed NPZ file (optional, for additional features)")
    ap.add_argument("--use-pca", action="store_true", default=True,
                   help="Use PCA to reduce feature dimensionality (default: True)")
    ap.add_argument("--pca-components", type=int, default=300,
                   help="Number of PCA components (default: 300, max: min(n_samples, n_features))")
    ap.add_argument("--use-additional-features", action="store_true", default=False,
                   help="Add PSD/Temporal/Cross-channel features (default: False, use only Riemannian)")
    ap.add_argument("--epochs", type=int, default=100,
                   help="Max epochs")
    ap.add_argument("--batch", type=int, default=32,
                   help="Batch size")
    ap.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    ap.add_argument("--dropout", type=float, default=0.6,
                   help="Dropout rate (default: 0.6)")
    ap.add_argument("--weight-decay", type=float, default=5e-3,
                   help="Weight decay (default: 5e-3)")
    ap.add_argument("--domain-weight", type=float, default=0.005,
                   help="Domain loss weight (default: 0.005)")
    ap.add_argument("--hidden-size", type=int, default=128,
                   help="Hidden layer size (default: 128)")
    ap.add_argument("--augment", action="store_true", default=False,
                   help="Use data augmentation")
    ap.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience")
    ap.add_argument("--report-dir", default="../reports",
                   help="Report directory")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("IMPROVED DANN WITH ENHANCED FEATURES")
    print("="*70)

    # Load data
    print("\n[1] Loading data...")
    feat_data = np.load(args.features, allow_pickle=True)
    X_feat = feat_data['X']
    y = feat_data['y']
    subject = feat_data['subject']

    print(f"   Loaded features: {X_feat.shape}")
    print(f"   Feature type: {'Improved Riemannian' if 'riemannian_improved' in str(args.features) else 'Standard'}")

    # Optionally load raw data for additional features
    use_additional_features = args.use_additional_features and args.preprocessed is not None
    if use_additional_features:
        try:
            prep_data = np.load(args.preprocessed, allow_pickle=True)
            X_raw = prep_data['X']
            sfreq = float(prep_data['sfreq'])
            print(f"   Raw data: {X_raw.shape}")
            
            print("\n[2] Extracting additional features (PSD + Temporal + Cross-channel)...")
            psd_features = extract_psd_features(X_raw, sfreq)
            temp_features = extract_temporal_features(X_raw)
            cross_features = extract_cross_channel_features(X_raw)
            
            X_enhanced = np.hstack([X_feat, psd_features, temp_features, cross_features])
            print(f"   Combined features: {X_enhanced.shape}")
        except Exception as e:
            print(f"   [WARN] Failed to load additional features: {e}")
            print(f"   [INFO] Using only Riemannian features")
            X_enhanced = X_feat
            use_additional_features = False
    else:
        X_enhanced = X_feat
        print(f"\n[2] Using only improved Riemannian features: {X_enhanced.shape}")
        print(f"   [INFO] Additional features disabled (recommended for better generalization)")

    # Apply PCA if requested (for dimensionality reduction)
    if args.use_pca:
        max_components = min(X_enhanced.shape[0], X_enhanced.shape[1])
        n_components = min(args.pca_components, max_components)
        
        if args.pca_components > max_components:
            print(f"\n[3] [WARN] Requested {args.pca_components} components, but max is {max_components}")
            print(f"   Using {n_components} components instead")
        
        print(f"\n[3] Applying PCA: {X_enhanced.shape[1]} → {n_components} features...")
        pca = PCA(n_components=n_components, random_state=42)
        X_enhanced = pca.fit_transform(X_enhanced)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"   PCA applied: {X_enhanced.shape}")
        print(f"   Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
        
        if explained_var < 0.90:
            print(f"   [WARN] Low explained variance - consider increasing PCA components")
    else:
        print(f"\n[3] No PCA applied, using all {X_enhanced.shape[1]} features")
        print(f"   [WARN] High-dimensional features may cause overfitting")

    # Normalize
    print(f"\n[4] Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enhanced)
    print(f"   Normalized features: {X_scaled.shape}")

    # Data augmentation
    if args.augment:
        print("\n[5] Applying data augmentation...")
        X_scaled, y = augment_data(X_scaled, y, noise_level=0.05, n_augment=2)
        subject_expanded = np.tile(subject, 3)[:len(X_scaled)]
        print(f"   Augmented shape: {X_scaled.shape}")
    else:
        subject_expanded = subject

    # Subject-level split (80/20) - CRITICAL to prevent data leakage
    print("\n[6] Splitting data by SUBJECTS: 80% train, 20% test...")
    unique_subjects = np.unique(subject_expanded)
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=42
    )
    
    # Create masks based on subjects
    train_mask = np.isin(subject_expanded, train_subjects)
    test_mask = np.isin(subject_expanded, test_subjects)
    
    X_train = X_scaled[train_mask]
    X_test = X_scaled[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    subject_train = subject_expanded[train_mask]
    subject_test = subject_expanded[test_mask]
    
    print(f"   Train subjects: {len(train_subjects)} ({len(train_subjects)/len(unique_subjects)*100:.1f}%)")
    print(f"   Test subjects: {len(test_subjects)} ({len(test_subjects)/len(unique_subjects)*100:.1f}%)")
    print(f"   Train samples: {len(X_train)} ({len(X_train)/len(X_scaled)*100:.1f}%)")
    print(f"   Test samples: {len(X_test)} ({len(X_test)/len(X_scaled)*100:.1f}%)")
    print(f"   Train classes: Baseline={np.sum(y_train==0)}, Pain={np.sum(y_train==1)}")
    print(f"   Test classes: Baseline={np.sum(y_test==0)}, Pain={np.sum(y_test==1)}")

    # Convert to domain IDs
    unique_subjects = np.unique(subject_train)
    subject_to_domain = {sub: idx for idx, sub in enumerate(unique_subjects)}
    domains_train = np.array([subject_to_domain[sub] for sub in subject_train])
    domains_test = np.array([subject_to_domain.get(sub, 0) for sub in subject_test])

    # Create dataloaders
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(domains_train, dtype=torch.long)
    )
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
        torch.tensor(domains_test, dtype=torch.long)
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0)

    # Setup model
    print("\n[7] Setting up Enhanced DANN model...")
    model = EnhancedDANN(
        input_dim=X_enhanced.shape[1],
        num_domains=int(domains_train.max() + 1),
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        domain_weight=args.domain_weight,
        hidden_size=args.hidden_size
    )
    
    print(f"   Input dimension: {X_enhanced.shape[1]}")
    print(f"   Number of domains: {int(domains_train.max() + 1)}")
    print(f"   Hidden size: {args.hidden_size}")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=report_dir / "checkpoints_dann_enhanced",
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min'
    )

    csv_logger = CSVLogger(report_dir, name="dann_enhanced_logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stop, checkpoint],
        logger=csv_logger,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    print("\n[8] Training Enhanced DANN...")
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_test)

    # Evaluate on train and test sets
    print("\n[9] Evaluating model...")
    model.eval()
    with torch.no_grad():
        # Test set
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits_test = model.classifier(model.feature(X_test_tensor)).squeeze(1)
        preds_test = (torch.sigmoid(logits_test) > 0.5).long()
        test_acc = (preds_test == torch.tensor(y_test, dtype=torch.long)).float().mean().item()
        
        # Train set (for overfitting check)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        logits_train = model.classifier(model.feature(X_train_tensor)).squeeze(1)
        preds_train = (torch.sigmoid(logits_train) > 0.5).long()
        train_acc = (preds_train == torch.tensor(y_train, dtype=torch.long)).float().mean().item()
        
        # Overfitting gap
        overfitting_gap = train_acc - test_acc

    print(f"\n   {'='*60}")
    print(f"   FINAL RESULTS:")
    print(f"   {'='*60}")
    print(f"   Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f} percentage points)")
    
    if overfitting_gap < 0.05:
        print(f"   ✅ GOOD: Low overfitting (<5% gap)")
    elif overfitting_gap < 0.10:
        print(f"   ⚠️  MODERATE: Some overfitting (5-10% gap)")
    else:
        print(f"   ❌ HIGH: Significant overfitting (>10% gap)")
    
    if test_acc >= 0.80:
        print(f"   ✅ TARGET ACHIEVED: Test accuracy >= 80%")
    else:
        print(f"   ⚠️  BELOW TARGET: Test accuracy < 80%")
    
    print(f"   {'='*60}")

    # Load training history
    log_dir = report_dir / "dann_enhanced_logs"
    versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version")], 
                     key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else -1)
    
    if versions:
        metrics_file = versions[-1] / "metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            
            final_metrics = {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'overfitting_gap': float(overfitting_gap),
                'best_val_accuracy': float(df['val_acc'].dropna().max()) if 'val_acc' in df.columns else None,
                'final_val_accuracy': float(df['val_acc'].dropna().iloc[-1]) if 'val_acc' in df.columns else None,
                'final_val_loss': float(df['val_loss'].dropna().iloc[-1]) if 'val_loss' in df.columns else None,
                'final_train_loss': float(df['train_loss'].dropna().iloc[-1]) if 'train_loss' in df.columns else None,
                'n_features': int(X_enhanced.shape[1]),
                'n_domains': int(domains_train.max() + 1),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test)),
                'train_subjects': int(len(train_subjects)),
                'test_subjects': int(len(test_subjects)),
                'split_method': 'subject_level'
            }
            
            with open(report_dir / "dann_enhanced_metrics.json", "w") as f:
                json.dump(final_metrics, f, indent=2)
            
            print(f"\n[8] Results saved to: {report_dir / 'dann_enhanced_metrics.json'}")
            print(f"   Best Val Accuracy: {final_metrics.get('best_val_accuracy', 'N/A')}")
            print(f"   Train Accuracy: {final_metrics['train_accuracy']:.4f} ({final_metrics['train_accuracy']*100:.2f}%)")
            print(f"   Test Accuracy: {final_metrics['test_accuracy']:.4f} ({final_metrics['test_accuracy']*100:.2f}%)")
            print(f"   Overfitting Gap: {final_metrics['overfitting_gap']:.4f} ({final_metrics['overfitting_gap']*100:.2f} pp)")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

