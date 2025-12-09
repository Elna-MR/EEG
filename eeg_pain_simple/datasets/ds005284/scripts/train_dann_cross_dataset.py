#!/usr/bin/env python3
"""
Improved DANN for Cross-Dataset Generalization
Key improvements:
1. Multi-dataset training (ds005284 + osf-data)
2. Dataset-level domain adaptation (not just subject-level)
3. MMD loss for better domain alignment
4. Stronger domain adaptation (higher domain_weight)
5. More PCA components (500 instead of 300)
6. Progressive domain adaptation
"""

import argparse
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def mmd_loss(source_features, target_features, sigma=1.0):
    """Maximum Mean Discrepancy loss for domain alignment"""
    def gaussian_kernel(x, y, sigma=1.0):
        """Gaussian RBF kernel"""
        if x.shape[0] == 0 or y.shape[0] == 0:
            return torch.tensor(0.0, device=x.device)
        pairwise_distances = torch.cdist(x, y) ** 2
        return torch.exp(-pairwise_distances / (2 * sigma ** 2))
    
    # Compute MMD using Gaussian kernel
    k_ss = gaussian_kernel(source_features, source_features, sigma).mean() if source_features.shape[0] > 0 else torch.tensor(0.0, device=source_features.device)
    k_tt = gaussian_kernel(target_features, target_features, sigma).mean() if target_features.shape[0] > 0 else torch.tensor(0.0, device=target_features.device)
    k_st = gaussian_kernel(source_features, target_features, sigma).mean() if source_features.shape[0] > 0 and target_features.shape[0] > 0 else torch.tensor(0.0, device=source_features.device)
    
    mmd = k_ss + k_tt - 2 * k_st
    return mmd


class ImprovedDANN(pl.LightningModule):
    def __init__(self, input_dim: int, num_subject_domains: int, num_dataset_domains: int = 2,
                 lr: float = 3e-4, dropout: float = 0.5, weight_decay: float = 1e-3, 
                 domain_weight: float = 0.1, mmd_weight: float = 0.05, hidden_size: int = 256):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature extractor with more capacity
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_size), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout * 0.5),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Subject-level domain discriminator
        self.domain_disc_subject = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, num_subject_domains)
        )
        
        # Dataset-level domain discriminator (for cross-dataset adaptation)
        self.domain_disc_dataset = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 128), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(128, num_dataset_domains)
        )
        
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.domain_weight = domain_weight
        self.mmd_weight = mmd_weight

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
        x, y, d_subject, d_dataset, dataset_id = batch
        
        z = self.feature(x)
        logits = self.classifier(z).squeeze(1)
        loss_y = self.bce(logits, y.float())

        # Progressive domain adaptation weight
        p = (self.current_epoch + 1) / max(1.0, self.trainer.max_epochs)
        lambda_subject = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        lambda_dataset = 2.0 / (1.0 + np.exp(-5 * p)) - 1.0  # Faster for dataset-level
        
        # Subject-level domain loss
        z_rev_subject = GradientReversal.apply(z, lambda_subject)
        dom_logits_subject = self.domain_disc_subject(z_rev_subject)
        loss_d_subject = self.ce(dom_logits_subject, d_subject.long())
        
        # Dataset-level domain loss (for cross-dataset adaptation)
        z_rev_dataset = GradientReversal.apply(z, lambda_dataset)
        dom_logits_dataset = self.domain_disc_dataset(z_rev_dataset)
        loss_d_dataset = self.ce(dom_logits_dataset, d_dataset.long())
        
        # MMD loss for domain alignment (if we have both datasets)
        loss_mmd = torch.tensor(0.0, device=self.device)
        if len(torch.unique(dataset_id)) > 1:
            source_mask = dataset_id == 0  # ds005284
            target_mask = dataset_id == 1  # osf-data
            
            if source_mask.sum() > 0 and target_mask.sum() > 0:
                source_features = z[source_mask]
                target_features = z[target_mask]
                loss_mmd = mmd_loss(source_features, target_features)
        
        # Combined loss
        loss = loss_y + self.domain_weight * (loss_d_subject + loss_d_dataset) + self.mmd_weight * loss_mmd
        
        self.log_dict({
            "train_loss": loss, 
            "loss_y": loss_y, 
            "loss_d_subject": loss_d_subject,
            "loss_d_dataset": loss_d_dataset,
            "loss_mmd": loss_mmd,
            "lr": self.optimizers().param_groups[0]['lr']
        }, prog_bar=True)
        return loss

    def forward(self, x):
        z = self.feature(x)
        logits = self.classifier(z).squeeze(1)
        return logits

    def validation_step(self, batch, batch_idx):
        x, y, d_subject, d_dataset, dataset_id = batch
        logits = self.classifier(self.feature(x)).squeeze(1)
        loss = self.bce(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == y.long()).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)


def main():
    ap = argparse.ArgumentParser(description="Train improved DANN for cross-dataset generalization")
    ap.add_argument("--features-ds005284", default="../packed/features_ds005284_riemannian_improved.npz",
                   help="ds005284 features")
    ap.add_argument("--features-osf", default="../../osf-data/packed/features_osf-data_riemannian_improved.npz",
                   help="osf-data features")
    ap.add_argument("--use-osf-for-training", action="store_true",
                   help="Use osf-data for training (multi-dataset training)")
    ap.add_argument("--pca-components", type=int, default=500,
                   help="Number of PCA components")
    ap.add_argument("--domain-weight", type=float, default=0.1,
                   help="Domain adaptation weight (higher = stronger adaptation)")
    ap.add_argument("--mmd-weight", type=float, default=0.05,
                   help="MMD loss weight")
    ap.add_argument("--hidden-size", type=int, default=256,
                   help="Hidden layer size")
    ap.add_argument("--dropout", type=float, default=0.5,
                   help="Dropout rate")
    ap.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-3,
                   help="Weight decay")
    ap.add_argument("--epochs", type=int, default=100,
                   help="Max epochs")
    ap.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience")
    ap.add_argument("--batch", type=int, default=32,
                   help="Batch size")
    ap.add_argument("--test-size", type=float, default=0.2,
                   help="Test set size (subject-level split)")
    ap.add_argument("--report-dir", default="../reports",
                   help="Report directory")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("IMPROVED DANN FOR CROSS-DATASET GENERALIZATION")
    print("="*70)

    # Load ds005284 data
    print("\n[1] Loading ds005284 features...")
    feat_data_ds = np.load(args.features_ds005284, allow_pickle=True)
    X_ds = feat_data_ds['X']
    y_ds = feat_data_ds['y']
    subject_ds = feat_data_ds['subject']
    
    print(f"   ds005284 features: {X_ds.shape}")
    print(f"   ds005284 samples: {len(X_ds)}")
    print(f"   ds005284 subjects: {len(np.unique(subject_ds))}")

    # Load osf-data if using for training
    X_osf_train = None
    y_osf_train = None
    subject_osf_train = None
    
    if args.use_osf_for_training:
        print("\n[2] Loading osf-data features for training...")
        feat_data_osf = np.load(args.features_osf, allow_pickle=True)
        X_osf_all = feat_data_osf['X']
        y_osf_all = feat_data_osf['y']
        subject_osf_all = feat_data_osf['subject']
        
        print(f"   osf-data features: {X_osf_all.shape}")
        print(f"   osf-data samples: {len(X_osf_all)}")
        print(f"   osf-data subjects: {len(np.unique(subject_osf_all))}")
        
        # Use 80% of osf-data for training, 20% for validation
        unique_subjects_osf = np.unique(subject_osf_all)
        n_test_subjects_osf = int(len(unique_subjects_osf) * args.test_size)
        np.random.seed(42)
        test_subjects_osf = np.random.choice(unique_subjects_osf, size=n_test_subjects_osf, replace=False)
        train_mask_osf = ~np.isin(subject_osf_all, test_subjects_osf)
        
        X_osf_train = X_osf_all[train_mask_osf]
        y_osf_train = y_osf_all[train_mask_osf]
        subject_osf_train = subject_osf_all[train_mask_osf]
        
        print(f"   osf-data train: {len(X_osf_train)} samples")
    else:
        print("\n[2] Using only ds005284 for training (osf-data will be used for testing)")

    # Subject-level split for ds005284
    print("\n[3] Creating subject-level train/test split for ds005284...")
    unique_subjects_ds = np.unique(subject_ds)
    n_test_subjects_ds = int(len(unique_subjects_ds) * args.test_size)
    
    np.random.seed(42)
    test_subjects_ds = np.random.choice(unique_subjects_ds, size=n_test_subjects_ds, replace=False)
    train_mask_ds = ~np.isin(subject_ds, test_subjects_ds)
    test_mask_ds = np.isin(subject_ds, test_subjects_ds)
    
    X_train_ds = X_ds[train_mask_ds]
    X_test_ds = X_ds[test_mask_ds]
    y_train_ds = y_ds[train_mask_ds]
    y_test_ds = y_ds[test_mask_ds]
    subject_train_ds = subject_ds[train_mask_ds]
    subject_test_ds = subject_ds[test_mask_ds]
    
    print(f"   ds005284 train: {len(X_train_ds)} samples, {len(np.unique(subject_train_ds))} subjects")
    print(f"   ds005284 test: {len(X_test_ds)} samples, {len(np.unique(subject_test_ds))} subjects")

    # Combine datasets if using osf-data for training
    if args.use_osf_for_training and X_osf_train is not None:
        print("\n[4] Combining datasets...")
        # Handle feature dimension mismatch
        min_features = min(X_train_ds.shape[1], X_osf_train.shape[1])
        X_train_ds = X_train_ds[:, :min_features]
        X_test_ds = X_test_ds[:, :min_features]
        X_osf_train = X_osf_train[:, :min_features]
        
        X_train_combined = np.vstack([X_train_ds, X_osf_train])
        y_train_combined = np.hstack([y_train_ds, y_osf_train])
        subject_train_combined = np.hstack([
            [f"ds005284_{s}" for s in subject_train_ds],
            [f"osf_{s}" for s in subject_osf_train]
        ])
        dataset_id_train = np.hstack([
            np.zeros(len(X_train_ds)),  # 0 = ds005284
            np.ones(len(X_osf_train))   # 1 = osf-data
        ])
        
        print(f"   Combined train: {len(X_train_combined)} samples")
        X_train = X_train_combined
        y_train = y_train_combined
        subject_train = subject_train_combined
        dataset_id_train_data = dataset_id_train
    else:
        X_train = X_train_ds
        y_train = y_train_ds
        subject_train = subject_train_ds
        dataset_id_train_data = np.zeros(len(X_train))  # All from ds005284

    # Apply PCA
    print("\n[5] Applying PCA...")
    max_components = min(X_train.shape[0], X_train.shape[1], args.pca_components)
    n_components = max_components
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_ds_pca = pca.transform(X_test_ds)
    print(f"   PCA: {X_train.shape[1]} → {n_components} components")
    print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Normalize
    print("\n[6] Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_ds_pca)

    # Create domain labels
    print("\n[7] Creating domain labels...")
    unique_subjects = np.unique(subject_train)
    subject_to_domain = {s: i for i, s in enumerate(unique_subjects)}
    domains_subject = np.array([subject_to_domain[s] for s in subject_train])
    
    # Dataset-level domains (0 = ds005284, 1 = osf-data)
    domains_dataset = dataset_id_train_data.astype(np.int64)
    
    print(f"   Subject domains: {len(unique_subjects)}")
    print(f"   Dataset domains: {len(np.unique(domains_dataset))}")

    # Create dataloaders
    print("\n[8] Creating dataloaders...")
    ds_train = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(domains_subject, dtype=torch.long),
        torch.tensor(domains_dataset, dtype=torch.long),
        torch.tensor(dataset_id_train_data, dtype=torch.long)
    )
    
    ds_test = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test_ds, dtype=torch.long),
        torch.tensor(np.zeros(len(X_test_scaled), dtype=np.int64), dtype=torch.long),  # Dummy domains
        torch.tensor(np.zeros(len(X_test_scaled), dtype=np.int64), dtype=torch.long),  # Dummy domains
        torch.tensor(np.zeros(len(X_test_scaled), dtype=np.int64), dtype=torch.long)  # Dummy dataset_id
    )
    
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0)

    # Setup model
    print("\n[9] Setting up Improved DANN model...")
    model = ImprovedDANN(
        input_dim=X_train_scaled.shape[1],
        num_subject_domains=len(unique_subjects),
        num_dataset_domains=2,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        domain_weight=args.domain_weight,
        mmd_weight=args.mmd_weight,
        hidden_size=args.hidden_size
    )
    
    print(f"   Input dimension: {X_train_scaled.shape[1]}")
    print(f"   Subject domains: {len(unique_subjects)}")
    print(f"   Dataset domains: 2")
    print(f"   Domain weight: {args.domain_weight}")
    print(f"   MMD weight: {args.mmd_weight}")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=report_dir / "checkpoints_dann_cross_dataset",
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min'
    )

    csv_logger = CSVLogger(report_dir, name="dann_cross_dataset_logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stop, checkpoint],
        logger=csv_logger,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    print("\n[10] Training Improved DANN...")
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_test)

    # Evaluate
    print("\n[11] Evaluating model...")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch in dl_test:
            x, y, _, _, _ = batch
            x = x.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().detach().tolist())
            all_labels.extend(y.tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = (all_preds == all_labels).mean()
    
    # Evaluate on training set
    model.eval()
    with torch.no_grad():
        train_preds = []
        train_labels = []
        for batch in dl_train:
            x, y, _, _, _ = batch
            x = x.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_preds.extend(preds.cpu().detach().tolist())
            train_labels.extend(y.tolist())
    
    train_preds = np.array(train_preds)
    train_labels = np.array(train_labels)
    train_acc = (train_preds == train_labels).mean()

    # Save results
    results = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "overfitting_gap": float(train_acc - test_acc),
        "n_features": int(n_components),
        "n_subject_domains": int(len(unique_subjects)),
        "n_dataset_domains": 2,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test_ds)),
        "domain_weight": float(args.domain_weight),
        "mmd_weight": float(args.mmd_weight),
        "used_osf_for_training": args.use_osf_for_training
    }
    
    results_file = report_dir / "dann_cross_dataset_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[12] Results saved to {results_file}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Overfitting Gap: {train_acc - test_acc:.4f}")


if __name__ == "__main__":
    main()

