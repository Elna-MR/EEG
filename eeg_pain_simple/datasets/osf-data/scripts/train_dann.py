#!/usr/bin/env python3
"""
DANN Training for osf-data (same as ds005284):
- Uses PyTorch Lightning
- Domain-Adversarial Neural Network
- Cross-subject validation
- Same architecture and hyperparameters as ds005284 for comparison
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DANN(pl.LightningModule):
    def __init__(self, input_dim: int, num_domains: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        hidden = 256
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
        )
        self.classifier = nn.Sequential(nn.Linear(hidden, 1))
        self.domain_disc = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, num_domains)
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y, d = batch
        z = self.feature(x)
        logits = self.classifier(z).squeeze(1)
        loss_y = self.bce(logits, y.float())

        p = (self.current_epoch + 1) / max(1.0, self.trainer.max_epochs)
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        z_rev = GradientReversal.apply(z, lambda_)
        dom_logits = self.domain_disc(z_rev)
        loss_d = self.ce(dom_logits, d.long())

        # Weight domain loss lower to prevent it from dominating classification
        loss = loss_y + 0.1 * loss_d
        self.log_dict({"train_loss": loss, "loss_y": loss_y, "loss_d": loss_d}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, d = batch
        logits = self.classifier(self.feature(x)).squeeze(1)
        loss = self.bce(logits, y.float())
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == y.long()).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../packed/features_osf-data.npz", help="NPZ with X,y,subject (default: ../packed/features_osf-data.npz)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--report-dir", default="../reports", help="Directory to save reports and visualizations (default: ../reports)")
    args = ap.parse_args()

    # Create report directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    X, y, subject = data['X'], data['y'], data['subject']
    
    # Normalize features for better training
    print(f"[INFO] Feature stats before normalization: mean={X.mean():.6f}, std={X.std():.6f}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"[INFO] Feature stats after normalization: mean={X.mean():.6f}, std={X.std():.6f}")
    
    # Convert string subject IDs to numeric domain IDs
    unique_subjects = np.unique(subject)
    subject_to_domain = {sub: idx for idx, sub in enumerate(unique_subjects)}
    domains = np.array([subject_to_domain[sub] for sub in subject])
    
    unique_subjs = np.unique(subject)
    train_subjs, val_subjs = train_test_split(unique_subjs, test_size=0.2, random_state=42)
    
    train_mask = np.isin(subject, train_subjs)
    val_mask = np.isin(subject, val_subjs)
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    d_train, d_val = domains[train_mask], domains[val_mask]
    
    print(f"[INFO] Train: {len(X_train)} samples from {len(train_subjs)} subjects")
    print(f"[INFO] Val: {len(X_val)} samples from {len(val_subjs)} subjects")
    print(f"[INFO] Train subjects: {sorted(train_subjs)}")
    print(f"[INFO] Val subjects: {sorted(val_subjs)}")
    
    # Auto-detect best available device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        print(f"[INFO] Using Apple Silicon GPU (MPS)")
    else:
        accelerator = "cpu"
        devices = 1
        print(f"[INFO] Using CPU (consider Google Colab for free GPU)")
    
    # Create separate dataloaders for train and validation
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(d_train, dtype=torch.long)
    )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
        torch.tensor(d_val, dtype=torch.long)
    )
    
    # Optimize num_workers based on device (GPU can handle more workers)
    num_workers = 4 if accelerator == "gpu" else 0
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=num_workers, pin_memory=(accelerator == "gpu"))
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=num_workers, pin_memory=(accelerator == "gpu"))

    # Setup logger to save training metrics
    csv_logger = CSVLogger(report_dir, name="dann_logs")
    
    model = DANN(input_dim=X.shape[1], num_domains=int(domains.max() + 1), lr=args.lr)
    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        enable_checkpointing=False, 
        logger=csv_logger,
        log_every_n_steps=1,
        accelerator=accelerator,
        devices=devices
    )
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)
    
    # Load training history (find latest version)
    log_dir = report_dir / "dann_logs"
    versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version")], 
                     key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else -1)
    if not versions:
        print("[WARN] No training logs found")
        return
    log_dir = versions[-1]
    metrics_file = log_dir / "metrics.csv"
    
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        
        # Save final metrics
        final_metrics = {}
        if 'val_acc' in df.columns and df['val_acc'].notna().any():
            final_metrics['final_val_accuracy'] = float(df['val_acc'].dropna().iloc[-1])
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            final_metrics['final_val_loss'] = float(df['val_loss'].dropna().iloc[-1])
        if 'train_loss' in df.columns and df['train_loss'].notna().any():
            final_metrics['final_train_loss'] = float(df['train_loss'].dropna().iloc[-1])
        
        with open(report_dir / "dann_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training and Validation Loss
        if 'train_loss' in df.columns:
            train_losses = df['train_loss'].dropna()
            axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.7)
        if 'val_loss' in df.columns:
            val_losses = df['val_loss'].dropna()
            if len(val_losses) > 0:
                axes[0, 0].plot(val_losses.index, val_losses, label='Val Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Classification vs Domain Loss
        if 'loss_y' in df.columns and 'loss_d' in df.columns:
            loss_y = df['loss_y'].dropna()
            loss_d = df['loss_d'].dropna()
            axes[0, 1].plot(loss_y, label='Classification Loss', alpha=0.7)
            axes[0, 1].plot(loss_d, label='Domain Loss', alpha=0.7)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Component Losses')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Validation Accuracy
        if 'val_acc' in df.columns:
            val_accs = df['val_acc'].dropna()
            if len(val_accs) > 0:
                axes[1, 0].plot(val_accs.index, val_accs, label='Val Accuracy', color='green', linewidth=2)
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].set_title('Validation Accuracy')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim([0, 1])
        
        # 4. Loss Components Ratio
        if 'loss_y' in df.columns and 'loss_d' in df.columns:
            loss_y = df['loss_y'].dropna()
            loss_d = df['loss_d'].dropna()
            if len(loss_y) > 0 and len(loss_d) > 0:
                min_len = min(len(loss_y), len(loss_d))
                ratio = loss_y[:min_len].values / (loss_d[:min_len].values + 1e-8)
                axes[1, 1].plot(ratio, label='Loss Ratio (Y/D)', alpha=0.7, color='purple')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Ratio')
                axes[1, 1].set_title('Classification / Domain Loss Ratio')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(report_dir / "dann_training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[INFO] Results saved to {report_dir}/")
        print(f"  - dann_metrics.json")
        print(f"  - dann_training_curves.png")
        print(f"  - dann_logs/ (training history)")


if __name__ == "__main__":
    main()


