#!/usr/bin/env python3
"""
DANN Training for cpCGX dataset
Uses PyTorch with domain adaptation across subjects
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DANN(nn.Module):
    def __init__(self, input_dim: int, num_domains: int, lr: float = 1e-3):
        super().__init__()
        hidden = 256
        
        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
        )
        
        # Task classifier (EO vs EC)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Domain discriminator (subject identification)
        self.domain_disc = nn.Sequential(
            nn.Linear(hidden, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, num_domains)
        )
        
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, x, lambda_=1.0):
        z = self.feature(x)
        logits = self.classifier(z).squeeze(1)
        
        # Gradient reversal for domain adaptation
        z_rev = GradientReversal.apply(z, lambda_)
        dom_logits = self.domain_disc(z_rev)
        
        return logits, dom_logits


def train_epoch(model, train_loader, optimizer, epoch, max_epochs, device):
    model.train()
    total_loss = 0
    total_loss_y = 0
    total_loss_d = 0
    correct = 0
    total = 0
    
    # Adaptive lambda for gradient reversal
    p = (epoch + 1) / max_epochs
    lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    
    for batch_idx, (x, y, d) in enumerate(train_loader):
        x, y, d = x.to(device), y.to(device), d.to(device)
        
        optimizer.zero_grad()
        
        logits, dom_logits = model(x, lambda_)
        
        # Classification loss
        loss_y = model.bce(logits, y.float())
        
        # Domain loss
        loss_d = model.ce(dom_logits, d.long())
        
        # Total loss
        loss = loss_y + loss_d
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_y += loss_y.item()
        total_loss_d += loss_d.item()
        
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)
    
    return {
        'loss': total_loss / len(train_loader),
        'loss_y': total_loss_y / len(train_loader),
        'loss_d': total_loss_d / len(train_loader),
        'acc': correct / total
    }


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y, d in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x, lambda_=0)  # No gradient reversal for validation
            
            loss = model.bce(logits, y.float())
            total_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)
    
    return {
        'loss': total_loss / len(val_loader),
        'acc': correct / total
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="NPZ with X,y,subject")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--report-dir", default="reports", help="Directory to save reports")
    args = ap.parse_args()
    
    # Load features
    data = np.load(args.features, allow_pickle=True)
    X, y, subject = data['X'], data['y'], data['subject']
    
    print(f"\n{'='*70}")
    print(f"[INFO] Loaded features: X.shape={X.shape}, y.shape={y.shape}")
    print(f"[INFO] Unique subjects: {len(np.unique(subject))}")
    print(f"{'='*70}\n")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Domain = subject ID
    le = LabelEncoder()
    domains = le.fit_transform(subject)
    
    # Split by subjects (not random - to test cross-subject generalization)
    unique_subjects = np.unique(subject)
    
    if len(unique_subjects) == 1:
        # Only one subject - split data randomly instead
        print(f"[WARN] Only 1 subject found. Splitting data randomly (not by subject).")
        print(f"[WARN] This limits cross-subject generalization evaluation.")
        train_idx, val_idx = train_test_split(np.arange(len(X_scaled)), test_size=0.2, random_state=42)
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        d_train, d_val = domains[train_idx], domains[val_idx]
        
        print(f"[INFO] Train: {len(X_train)} samples")
        print(f"[INFO] Val: {len(X_val)} samples")
    else:
        # Multiple subjects - split by subject
        train_subjs, val_subjs = train_test_split(unique_subjects, test_size=0.2, random_state=42)
        
        train_mask = np.isin(subject, train_subjs)
        val_mask = np.isin(subject, val_subjs)
        
        X_train, X_val = X_scaled[train_mask], X_scaled[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        d_train, d_val = domains[train_mask], domains[val_mask]
        
        print(f"[INFO] Train: {len(X_train)} samples from {len(train_subjs)} subjects")
        print(f"[INFO] Val: {len(X_val)} samples from {len(val_subjs)} subjects")
        print(f"[INFO] Train subjects: {sorted(train_subjs)[:5]}...")
        print(f"[INFO] Val subjects: {sorted(val_subjs)[:5]}...")
    
    # Create dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}\n")
    
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
    
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # Create model
    num_domains = len(np.unique(domains))
    
    # Check if we have multiple classes
    unique_labels = np.unique(y)
    if len(unique_labels) == 1:
        print(f"[WARN] Only one class found (label={unique_labels[0]}). Classification will not be meaningful.")
        print(f"[WARN] Consider processing more files with different task types (EO/EC).")
    
    if num_domains == 1:
        print(f"[WARN] Only one domain (subject). Domain adaptation will not be effective.")
        print(f"[WARN] DANN will still train but domain discriminator has no effect.")
    
    model = DANN(input_dim=X.shape[1], num_domains=max(num_domains, 2), lr=args.lr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"[INFO] Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, dl_train, optimizer, epoch, args.epochs, device)
        val_metrics = validate(model, dl_val, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['acc']:.4f}")
    
    # Save results
    final_metrics = {
        'final_val_accuracy': float(history['val_acc'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_train_accuracy': float(history['train_acc'][-1]),
        'final_train_loss': float(history['train_loss'][-1]),
        'best_val_accuracy': float(max(history['val_acc'])),
        'best_val_epoch': int(np.argmax(history['val_acc']))
    }
    
    with open(report_dir / "dann_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training and validation accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', alpha=0.7)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Loss components (from last epoch)
    axes[1, 0].bar(['Classification', 'Domain'], 
                   [history['train_loss'][-1], history['train_loss'][-1]], 
                   alpha=0.7)
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final metrics summary
    axes[1, 1].axis('off')
    metrics_text = f"""
    Final Validation Accuracy: {final_metrics['final_val_accuracy']:.4f}
    Best Validation Accuracy: {final_metrics['best_val_accuracy']:.4f}
    Best Epoch: {final_metrics['best_val_epoch'] + 1}
    Final Train Accuracy: {final_metrics['final_train_accuracy']:.4f}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(report_dir / "dann_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"[INFO] Training Complete!")
    print(f"[INFO] Final Validation Accuracy: {final_metrics['final_val_accuracy']:.4f}")
    print(f"[INFO] Best Validation Accuracy: {final_metrics['best_val_accuracy']:.4f}")
    print(f"[INFO] Results saved to: {report_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

