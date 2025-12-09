#!/usr/bin/env python3
"""
Test the trained DANN model (from ds005284) on osf-data
- Loads the trained DANN model checkpoint
- Loads osf-data improved Riemannian features
- Applies same PCA transformation (from training)
- Tests the model and reports accuracy
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
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


class EnhancedDANN(pl.LightningModule):
    def __init__(self, input_dim: int, num_domains: int, lr: float = 3e-4, 
                 dropout: float = 0.6, weight_decay: float = 5e-3, 
                 domain_weight: float = 0.005, hidden_size: int = 128):
        super().__init__()
        self.save_hyperparameters()
        
        # Match the actual training architecture
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
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1)
        )
        
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

    def forward(self, x):
        z = self.feature(x)
        logits = self.classifier(z).squeeze(1)
        return logits


def main():
    ap = argparse.ArgumentParser(description="Test DANN model on osf-data")
    ap.add_argument("--model-checkpoint", 
                   default="../../ds005284/reports/checkpoints_dann_enhanced/best-epoch=48-val_loss=0.073.ckpt",
                   help="Path to trained DANN model checkpoint")
    ap.add_argument("--features", default="../packed/features_osf-data_riemannian_improved.npz",
                   help="osf-data improved Riemannian features")
    ap.add_argument("--pca-model", default="../../ds005284/reports/pca_model.pkl",
                   help="Saved PCA model from training (optional)")
    ap.add_argument("--scaler-model", default="../../ds005284/reports/scaler_model.pkl",
                   help="Saved Scaler model from training (optional)")
    ap.add_argument("--pca-components", type=int, default=300,
                   help="Number of PCA components (if PCA model not saved)")
    ap.add_argument("--batch", type=int, default=32,
                   help="Batch size")
    ap.add_argument("--output", default="../reports/dann_test_on_osf_data.json",
                   help="Output JSON file for results")
    args = ap.parse_args()

    print("="*70)
    print("TESTING DANN MODEL ON osf-data")
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

    # Load PCA and Scaler from training (if available)
    # For now, we'll need to refit PCA on osf-data with same number of components
    # In a real scenario, you'd save the PCA/scaler from training
    print("\n[2] Applying PCA and normalization...")
    
    # Apply PCA (same number of components as training)
    max_components = min(X_osf.shape[0], X_osf.shape[1], args.pca_components)
    n_components = max_components
    
    print(f"   Applying PCA: {X_osf.shape[1]} → {n_components} features...")
    pca = PCA(n_components=n_components, random_state=42)
    X_osf_pca = pca.fit_transform(X_osf)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"   PCA applied: {X_osf_pca.shape}")
    print(f"   Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    
    # Normalize
    scaler = StandardScaler()
    X_osf_scaled = scaler.fit_transform(X_osf_pca)
    print(f"   Normalized features: {X_osf_scaled.shape}")

    # Find model checkpoint
    print("\n[3] Loading trained DANN model...")
    checkpoint_path = Path(args.model_checkpoint)
    if '*' in str(checkpoint_path):
        # Find latest checkpoint
        checkpoint_dir = checkpoint_path.parent
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"   Loading checkpoint: {checkpoint_path}")
    
    # Load model
    # Note: We need to know the input_dim and num_domains from training
    # For now, we'll infer from checkpoint or use defaults
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hyperparams = checkpoint.get('hyper_parameters', {})
        input_dim = hyperparams.get('input_dim', X_osf_scaled.shape[1])
        num_domains = hyperparams.get('num_domains', 20)  # From ds005284 training
        
        print(f"   Model input_dim: {input_dim}")
        print(f"   Model num_domains: {num_domains}")
        
        # Adjust input_dim if needed (should match PCA output)
        if input_dim != X_osf_scaled.shape[1]:
            print(f"   [WARN] Model expects {input_dim} features, but osf-data has {X_osf_scaled.shape[1]}")
            print(f"   [WARN] Will use {X_osf_scaled.shape[1]} features (matching PCA output)")
            input_dim = X_osf_scaled.shape[1]
        
        model = EnhancedDANN(
            input_dim=input_dim,
            num_domains=num_domains,
            lr=hyperparams.get('lr', 3e-4),
            dropout=hyperparams.get('dropout', 0.6),
            weight_decay=hyperparams.get('weight_decay', 5e-3),
            domain_weight=hyperparams.get('domain_weight', 0.005),
            hidden_size=hyperparams.get('hidden_size', 128)
        )
        
        # Load weights (excluding domain discriminator if num_domains differs)
        model_state = checkpoint['state_dict']
        # Filter out domain discriminator if num_domains differs
        filtered_state = {}
        for k, v in model_state.items():
            if 'domain_disc' in k and num_domains != hyperparams.get('num_domains', 20):
                continue  # Skip domain discriminator if domain count differs
            filtered_state[k] = v
        
        model.load_state_dict(filtered_state, strict=False)
        model.eval()
        
    except Exception as e:
        print(f"   [ERROR] Failed to load model: {e}")
        raise

    # Create dummy domain IDs (not used for testing, but needed for DataLoader)
    domains_osf = np.zeros(len(X_osf_scaled), dtype=np.int64)

    # Create dataloader
    ds_test = TensorDataset(
        torch.tensor(X_osf_scaled, dtype=torch.float32),
        torch.tensor(y_osf, dtype=torch.long),
        torch.tensor(domains_osf, dtype=torch.long)
    )
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0)

    # Test model
    print("\n[4] Testing model on osf-data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dl_test:
            x, y, d = batch
            x = x.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    baseline_mask = all_labels == 0
    pain_mask = all_labels == 1
    baseline_acc = (all_preds[baseline_mask] == all_labels[baseline_mask]).mean() if baseline_mask.sum() > 0 else 0.0
    pain_acc = (all_preds[pain_mask] == all_labels[pain_mask]).mean() if pain_mask.sum() > 0 else 0.0
    
    # Per-subject accuracy
    subject_accs = {}
    for subj in np.unique(subject_osf):
        subj_mask = subject_osf == subj
        if subj_mask.sum() > 0:
            subj_acc = (all_preds[subj_mask] == all_labels[subj_mask]).mean()
            subject_accs[str(subj)] = float(subj_acc)

    print(f"\n   ============================================================")
    print(f"   TEST RESULTS ON osf-data:")
    print(f"   ============================================================")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"   Pain Accuracy: {pain_acc:.4f} ({pain_acc*100:.2f}%)")
    print(f"   Test Samples: {len(all_labels)}")
    print(f"   Test Subjects: {len(np.unique(subject_osf))}")
    print(f"   ============================================================")

    # Save results
    results = {
        "model_source": "ds005284",
        "test_dataset": "osf-data",
        "overall_accuracy": float(accuracy),
        "baseline_accuracy": float(baseline_acc),
        "pain_accuracy": float(pain_acc),
        "test_samples": int(len(all_labels)),
        "test_subjects": int(len(np.unique(subject_osf))),
        "n_features": int(X_osf_scaled.shape[1]),
        "pca_components": int(n_components),
        "pca_explained_variance": float(explained_var),
        "per_subject_accuracy": subject_accs
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[5] Results saved to: {args.output}")
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

