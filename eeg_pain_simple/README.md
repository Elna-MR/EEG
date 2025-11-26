# EEG Pain Analysis Pipeline

This repository contains preprocessing, feature extraction, and DANN training pipelines for EEG pain datasets.

## 📁 Dataset Structure

Each dataset is organized in its own folder under `datasets/`:

```
datasets/
├── cpcgx/              # cpCGX_BIDS (Chronic Pain Resting-State)
│   ├── data/           # Raw BrainVision files
│   ├── scripts/        # Preprocessing, feature extraction, DANN
│   ├── packed/         # Preprocessed epochs and features
│   └── reports/        # Training results and metrics
│
└── ds005284/           # ds005284 (Pain vs Baseline)
    ├── data/           # Raw BIDS data
    ├── scripts/        # Preprocessing, feature extraction, DANN
    ├── packed/         # Preprocessed epochs and features
    └── reports/        # Training results and metrics
```

## 🚀 Quick Start

### cpCGX_BIDS Dataset

```bash
cd datasets/cpcgx
./run_pipeline.sh
```

Or run step-by-step:
```bash
cd datasets/cpcgx

# Preprocess
python scripts/preprocess_cpcgx.py --root data --out packed/cpcgx_pain.npz

# Extract features
python scripts/extract_features.py --npz packed/cpcgx_pain.npz --out packed/features_cpcgx.npz

# Train DANN
python scripts/train_dann_cpcgx.py --features packed/features_cpcgx.npz --report-dir reports
```

### ds005284 Dataset

```bash
cd datasets/ds005284
./run_pipeline.sh
```

## 📊 Current Results

### cpCGX_BIDS
- **Subjects:** 74
- **Epochs:** 19,918 (balanced EO/EC)
- **DANN Accuracy:** 87.44% (best validation)

### ds005284
- See `datasets/ds005284/reports/` for results

## 🔧 Requirements

See `requirements.txt` for Python dependencies.

## 📝 Notes

- Each dataset folder is self-contained with its own scripts and results
- Scripts use relative paths (run from dataset folder)
- All preprocessing uses 10-20 system channel selection
- Feature extraction uses Riemannian geometry (frequency-band covariance)
