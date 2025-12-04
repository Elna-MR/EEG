# osf-data Dataset

EEG pain detection dataset from OSF (Open Science Framework) - Preprocessing, Riemannian Features, and DANN Training

## Structure

```
osf-data/
├── data/              # Raw BrainVision data (.dat/.evt files)
│   ├── Study One/
│   │   └── P_XX/
│   └── Study Two/
│       └── P_XX/
├── scripts/           # Preprocessing, feature extraction, DANN training
│   ├── preprocess.py
│   ├── extract_features.py
│   └── train_dann.py
├── packed/            # Preprocessed epochs and features
└── reports/           # DANN training results and metrics
```

## Pipeline

This dataset uses a 3-step pipeline:
1. **Preprocessing** - Load BrainVision files, filter, epoch (baseline vs pain)
2. **Riemannian Feature Extraction** - Extract frequency-band covariance features
3. **DANN Training** - Domain Adversarial Neural Network for domain adaptation

## Quick Start

### Prerequisites
Activate your Python virtual environment with required packages:
```bash
# Activate your venv (adjust path as needed)
source /path/to/venv/bin/activate  # or: conda activate your_env
```

### Run Complete Pipeline
```bash
cd datasets/osf-data
./run_pipeline.sh
```

### Or Run Step-by-Step
```bash
cd datasets/osf-data

# Step 1: Preprocess
python3 scripts/preprocess.py

# Step 2: Extract Riemannian features
python3 scripts/extract_features.py \
    --npz packed/osf-data_pain.npz \
    --out packed/features_osf-data.npz

# Step 3: Train DANN
python3 scripts/train_dann.py \
    --features packed/features_osf-data.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports
```

```

## Results

Results are saved to `reports/` directory:
- `dann_metrics.json` - Final validation accuracy and loss
- `dann_training_curves.png` - Training visualization
- `dann_logs/` - Detailed training history

## Comparison with ds005284

This dataset uses the **exact same preprocessing and feature extraction** as ds005284 to enable fair comparison:
- Same filtering (1-45 Hz)
- Same channel selection (10-20 system)
- Same epoch timing (baseline: -1.5 to 0.0s, pain: 0.0 to 2.0s)
- Same feature extraction (Riemannian geometry, same frequency bands)
- Same DANN architecture and hyperparameters

## Notes

- Event codes 21 and 22 are used as pain markers (based on .evt file structure)
- Files are BrainVision format (.dat/.evt) without .vhdr headers
- Scripts automatically detect channel count and sampling rate
- Uses 20 channels from 10-20 system (selected from initial 64 channels)


