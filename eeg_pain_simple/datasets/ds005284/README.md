# ds005284 Dataset

Pain EEG Dataset (OpenNeuro) - Preprocessing, Riemannian Features, and DANN Training

## Structure

```
ds005284/
├── data/              # Raw BIDS data (.bdf files)
├── scripts/           # Preprocessing, feature extraction, DANN training
│   ├── preprocess.py
│   ├── extract_features.py
│   └── train_dann.py
├── packed/            # Preprocessed epochs and features
└── reports/           # DANN training results and metrics
```

## Pipeline

This dataset uses a 3-step pipeline:
1. **Preprocessing** - Load BDF files, filter, epoch (baseline vs pain)
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
cd datasets/ds005284
./run_pipeline.sh
```

### Or Run Step-by-Step
```bash
cd datasets/ds005284

# Step 1: Preprocess
python3 scripts/preprocess.py

# Step 2: Extract Riemannian features
python3 scripts/extract_features.py \
    --npz packed/ds005284_pain_19ch.npz \
    --out packed/features_ds005284_19ch.npz

# Step 3: Train DANN
python3 scripts/train_dann.py \
    --features packed/features_ds005284_19ch.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports
```

## Dataset Info

- **Format:** BDF (Biosemi)
- **Task:** Pain vs Baseline classification
- **Events:** Condition 54 (pain onset)
- **Channels:** 19 or 20 (10-20 system)
- **Epochs:** 
  - Baseline: -1.5 to 0.0s (pre-stim)
  - Pain: 0.0 to 2.0s (post-stim)

## Output Files

- **Preprocessed:** `packed/ds005284_pain_19ch.npz` (or `_20ch.npz`)
- **Features:** `packed/features_ds005284_19ch.npz`
- **Reports:** `reports/dann_metrics.json`, `reports/dann_training_curves.png`

## Notes

- Preprocessing creates both 19-channel and 20-channel versions
- Use the appropriate channel version for feature extraction and DANN training
- All scripts use relative paths (run from `datasets/ds005284/` directory)
