# cpCGX_BIDS Dataset

Chronic Pain Resting-State EEG Dataset (BIDS format)

## Structure

```
cpcgx/
├── data/              # Raw BrainVision files (.vhdr/.eeg/.vmrk)
├── scripts/           # Preprocessing, feature extraction, DANN training
├── packed/            # Preprocessed epochs and features
└── reports/           # DANN training results and metrics
```

## Quick Start

```bash
cd datasets/cpcgx

# Step 1: Preprocess
python scripts/preprocess_cpcgx.py \
    --root data \
    --out packed/cpcgx_pain.npz

# Step 2: Extract features
python scripts/extract_features.py \
    --npz packed/cpcgx_pain.npz \
    --out packed/features_cpcgx.npz

# Step 3: Train DANN
python scripts/train_dann_cpcgx.py \
    --features packed/features_cpcgx.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports
```

## Dataset Info

- **Subjects:** 74
- **Tasks:** Eyes Open (EO), Eyes Closed (EC)
- **Channels:** 29 (10-20 system)
- **Sampling Rate:** 500 Hz (downsampled to 256 Hz)
- **Format:** BrainVision (.vhdr/.eeg/.vmrk)

## Results

- **Preprocessed:** 19,918 epochs (9,959 EO + 9,959 EC)
- **Features:** 1,740 Riemannian features
- **DANN Accuracy:** 87.44% (best validation)

