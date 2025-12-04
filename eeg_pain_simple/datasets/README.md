# Datasets Organization

This directory contains separate folders for each EEG pain dataset, each with its own standardized structure:
- **data/** - Raw data files (BIDS-compatible structure)
- **scripts/** - Preprocessing, feature extraction, and DANN training scripts
- **packed/** - Preprocessed epochs and extracted features
- **reports/** - Training results, metrics, and visualizations
- **run_pipeline.sh** - Complete pipeline script

## Available Datasets

### ds005284
- **Location:** `ds005284/`
- **Type:** Pain vs Baseline EEG
- **Format:** BDF (Biosemi), BIDS-compliant
- **Subjects:** 26
- **Trials:** 832 (416 pain + 416 baseline)
- **Channels:** 20 (10-20 system: FP1, FP2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, T7, T8, P7, P8, O1, O2, OZ)
- **Task:** Condition 54 (pain onset)
- **Preprocessing:** 1-45 Hz filter, epochs: baseline (-1.5 to 0.0s), pain (0.0 to 2.0s)
- **BIDS Metadata:** Includes `dataset_description.json`, `participants.tsv`, `task-26ByBiosemi_events.json` at root level

### osf-data
- **Location:** `osf-data/`
- **Type:** Pain vs Baseline EEG
- **Format:** BrainVision (.dat/.evt files)
- **Subjects:** 22
- **Trials:** 832 (416 pain + 416 baseline, balanced and limited to match ds005284)
- **Channels:** 20 (10-20 system, selected from initial 64 channels)
- **Preprocessing:** Same pipeline as ds005284 for fair comparison
  - 1-45 Hz filter
  - Epochs: baseline (-1.5 to 0.0s), pain (0.0 to 2.0s)
  - Event codes 21/22 used as pain markers
- **Structure:** Matches ds005284 folder structure exactly

## Standardized Pipeline

Both datasets use the **exact same preprocessing and feature extraction pipeline** for fair comparison:

1. **Preprocessing** (`scripts/preprocess.py`)
   - Filter: 1-45 Hz
   - Channel selection: 20 channels (10-20 system)
   - Epoch timing: Baseline (-1.5 to 0.0s), Pain (0.0 to 2.0s)
   - Class balancing: Global balance (416 pain + 416 baseline)
   - Output: `packed/{dataset}_pain.npz`

2. **Feature Extraction** (`scripts/extract_features.py`)
   - Riemannian geometry features
   - Frequency bands: (4-8 Hz), (8-12 Hz), (13-30 Hz), (30-45 Hz)
   - Covariance estimation: LWF (Ledoit-Wolf)
   - Tangent space mapping: Log-Euclidean metric
   - Output: `packed/features_{dataset}.npz`

3. **DANN Training** (`scripts/train_dann.py`)
   - Domain-Adversarial Neural Network
   - Cross-subject validation
   - Hyperparameters: 30 epochs, batch size 128, learning rate 1e-3
   - Output: `reports/dann_metrics.json`, training logs, visualizations

## Usage

Each dataset folder is self-contained. Navigate to the dataset folder and run:

```bash
cd datasets/ds005284  # or datasets/osf-data
./run_pipeline.sh
```

Or run scripts individually from the dataset folder:
```bash
cd datasets/ds005284  # or datasets/osf-data

# Step 1: Preprocess
python3 scripts/preprocess.py

# Step 2: Extract features
python3 scripts/extract_features.py \
    --npz packed/ds005284_pain.npz \
    --out packed/features_ds005284.npz

# Step 3: Train DANN
python3 scripts/train_dann.py \
    --features packed/features_ds005284.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports
```

All scripts use relative paths, so they work when run from the dataset folder.

## Dataset Comparison

| Feature | ds005284 | osf-data |
|---------|----------|----------|
| **Format** | BDF (Biosemi) | BrainVision (.dat/.evt) |
| **Subjects** | 26 | 22 |
| **Trials** | 832 | 832 |
| **Channels** | 20 (10-20) | 20 (10-20, from 64) |
| **Sampling Rate** | 256 Hz | Variable (downsampled to 256 Hz) |
| **Pain Events** | Condition 54 | Event codes 21/22 |
| **Preprocessing** | Identical | Identical |
| **Feature Extraction** | Identical | Identical |
| **DANN Training** | Identical | Identical |

Both datasets are processed identically to enable fair cross-dataset comparison and domain adaptation studies.









