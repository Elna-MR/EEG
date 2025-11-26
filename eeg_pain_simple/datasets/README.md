# Datasets Organization

This directory contains separate folders for each EEG pain dataset, each with its own:
- **data/** - Raw data files
- **scripts/** - Preprocessing, feature extraction, and DANN training scripts
- **packed/** - Preprocessed epochs and extracted features
- **reports/** - Training results, metrics, and visualizations

## Available Datasets

### cpCGX_BIDS
- **Location:** `cpcgx/`
- **Type:** Chronic Pain Resting-State EEG
- **Format:** BrainVision (.vhdr/.eeg/.vmrk)
- **Subjects:** 74
- **Tasks:** Eyes Open (EO), Eyes Closed (EC)

### ds005284
- **Location:** `ds005284/`
- **Type:** Pain vs Baseline EEG
- **Format:** BDF (Biosemi)
- **Task:** Condition 54 (pain onset)

## Usage

Each dataset folder is self-contained. Navigate to the dataset folder and run:

```bash
cd datasets/cpcgx  # or datasets/ds005284
./run_pipeline.sh
```

Or run scripts individually from the dataset folder:
```bash
python scripts/preprocess_*.py
python scripts/extract_features.py
python scripts/train_dann*.py
```

All scripts use relative paths, so they work when run from the dataset folder.

