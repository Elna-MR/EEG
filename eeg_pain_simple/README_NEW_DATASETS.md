# Processing New EEG Pain Datasets

This guide explains how to process additional EEG pain datasets (BioVid, PainMonit, SEED-Pain) using the same pipeline as ds005284.

## Available Datasets

1. **BioVid Heat Pain Database**
   - 87 subjects, 30-channel EEG
   - Thermal stimulation
   - Download: Contact dataset authors or check BCI competition repositories

2. **PainMonit Database**
   - 56 subjects
   - Electrical stimulation
   - Download: https://www.painmonit.com/

3. **SEED-Pain (SEED-IV subset)**
   - 15 subjects, 62-channel EEG
   - Electrical stimulation with emotional valence
   - Download: http://bcmi.sjtu.edu.cn/~seed/seed-iv.html

## Quick Start

### Step 1: Download a Dataset

Download one of the datasets and extract it to:
```
eeg_pain_simple/data/{dataset_name}/
```

### Step 2: Preprocess the Dataset

```bash
cd eeg_pain_simple
python src/preprocess_multidataset.py \
    --dataset {dataset_name} \
    --data-dir data/{dataset_name}/ \
    --output {dataset_name}_pain.npz
```

Example:
```bash
python src/preprocess_multidataset.py \
    --dataset biovid \
    --data-dir data/biovid/ \
    --output biovid_pain.npz
```

### Step 3: Extract Features

```bash
python src/extract_features.py \
    --npz packed/{dataset_name}_pain.npz \
    --out packed/features_{dataset_name}.npz
```

### Step 4: Train DANN

```bash
python src/train_dann.py \
    --features packed/features_{dataset_name}.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports/{dataset_name}
```

## Dataset-Specific Configuration

The preprocessing script automatically adapts to different datasets:
- **File formats**: BDF, EDF (MAT needs conversion first)
- **Event codes**: Auto-detects pain events or uses config
- **Channel naming**: Handles Biosemi format or standard names
- **Timing**: Configurable baseline/pain windows

## Current Pipeline Results

- **Dataset**: ds005284 (26 subjects)
- **Channels**: 20 (10-20 system)
- **Features**: 840 (Riemannian tangent space)
- **DANN Accuracy**: 73.96% (subject-wise split)
- **Pipeline**: Preprocess → Feature Extraction → DANN Training

## Notes

- Large data files (.bdf, .npz) are excluded from git (see .gitignore)
- Preprocessed data is saved in `packed/` directory
- Results are saved in `reports/` directory
- All datasets use the same 20-channel 10-20 system configuration

