# Dataset Download and Processing Guide

## Overview

This guide helps you download and process one of the three publicly available EEG pain datasets using the same pipeline as ds005284.

## Available Datasets

### 1. BioVid Heat Pain Database (Recommended - Largest)
- **Subjects**: 87
- **Channels**: 30-channel EEG
- **Stimulation**: Thermal (heat pain)
- **Download**: 
  - Check: https://www.aau.at/en/research/research-areas/affective-computing/
  - Or search: "BioVid dataset download" in academic databases
  - May require email to authors

### 2. PainMonit Database
- **Subjects**: 56  
- **Channels**: Variable
- **Stimulation**: Electrical
- **Download**: https://www.painmonit.com/
  - May require registration

### 3. SEED-Pain (SEED-IV subset)
- **Subjects**: 15
- **Channels**: 62-channel EEG
- **Stimulation**: Electrical with emotional valence
- **Download**: http://bcmi.sjtu.edu.cn/~seed/seed-iv.html
  - Download full SEED-IV dataset
  - Extract pain-related trials

## Quick Start

### Step 1: Download Dataset

1. Choose a dataset (BioVid recommended for largest sample size)
2. Download and extract to: `eeg_pain_simple/data/{dataset_name}/`
3. Ensure files are in BDF or EDF format (SEED uses MAT - needs conversion)

### Step 2: Run Full Pipeline

```bash
cd eeg_pain_simple
./run_new_dataset.sh {dataset_name} {data_directory}
```

Example:
```bash
./run_new_dataset.sh biovid data/biovid/
```

This will:
1. Preprocess → Extract 20-channel 10-20 system
2. Extract features → 840 Riemannian tangent space features  
3. Train DANN → Subject-wise split, domain adaptation

### Step 3: Compare Results

Results will be saved to `reports/{dataset_name}/`:
- `dann_metrics.json` - Validation accuracy, loss
- `dann_training_curves.png` - Training visualization

Compare with current ds005284 results:
- **Current**: 73.96% validation accuracy (26 subjects)
- **New dataset**: Will show cross-dataset generalization

## Manual Step-by-Step

If you prefer to run steps individually:

```bash
# 1. Preprocess
python src/preprocess_multidataset.py \
    --dataset biovid \
    --data-dir data/biovid/ \
    --output biovid_pain.npz

# 2. Extract Features  
python src/extract_features.py \
    --npz packed/biovid_pain.npz \
    --out packed/features_biovid.npz

# 3. Train DANN
python src/train_dann.py \
    --features packed/features_biovid.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports/biovid
```

## Dataset Configuration

The script auto-detects:
- **File format**: BDF, EDF (MAT needs conversion)
- **Event codes**: Auto-detects pain events or uses config
- **Channel names**: Handles Biosemi format or standard names
- **Timing windows**: Configurable baseline (-1.5 to 0s) and pain (0 to 2s)

## Expected Results

Using the same approach as ds005284:
- **Preprocessing**: 20-channel 10-20 system
- **Features**: 840 Riemannian features (4 frequency bands)
- **Model**: DANN with subject-wise train/val split
- **Expected**: Similar or improved accuracy compared to ds005284

## Troubleshooting

**Issue**: "No files found"
- Check file pattern matches dataset structure
- Verify files are in correct directory
- Check file format (BDF/EDF supported)

**Issue**: "No pain events found"  
- Check event codes in dataset
- May need to manually specify event code in config
- Check events.tsv file for correct event mapping

**Issue**: "Channel names not found"
- Dataset may use non-standard channel names
- Check channels.tsv file
- May need to add channel mapping to `rename_channels_from_biosemi()` function

## Next Steps

After processing a new dataset:
1. Compare results with ds005284
2. Analyze cross-dataset generalization
3. Consider combining datasets for larger training set
4. Test transfer learning across datasets

