# Subject-Level Performance Analysis

This document describes how to generate subject-level performance visualizations and analysis for both datasets.

## Overview

The `analyze_subject_performance.py` script generates comprehensive visualizations showing:

1. **Per-Subject Performance**: Classification accuracy for 3-5 selected subjects
2. **Bias Reduction**: Domain confusion matrices demonstrating how DANN reduces subject-specific bias
3. **Covariance Matrices**: Example covariance matrices across frequency bands
4. **Riemannian Mean Trajectories**: Mean covariance patterns for baseline vs pain conditions per subject

## Usage

### For osf-data dataset:

```bash
cd datasets/osf-data
python3 scripts/analyze_subject_performance.py \
    --features packed/features_osf-data.npz \
    --preprocessed packed/osf-data_pain.npz \
    --report-dir reports \
    --n-subjects 5 \
    --device cpu
```

### For ds005284 dataset:

```bash
cd datasets/ds005284
python3 scripts/analyze_subject_performance.py \
    --features packed/features_ds005284.npz \
    --preprocessed packed/ds005284_pain.npz \
    --report-dir reports \
    --n-subjects 5 \
    --device cpu
```

## Arguments

- `--features`: Path to features NPZ file (default: `../packed/features_<dataset>.npz`)
- `--preprocessed`: Path to preprocessed NPZ file (default: `../packed/<dataset>_pain.npz`)
- `--report-dir`: Directory to save reports (default: `../reports`)
- `--n-subjects`: Number of subjects to show in detail (default: 5)
- `--device`: Device to use (`cpu`, `cuda`, or `mps`, default: `cpu`)

## Output Files

The script generates the following files in the `--report-dir` directory:

1. **subject_performance.png**: Bar chart showing per-subject accuracy with summary statistics
2. **bias_reduction.png**: Domain confusion matrix and domain discriminator performance
3. **covariance_matrices.png**: Example covariance matrices for 5 trials across 4 frequency bands
4. **riemannian_trajectories.png**: Riemannian mean covariance trajectories for selected subjects
5. **subject_analysis_summary.json**: JSON file with detailed metrics

## What Each Visualization Shows

### 1. Subject Performance
- Horizontal bar chart showing accuracy for selected subjects
- Color-coded by train/validation split
- Summary statistics (mean ± std, min, max)
- Helps identify subjects with particularly good or poor performance

### 2. Bias Reduction
- **Left panel**: Domain confusion matrix showing how well the domain discriminator can identify subjects
  - Lower diagonal values = better domain adaptation (less subject-specific bias)
- **Right panel**: Domain prediction accuracy per subject
  - Lower accuracy = better bias reduction (domain discriminator struggles to identify subjects)

### 3. Covariance Matrices
- Shows example covariance matrices for 5 randomly selected trials
- Organized by frequency band (Theta, Alpha, Beta, Gamma)
- Labeled as "Pain" or "Baseline"
- Demonstrates the structure of Riemannian features used for classification

### 4. Riemannian Mean Trajectories
- Shows mean covariance matrices for baseline and pain conditions
- Computed per subject using Riemannian mean
- Organized by subject (rows) and frequency band (columns)
- Visualizes how covariance patterns differ between conditions

## Notes

- The script will retrain the DANN model if no checkpoint is found (takes a few minutes)
- For faster execution, ensure the dataset has been preprocessed and features extracted
- The script automatically selects a mix of validation and training subjects to display
- GPU/MPS acceleration is supported if available (use `--device cuda` or `--device mps`)

## Example Output

After running the script, you should see output like:

```
[INFO] Loading data...
[INFO] Features shape: (832, 840)
[INFO] Raw data shape: (832, 20, 512)
[INFO] Unique subjects: 22
[INFO] Using device: cpu
[INFO] Setting up model...
[INFO] Training model for analysis (this may take a few minutes)...
[INFO] Computing per-subject metrics...
[INFO] Computing domain confusion...
[INFO] Extracting covariance matrices...
[INFO] Computing Riemannian mean trajectories...
[INFO] Creating visualizations...

[INFO] Analysis complete! Results saved to reports/
  - subject_performance.png
  - bias_reduction.png
  - covariance_matrices.png
  - riemannian_trajectories.png
  - subject_analysis_summary.json
```

