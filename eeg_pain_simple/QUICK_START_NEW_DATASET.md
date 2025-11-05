# Quick Start: Processing a New Dataset

## Current Status

✅ **Pipeline ready**: All scripts configured for multi-dataset support
✅ **Tested**: Works with ds005284 (26 subjects, 20 channels, 73.96% accuracy)

## Next Steps to Process BioVid/PainMonit/SEED-Pain

Since I cannot directly download datasets from the web, here's what you need to do:

### Option 1: Download BioVid (Recommended - 87 subjects)

1. **Download BioVid dataset**:
   ```bash
   # Check these sources:
   # - https://www.aau.at/en/research/research-areas/affective-computing/
   # - Search "BioVid dataset" in academic databases
   # - Contact dataset authors if needed
   ```

2. **Extract to data directory**:
   ```bash
   mkdir -p eeg_pain_simple/data/biovid
   # Extract downloaded files to eeg_pain_simple/data/biovid/
   ```

3. **Run the pipeline**:
   ```bash
   cd eeg_pain_simple
   ./run_new_dataset.sh biovid data/biovid/
   ```

### Option 2: Use Existing Dataset Structure

If you already have one of these datasets:

```bash
cd eeg_pain_simple

# Check what you have
ls -la data/

# Run preprocessing (replace dataset_name with actual name)
python src/preprocess_multidataset.py \
    --dataset {dataset_name} \
    --data-dir data/{dataset_name}/ \
    --output {dataset_name}_pain.npz

# Extract features
python src/extract_features.py \
    --npz packed/{dataset_name}_pain.npz \
    --out packed/features_{dataset_name}.npz

# Train DANN
python src/train_dann.py \
    --features packed/features_{dataset_name}.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports/{dataset_name}
```

## What the Pipeline Does

1. **Preprocessing** (`preprocess_multidataset.py`):
   - Loads BDF/EDF files
   - Filters 1-45 Hz
   - Selects 20-channel 10-20 system
   - Extracts baseline (-1.5 to 0s) and pain (0 to 2s) epochs
   - Saves: `packed/{dataset}_pain.npz`

2. **Feature Extraction** (`extract_features.py`):
   - Applies 4 frequency band filters (theta, alpha, beta, gamma)
   - Computes Riemannian covariance matrices
   - Maps to tangent space (840 features)
   - Saves: `packed/features_{dataset}.npz`

3. **DANN Training** (`train_dann.py`):
   - Subject-wise train/val split (80/20)
   - Domain-adversarial neural network
   - Gradient reversal for domain invariance
   - Saves: `reports/{dataset}/dann_metrics.json`

## Expected Results Format

After running, you'll get:
- **Preprocessed data**: `packed/{dataset}_pain.npz`
- **Features**: `packed/features_{dataset}.npz`
- **Metrics**: `reports/{dataset}/dann_metrics.json`
  - `final_val_accuracy`: Cross-subject validation accuracy
  - `final_val_loss`: Validation loss
  - `final_train_loss`: Training loss

## Comparison with Current Results

**ds005284 (Current)**:
- Subjects: 26
- Validation Accuracy: 73.96%
- Features: 840
- Channels: 20 (10-20 system)

**New Dataset** (expected):
- Will use same pipeline
- Results will show cross-dataset generalization
- Can compare accuracy across different pain modalities

## Need Help?

If you have a dataset downloaded, I can help you:
1. Configure the preprocessing for your specific dataset format
2. Adjust event codes if needed
3. Run the full pipeline
4. Compare results

Just let me know which dataset you have or want to use!

