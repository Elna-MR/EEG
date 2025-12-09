# DANN Model Technical Details

## Key Concepts: Subject vs Sample

**Subject**: A person/participant in the EEG study (e.g., "sub-001", "Study One_P_01")
- Each subject has their own EEG recording session(s)
- Multiple subjects = multiple people in the study

**Sample**: A single EEG trial/epoch (one time window of EEG data)
- Each sample represents a time segment of EEG signals
- One sample = one feature vector (9,455 or 9,465 dimensions)
- Samples come from extracting baseline and pain segments from EEG epochs

### How Samples Are Created

During preprocessing, for each subject:
1. **Pain events are detected** in the EEG recording (condition 54 events)
2. **Epochs are created** around each pain event (time window: -1.5s to +2.0s)
3. **Two samples are extracted from each epoch**:
   - **Baseline sample**: EEG from -1.5s to 0.0s (before pain onset)
   - **Pain sample**: EEG from 0.0s to +2.0s (during pain)
4. Each sample is then converted to a feature vector using Riemannian feature extraction

**Example**: If a subject has 16 pain events:
- 16 epochs → 16 baseline samples + 16 pain samples = **32 samples total**

---

## 1. Riemannian Features Extraction

### ds005284 Dataset
- **Features per sample**: 9,455 dimensions
- **Samples per subject**: 32 (consistent across all subjects)
  - **Why 32?** Each subject has exactly 16 pain events
  - 16 pain events → 16 epochs → 16 baseline + 16 pain = 32 samples
  - Breakdown: 16 baseline samples (label=0) + 16 pain samples (label=1)
- **Total subjects**: 26
- **Total samples**: 832 (26 subjects × 32 samples)
- **Feature extraction mode**: `improved` (advanced Riemannian features)
- **Frequency bands**: 5 bands (Delta: 1-4 Hz, Theta: 4-8 Hz, Alpha: 8-12 Hz, Beta: 12-30 Hz, Gamma: 30-45 Hz)
- **Feature components**:
  - Multiple covariance estimators: SCM, LWF, OAS
  - Multiple tangent space metrics: Riemann, LogEuclid
  - Time-windowed covariance (multi-scale)
  - Geodesic distances to reference covariance
  - Cross-band covariance features

### osf-data Dataset
- **Features per sample**: 9,465 dimensions
- **Samples per subject**: Variable (average ~37.8 samples/subject)
  - **Why variable?** Different subjects have different numbers of pain events
  - Range: 14-69 samples per subject
  - Example: Subject "Study One_P_01" has 7 pain events → 14 samples (7 baseline + 7 pain)
  - Example: Subject "Study One_P_07" has 34 pain events → 69 samples (38 baseline + 31 pain, after balancing)
- **Total subjects**: 22
- **Total samples**: 832 (after balancing/limiting to match ds005284)
- **Feature extraction mode**: `improved` (advanced Riemannian features)
- **Frequency bands**: Same as ds005284 (5 bands)
- **Feature components**: Same as ds005284

### Feature Breakdown per Subject

**ds005284:**
- Each subject has: **16 pain events → 32 samples** (16 baseline + 16 pain)
- Each sample has: **9,455 features**
- Total per subject: **32 samples × 9,455 features = 302,560 feature values**
- Memory (float32): ~1.15 MB per subject

**osf-data:**
- Variable pain events per subject → variable samples
- Average: **~19 pain events → ~38 samples** (varies by subject)
- Each sample has: **9,465 features**
- Average per subject: **~38 samples × 9,465 features = ~359,670 feature values**
- Memory (float32): ~1.37 MB per subject (average)
- Range: 14-69 samples per subject depending on number of pain events recorded

---

## 2. Training Time

### Training Configuration
- **Model**: Improved DANN (Domain-Adversarial Neural Network)
- **Training epochs**: 48 epochs (0-47)
- **Early stopping**: Enabled (patience=15, monitor='val_loss')
- **Best epoch**: 47 (validation loss: 0.034)
- **Batch size**: 32
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-3)
- **Learning rate scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

### Training Data
- **Training samples**: 672 (21 subjects, 80% of ds005284)
- **Test samples**: 160 (5 subjects, 20% of ds005284)
- **PCA components**: 500 (reduced from 9,455 features)
- **Input dimension after PCA**: 500

### Estimated Training Time
- **Per epoch**: ~20-30 seconds (based on log frequency and batch processing)
- **Total training time**: ~16-24 minutes for 48 epochs
- **Note**: Actual time depends on hardware (CPU/GPU), but training uses PyTorch Lightning which optimizes performance

### Training Results
- **Train Accuracy**: 100.00%
- **Test Accuracy**: 96.88%
- **Overfitting Gap**: 3.12%
- **Validation Loss**: 0.034 (best epoch)

---

## 3. Cross-Validation

### Current Approach: Subject-Level Train/Test Split
**No k-fold cross-validation is used.** Instead, the model uses a **single subject-level split**:

- **Split ratio**: 80% train / 20% test
- **Split method**: Subject-level (prevents data leakage)
- **Random seed**: 42 (for reproducibility)
- **ds005284 split**:
  - Train: 21 subjects (672 samples)
  - Test: 5 subjects (160 samples)
- **osf-data**: Used entirely for external testing (no training split)

### Why Subject-Level Split?
- Prevents data leakage (samples from the same subject don't appear in both train and test)
- More realistic evaluation (tests generalization to new subjects)
- Standard practice for EEG/BCI applications

### Cross-Validation Alternative
If k-fold cross-validation were desired, it would require:
- **GroupKFold** or **StratifiedGroupKFold** (from sklearn)
- Groups: Subject IDs
- Typical folds: 5-fold or 10-fold
- **Note**: This would require retraining the model k times, significantly increasing computational cost

---

## 4. Computational Load for 3-5 Subjects

### Feature Extraction

#### ds005284 (3 subjects)
- **Samples**: 96 (3 × 32)
- **Features per sample**: 9,455
- **Total feature values**: 907,680
- **Memory (float32)**: ~3.46 MB
- **Estimated extraction time**: ~2-5 minutes (depends on CPU)

#### ds005284 (5 subjects)
- **Samples**: 160 (5 × 32)
- **Features per sample**: 9,455
- **Total feature values**: 1,512,800
- **Memory (float32)**: ~5.77 MB
- **Estimated extraction time**: ~4-8 minutes

#### osf-data (3 subjects, average)
- **Samples**: ~114 (3 × 38)
- **Features per sample**: 9,465
- **Total feature values**: ~1,079,010
- **Memory (float32)**: ~4.12 MB
- **Estimated extraction time**: ~3-6 minutes

#### osf-data (5 subjects, average)
- **Samples**: ~190 (5 × 38)
- **Features per sample**: 9,465
- **Total feature values**: ~1,798,350
- **Memory (float32)**: ~6.86 MB
- **Estimated extraction time**: ~5-10 minutes

### Model Training (for 3-5 subjects)

#### Training Data Size
- **3 subjects**: ~96-114 samples (depending on dataset)
- **5 subjects**: ~160-190 samples (depending on dataset)

#### Computational Requirements
- **PCA**: O(n_samples × n_features) → O(n_samples × 500) after PCA
- **DANN Training**:
  - Forward pass: O(batch_size × hidden_size × input_dim)
  - Backward pass: ~2× forward pass
  - Per epoch: ~3-10 batches (depending on batch size)
  - Total: ~48 epochs × batches per epoch

#### Estimated Training Time (3-5 subjects)
- **3 subjects**: ~5-10 minutes (fewer samples, faster convergence)
- **5 subjects**: ~8-15 minutes (more samples, similar convergence)

#### Memory Requirements (Training)
- **Feature matrix**: ~3-7 MB (as calculated above)
- **PCA transformation**: ~0.5-1 MB
- **Model parameters**: ~500 KB (DANN model size)
- **Gradients**: ~500 KB
- **Total**: ~5-10 MB (very lightweight)

### Inference (Prediction)
- **Per sample**: <1 ms (on CPU), <0.1 ms (on GPU)
- **3 subjects (96 samples)**: ~100 ms total
- **5 subjects (160 samples)**: ~160 ms total

---

## Summary Table

| Metric | ds005284 | osf-data |
|--------|----------|----------|
| **Features per sample** | 9,455 | 9,465 |
| **Samples per subject** | 32 (fixed) | ~38 (variable) |
| **Features per subject** | 302,560 | ~359,670 |
| **Memory per subject** | ~1.15 MB | ~1.37 MB |
| **3 subjects - Memory** | ~3.46 MB | ~4.12 MB |
| **5 subjects - Memory** | ~5.77 MB | ~6.86 MB |
| **3 subjects - Extraction time** | ~2-5 min | ~3-6 min |
| **5 subjects - Extraction time** | ~4-8 min | ~5-10 min |
| **Training time (full dataset)** | ~16-24 min (48 epochs) | N/A (test only) |
| **Training time (3-5 subjects)** | ~5-15 min | N/A |
| **Cross-validation** | No (subject-level split) | No (external test) |

---

## Notes

1. **Feature Extraction**: The most computationally intensive step. Uses multiple covariance estimators, tangent space mappings, and multi-scale features.

2. **Training**: Relatively fast due to PCA dimensionality reduction (9,455 → 500) and efficient PyTorch Lightning implementation.

3. **Memory**: Very lightweight - can run on standard laptops/desktops without GPU.

4. **Scalability**: The model scales linearly with the number of subjects/samples. For larger datasets, consider:
   - Batch processing for feature extraction
   - GPU acceleration for training
   - Distributed training for very large datasets

5. **Cross-Dataset Generalization**: The improved DANN achieves 86.18% accuracy on osf-data (external dataset), demonstrating good cross-dataset generalization.

