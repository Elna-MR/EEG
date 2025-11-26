# cpCGX_BIDS Dataset Information

## Brief Overview

**cpCGX** is a **Chronic Pain Resting-State EEG Dataset** in BIDS format.

### Key Characteristics

- **Type:** Resting-state EEG (no task, just resting)
- **Subjects:** 74 participants
- **Tasks:** 
  - **Eyes Open (EO)** - Label 0
  - **Eyes Closed (EC)** - Label 1
- **Format:** BrainVision (.vhdr/.eeg/.vmrk files)
- **Original Sampling Rate:** 500 Hz
- **Processed Sampling Rate:** 256 Hz

### Data Structure

- **Total Files:** 144 BrainVision files (2 per subject: EO + EC)
- **Preprocessed Epochs:** 19,918 epochs
  - 9,959 Eyes Open epochs
  - 9,959 Eyes Closed epochs
- **Epoch Duration:** 4 seconds (sliding windows with 50% overlap)
- **Channels:** 19 standard 10-20 system channels
  - FP1, FP2, F3, F4, F7, F8, FZ, C3, C4, CZ, P3, P4, PZ, T7, T8, P7, P8, O1, O2

### Processing Pipeline

1. **Load:** BrainVision files → MNE Raw objects
2. **Filter:** 1-45 Hz bandpass
3. **Resample:** 500 Hz → 256 Hz
4. **Channel Selection:** Select 19 standard 10-20 channels
5. **ICA:** Remove eye blink artifacts
6. **Epoch:** 4-second sliding windows (2-second overlap)
7. **Balance:** Equal number of EO and EC epochs

### Features

- **Riemannian Features:** 760 features
  - 4 frequency bands: (4-8 Hz), (8-12 Hz), (13-30 Hz), (30-45 Hz)
  - Ledoit-Wolf covariance estimation
  - Log-Euclidean tangent space mapping
  - 19 channels × 4 bands × ~10 features per band = 760 features

### Results

- **DANN Accuracy:** 87.80% (final), 88.93% (best)
- **Cross-subject generalization:** 59 train subjects, 15 validation subjects
- **Task:** Binary classification (Eyes Open vs Eyes Closed)

### Purpose

This dataset is used to study **chronic pain** through resting-state EEG patterns. The Eyes Open vs Eyes Closed comparison helps identify baseline brain activity differences that may be related to pain states.

