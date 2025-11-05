EEG Pain (Simple) — Riemannian + DANN

Stages
- Preprocessing (MNE): Load BDF, band-pass 1–45 Hz, epochs for pain(54) and baseline from pre-stim
- Feature extraction (MNE + pyRiemann): Filter theta/alpha/beta/gamma; covariance; tangent space
- Model training (sklearn): SVM baseline
- Domain adaptation (PyTorch Lightning): Minimal DANN with GRL

Quickstart
1) Copy data into data/ds005284 (BIDS-like folders `sub-XXX/eeg/*.bdf`)
2) Run pipeline (preprocess.py auto-detects data path via config):
   - python src/preprocess.py
   - python src/extract_features.py --npz packed/ds005284_pain.npz --out packed/features.npz
   - python src/train_svm.py --features packed/features.npz
3) Train DANN (optional)
   - python src/train_dann.py --features packed/features.npz --epochs 30

Configuration
Edit config variables at the top of `src/preprocess.py`:
- `DATA_ROOT`: Path to ds005284 (or set EEG_DATA_ROOT env var)
- `BALANCE_PER_SUBJECT`: Balance classes per-subject (False = global balance)
- Epoch timing, filter bands, sampling rate, etc.

Requirements
pip install mne pyriemann scikit-learn numpy scipy torch torchvision pytorch-lightning


