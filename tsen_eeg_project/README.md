# TsEn-EEG Project (POC)

Proof-of-concept pipeline to extract **Tsallis Entropy (TsEn)** features (q = 2, 3, 4) from EEG,
then train a simple **KNN** classifier. Designed to work with EEG datasets in EDF format.

## Project structure
```
tsen_eeg_project/
  ├─ data/                 # Put your EDF files here
  ├─ outputs/              # Features & models are written here
  ├─ src/
  │   ├─ config.py
  │   ├─ data_loader.py
  │   ├─ preprocess.py
  │   ├─ features.py
  │   ├─ model.py
  │   ├─ train.py          # End-to-end: extract features + (optional) train/eval
  │   └─ utils.py
  ├─ requirements.txt
  └─ README.md
```

## Quick start

1) **Install deps (recommended: Python 3.10+)**
```
pip install -r requirements.txt
```

2) **Place data**
- Put your `.edf` files into `./data/`.
- Create a `labels.csv` in project root (or pass a path via `--labels`), with two columns:
```
filename,label
subject01_eyes_open.edf,eyes_open
subject01_eyes_closed.edf,eyes_closed
...
```
(If your dataset has different labels, adjust accordingly. If you have no labels yet, you can run
feature extraction only with `--no-train` to just generate the features CSV.)

3) **Run POC (default: resample to 200 Hz, compute TsEn q=2,3,4 on θ/α/β/γ bands)**
```
python -m src.train --data ./data --out ./outputs --labels ./labels.csv
```
To only extract features (no training):
```
python -m src.train --data ./data --out ./outputs --no-train
```

4) **Outputs**
- `./outputs/features.csv`: one row per file (and optionally per-channel) with TsEn features
- `./outputs/knn_metrics.json`: accuracy / F1 (if training performed)

## Notes
- Band definitions: θ(4–7Hz), α(8–15Hz), β(16–31Hz), γ(32–55Hz). Resampled to 200 Hz.
- TsEn computed via sliding window (default 2s window, 1s step), then **mean** and **variance**
  of TsEn per (channel, band, q). These become features.
- Default classifier: KNN (k=10, Euclidean), with 5-fold stratified CV.

## Customize
- Change bands or window sizes in `src/config.py`.
- Swap classifier in `src/model.py`.
- Add channel selection in `src/data_loader.py`.



### Dataset Overview:
- 36 GDF files (General Data Format for EEG)
- 2 classes: class_1 (10 files) and class_2 (26 files)
- 24 EEG channels per recording (ch00-ch23)
- 576 features extracted per file


- ch{channel:02d}_{band}_{q}_{statistic}
- Channels: 00-23 (24 channels)
- Frequency Bands: theta(4-7Hz), alpha(8-15Hz), beta(16-31Hz), gamma(32-55Hz)
- Tsallis q-values: 2, 3, 4
- Statistics: mean, var (variance)


- Class Imbalance: 26 vs 10 samples - may affect classifier performance
- High Dimensionality: 576 features from 36 samples - potential overfitting
- Good Separation: 72% accuracy suggests meaningful differences between classes
- EEG Complexity: TsEn captures subtle brain state variations

