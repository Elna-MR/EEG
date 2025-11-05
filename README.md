# EEG Research Repository

Repository for EEG-based pain detection and analysis projects using machine learning and transfer learning.

## Main Projects

### 🧠 [eeg_pain_simple](./eeg_pain_simple/)
Cross-subject EEG-based pain assessment using transfer learning.

- **Preprocessing**: 20-channel 10-20 system, baseline/pain epochs
- **Features**: Riemannian geometry (840 features from 4 frequency bands)
- **Models**: Domain-Adversarial Neural Network (DANN) with subject-wise validation
- **Results**: 73.96% cross-subject validation accuracy (26 subjects, ds005284 dataset)
- **Multi-dataset support**: BioVid, PainMonit, SEED-IV

### 📥 [seed_iv_downloader](./seed_iv_downloader/)
Utility for downloading and processing SEED-IV EEG dataset.

### 📊 [pain-eeg-figures](./pain-eeg-figures/)
Visualization and figure generation for EEG pain analysis.

### 🔬 [tsen_eeg_project](./tsen_eeg_project/)
EEG feature extraction and classification project.

## Quick Start

See individual project READMEs for detailed setup instructions:

- [eeg_pain_simple/README.md](./eeg_pain_simple/README.md)
- [seed_iv_downloader/README.md](./seed_iv_downloader/README.md)

## Datasets

- **ds005284**: OpenNeuro dataset (26 subjects, laser-evoked pain)
- **SEED-IV**: Emotion recognition dataset (15 subjects)
- Custom datasets in `data-additional/`

## Citation

If you use this repository, please cite:

```
Lu X, Thompson WF, Zhang L, Hu L. Music Reduces Pain Unpleasantness: Evidence from an EEG Study. 
J Pain Res. 2019 Dec 13;12:3331-3342. doi: 10.2147/JPR.S212080.
```

## License

See individual project directories for license information.

