# EEG Research Repository

Repository for EEG-based pain detection and analysis projects using machine learning and transfer learning.

## Main Projects

###  [eeg_pain_simple](./eeg_pain_simple/)
Cross-subject EEG-based pain assessment using transfer learning.

- **Preprocessing**: 20-channel 10-20 system, baseline/pain epochs
- **Features**: Riemannian geometry (840 features from 4 frequency bands)
- **Models**: Domain-Adversarial Neural Network (DANN) with subject-wise validation
- **Results**: 73.96% cross-subject validation accuracy (26 subjects, ds005284 dataset)
- **Multi-dataset support**: BioVid, PainMonit, SEED-IV


## Quick Start

See individual project READMEs for detailed setup instructions:

- [eeg_pain_simple/README.md](./eeg_pain_simple/README.md)



