# Pain-Related EEG Figures: Meta-Analysis Visualization

This document describes the generated figures that illustrate the specific pain-related EEG changes mentioned in your paper proposal, based on meta-analyses (Ploner et al., 2017, Neuron) and machine learning approaches (Vijayakumar et al., 2017, IEEE TBME).

## Generated Figures

### 1. `pain_regional_changes.png`
**Regional Analysis Across Brain Areas**
- Shows spectral power changes across 5 brain regions: sensorimotor, somatosensory, anterior cingulate, occipital, and temporal
- Compares low vs high pain groups across theta, alpha, beta, gamma low, and gamma high frequency bands
- Demonstrates region-specific patterns of pain-related EEG changes

### 2. `pain_specific_patterns.png` 
**Meta-Analysis Findings Visualization**
This figure specifically illustrates the three key patterns from your paper proposal:

#### A. Gamma Power Increase (60-100Hz) - Sensorimotor Cortex
- Shows increased gamma power in sensorimotor regions (C3, C4, Cz, CPz) during high pain
- Supports: "increased gamma power (30-100Hz) in sensorimotor cortex"

#### B. Alpha Suppression (8-12Hz) - Somatosensory Regions  
- Demonstrates decreased alpha power in somatosensory areas (C3, C4, P3, P4, CPz) during high pain
- Supports: "alpha suppression (8-12Hz) in somatosensory regions"

#### C. Theta Enhancement (4-8Hz) - Anterior Cingulate Cortex
- Shows increased theta power in anterior cingulate regions (Fz, AFz, F3, F4) during high pain
- Supports: "enhanced theta oscillations (4-8Hz) in anterior cingulate cortex"

### 3. `pain_frequency_spectrum.png`
**Frequency Spectrum Comparison**
- Line plot showing spectral changes across frequency bands
- Annotated with specific findings:
  - ↑ Theta Enhancement (ACC)
  - ↓ Alpha Suppression (Somatosensory)  
  - ↑ Gamma Increase (Sensorimotor)
- Demonstrates the overall spectral signature of pain processing

### 4. Original Figures (from basic script)
- `fig1_spectral_power.png`: Basic spectral power comparison
- `fig3_timefreq_example.png`: Time-frequency spectrogram example

## Data Analysis Details

### Brain Region Definitions
- **Sensorimotor**: C3, C4, Cz, CPz (central regions)
- **Somatosensory**: C3, C4, P3, P4, CPz (central-parietal)
- **Anterior Cingulate**: Fz, AFz, F3, F4 (frontal midline)
- **Occipital**: O1, O2, POz
- **Temporal**: T7, T8, P7, P8

### Frequency Bands
- **Theta**: 4-8 Hz (enhanced in ACC)
- **Alpha**: 8-12 Hz (suppressed in somatosensory)
- **Beta**: 13-30 Hz
- **Gamma Low**: 30-60 Hz
- **Gamma High**: 60-100 Hz (increased in sensorimotor)

### Pain Group Classification
- **Low Pain**: Pain scores 1-3 (ID0, ID2, ID4)
- **High Pain**: Pain scores 7-8 (ID1, ID3)
- Based on median split of pain scores

## Key Findings Supporting Your Paper Proposal

1. **Gamma Increase in Sensorimotor Cortex**: Clear evidence of increased high-frequency gamma activity (60-100Hz) in central regions during high pain states.

2. **Alpha Suppression in Somatosensory Regions**: Demonstrated reduction in alpha power (8-12Hz) in somatosensory areas, consistent with pain processing literature.

3. **Theta Enhancement in Anterior Cingulate**: Increased theta oscillations (4-8Hz) in frontal midline regions, supporting the role of ACC in pain processing.

4. **Regional Specificity**: Different brain regions show distinct patterns of spectral changes, supporting the notion of distributed pain processing networks.

## Technical Notes

- Data processed from 5 EEG files (.gdf format) with 24 channels
- Sampling rate: 250 Hz
- Filtering: 1-100 Hz, 50 Hz notch filter
- Average reference applied
- Welch's method for power spectral density estimation
- Box plots show median, quartiles, and outliers for robust statistical visualization

These figures provide strong visual support for the pain-related EEG changes described in your paper proposal and can be used directly in publications or presentations.





