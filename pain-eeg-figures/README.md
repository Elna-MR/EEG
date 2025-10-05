# Pain EEG Figures (Cursor-ready)

This mini project generates **publication-ready figures** that illustrate EEG changes associated with pain
(↑gamma, ↓alpha, ↑theta), using your **own EEG data** (e.g., from OSF).

It will produce:
- **Figure 1**: Bar plot of **theta (4–8 Hz)**, **alpha (8–12 Hz)**, **gamma (30–45+ Hz)** power for **low vs. high pain**.
- **Figure 2**: **Topographical maps** (alpha & gamma) for **low**, **high**, and **delta (high−low)**.
- **Figure 3** (optional): **Time–frequency spectrogram** example from one file.

> Tip: If your setup supports higher frequencies, pass `--lpf 70 --gamma-high 70` (or up to 100).

---

## 1) Setup (recommended in a fresh venv)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Put data in `data/`
Supported formats: **.edf, .bdf, .vhdr (BrainVision), .fif**

Optionally provide `data/labels.csv` with either:
```
file,condition,pain_score
subj01.edf,low,2
subj02.edf,high,7
subj03.vhdr,,
```
- Use **either** `condition` (`low`/`high`) **or** `pain_score` (0–10).
- If neither is provided, the script will perform a **median split** on a proxy (alpha power).

## 3) Run
Basic run (auto-detect low/high groups if no labels):
```bash
python make_pain_figures.py --data data
```

With labels + higher gamma band:
```bash
python make_pain_figures.py --data data --labels data/labels.csv --lpf 70 --gamma-high 70
```

Set montage name (if needed):
```bash
python make_pain_figures.py --data data --montage standard_1020
```

Outputs:
- PNGs in `figures/`:
  - `fig1_spectral_power.png`
  - `fig2_topo_alpha_low.png`, `fig2_topo_alpha_high.png`, `fig2_topo_alpha_delta.png`
  - `fig2_topo_gamma_low.png`, `fig2_topo_gamma_high.png`, `fig2_topo_gamma_delta.png`
  - `fig3_timefreq_example.png` (optional)

## 4) Notes
- For **topomaps**, channel locations are required. This script applies a standard montage (default: `standard_1020`).
- If channel names are non-standard or locations are missing, topomaps may be skipped gracefully.
- For clean gamma up to ~70–100 Hz, ensure adequate sampling rate (≥250–500 Hz) and minimal high‑frequency noise.
- You can safely rerun; plots will be overwritten.

## 5) Citation
Use your dataset’s license and citation. For the narrative (↑gamma/↓alpha/↑theta), cite a suitable review/meta‑analysis.
