from typing import Dict, Tuple
import numpy as np

def _probabilities(x: np.ndarray, bins: int) -> np.ndarray:
    # Histogram-based probability estimate
    hist, edges = np.histogram(x, bins=bins, density=False)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.zeros_like(p, dtype=np.float64)
    return p / s

def tsallis_entropy_from_probs(p: np.ndarray, q: float) -> float:
    if q == 1.0:
        # Shannon limit
        p_safe = p[p > 0]
        return -np.sum(p_safe * np.log(p_safe))
    s = np.sum(p ** q)
    return (1.0 - s) / (q - 1.0)

def tsallis_entropy_windowed(x: np.ndarray, fs: int, q: float, window_sec: float, step_sec: float, bins: int) -> Tuple[float, float]:
    """Compute TsEn over sliding windows; returns (mean_TsEn, var_TsEn)."""
    w = int(window_sec * fs)
    s = int(step_sec * fs)
    if w <= 1 or s <= 0 or len(x) < w:
        p = _probabilities(x, bins)
        val = tsallis_entropy_from_probs(p, q)
        return float(val), 0.0

    vals = []
    for start in range(0, len(x) - w + 1, s):
        seg = x[start:start+w]
        p = _probabilities(seg, bins)
        vals.append(tsallis_entropy_from_probs(p, q))
    vals = np.array(vals, dtype=np.float64) if len(vals) else np.array([0.0])
    return float(vals.mean()), float(vals.var())

def compute_tsen_features(data: np.ndarray, fs: int, bands: Dict[str, Tuple[float,float]], qs, window_sec: float, step_sec: float, bins: int) -> Dict[str, float]:
    """
    data: n_channels x n_samples
    Returns a flat dict of features: TsEn_mean/var per (channel, band, q)
    """
    feats = {}
    from .preprocess import extract_band
    n_ch, _ = data.shape
    for ch in range(n_ch):
        x = data[ch]
        for band_name, (lo, hi) in bands.items():
            xb = extract_band(x, fs, (lo, hi))
            for q in qs:
                mean_v, var_v = tsallis_entropy_windowed(xb, fs, q, window_sec, step_sec, bins)
                feats[f"ch{ch:02d}_{band_name}_q{q}_mean"] = mean_v
                feats[f"ch{ch:02d}_{band_name}_q{q}_var"] = var_v
    return feats
