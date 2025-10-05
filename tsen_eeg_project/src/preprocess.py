from typing import Tuple
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(data: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)

def extract_band(data: np.ndarray, fs: int, band: Tuple[float, float]) -> np.ndarray:
    return bandpass(data, fs, band[0], band[1])
