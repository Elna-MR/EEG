from dataclasses import dataclass

@dataclass
class TsEnConfig:
    fs_target: int = 200          # resample target Hz
    bands = {
        "theta": (4, 7),
        "alpha": (8, 15),
        "beta": (16, 31),
        "gamma": (32, 55),
    }
    window_sec: float = 2.0       # sliding window length (s)
    step_sec: float = 1.0         # sliding step (s)
    hist_bins: int = 64           # bins for probability estimate
    qs = (2, 3, 4)                # Tsallis q values

@dataclass
class TrainConfig:
    k_neighbors: int = 10
    n_splits: int = 5             # CV folds
    random_state: int = 42
