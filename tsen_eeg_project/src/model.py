from typing import Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def train_eval_knn(X: np.ndarray, y: np.ndarray, k: int = 10, n_splits: int = 5, random_state: int = 42) -> Tuple[float, float]:
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted").mean()
    return float(acc), float(f1)
