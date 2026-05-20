from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


def cluster_and_score(
    X: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    n_init: int = 20,
) -> dict[str, float]:
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    pred = model.fit_predict(X)
    true = LabelEncoder().fit_transform(y_true)
    mapped = _hungarian_map(true, pred)
    return {
        "ACC": clustering_accuracy(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
        "ARI": adjusted_rand_score(true, pred),
        "F1": f1_score(true, mapped, average="macro"),
    }


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    size = max(y_true.max(), y_pred.max()) + 1
    weight = np.zeros((size, size), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        weight[pred, truth] += 1
    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    return float(weight[row_ind, col_ind].sum() / y_true.size)


def _hungarian_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    size = max(y_true.max(), y_pred.max()) + 1
    weight = np.zeros((size, size), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        weight[pred, truth] += 1
    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, 0) for label in y_pred], dtype=np.int64)
