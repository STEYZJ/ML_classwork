from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import ConfusionMatrixDisplay


RANDOM_STATE = 42


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rows(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def load_feature_label_mat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(path)
    if "X" in mat and "Y" in mat:
        X = mat["X"]
        y = mat["Y"].reshape(-1)
        return np.asarray(X), np.asarray(y)
    if "Iris" in mat:
        data = np.asarray(mat["Iris"], dtype=float)
        return data[:, 1:], data[:, 0].astype(int)
    if "Sonar" in mat:
        data = np.asarray(mat["Sonar"], dtype=float)
        return data[:, 1:], data[:, 0].astype(int)
    if "fea" in mat and "gnd" in mat:
        return np.asarray(mat["fea"]), np.asarray(mat["gnd"]).reshape(-1)
    raise KeyError(f"Unknown .mat schema in {path}")


def plot_confusion_matrix(y_true, y_pred, labels, title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true).reshape(-1) == np.asarray(y_pred).reshape(-1)))
