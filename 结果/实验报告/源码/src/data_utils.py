from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray


def load_mat_dataset(path: str | Path) -> Dataset:
    """Load one assignment .mat dataset.

    The provided files use `fea` for samples and `gnd` for labels.
    Labels are returned only for evaluation, not for feature selection.
    """
    path = Path(path)
    mat = sio.loadmat(path)
    feature_key = "fea" if "fea" in mat else _first_array_key(mat)
    label_key = "gnd" if "gnd" in mat else _first_label_key(mat, feature_key)

    X = np.asarray(mat[feature_key], dtype=np.float64)
    y = np.asarray(mat[label_key]).reshape(-1)
    return Dataset(name=path.stem, X=X, y=y)


def iter_datasets(dataset_dir: str | Path) -> list[Dataset]:
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {dataset_dir}")
    return [load_mat_dataset(path) for path in files]


def _first_array_key(mat: dict) -> str:
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return key
    raise KeyError("No 2-D feature matrix found in .mat file.")


def _first_label_key(mat: dict, feature_key: str) -> str:
    n_samples = mat[feature_key].shape[0]
    for key, value in mat.items():
        if key.startswith("__") or key == feature_key:
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size == n_samples:
            return key
    raise KeyError("No label vector matching the sample count was found.")
