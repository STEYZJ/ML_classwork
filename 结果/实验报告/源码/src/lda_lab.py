from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import RANDOM_STATE, ensure_dir, load_feature_label_mat


class BinaryFisherLDA:
    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.classes_: np.ndarray | None = None
        self.w_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryFisherLDA":
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("BinaryFisherLDA only supports two classes.")
        X1 = X[y == classes[0]]
        X2 = X[y == classes[1]]
        m1, m2 = X1.mean(axis=0), X2.mean(axis=0)
        S1 = (X1 - m1).T @ (X1 - m1)
        S2 = (X2 - m2).T @ (X2 - m2)
        Sw = S1 + S2 + self.ridge * np.eye(X.shape[1])
        self.w_ = np.linalg.solve(Sw, m1 - m2)
        self.threshold_ = float(0.5 * ((m1 @ self.w_) + (m2 @ self.w_)))
        self.classes_ = classes
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None or self.threshold_ is None or self.classes_ is None:
            raise RuntimeError("Model has not been fitted.")
        scores = X @ self.w_
        return np.where(scores >= self.threshold_, self.classes_[0], self.classes_[1])


def run(experiment_root: str | Path, output_dir: str | Path) -> tuple[list[dict], list[dict]]:
    output_dir = ensure_dir(output_dir)
    figures = ensure_dir(output_dir / "figures")
    data_dir = Path(experiment_root) / "LDA实验" / "LDA实验"
    datasets = {
        "Iris": data_dir / "iris.mat",
        "Sonar": data_dir / "sonar.mat",
    }

    rows: list[dict] = []
    for dataset_name, path in datasets.items():
        X, y = load_feature_label_mat(path)
        rows.extend(_evaluate_dataset(dataset_name, X.astype(float), y))
        if dataset_name == "Iris":
            _plot_iris_projection(X.astype(float), y, figures / "lda_iris_projection.png")

    sonar_X, sonar_y = load_feature_label_mat(datasets["Sonar"])
    feature_curve = _sonar_feature_curve(sonar_X.astype(float), sonar_y)
    _plot_feature_curve(feature_curve, figures / "lda_sonar_feature_curve.png")
    return rows, feature_curve


def _evaluate_dataset(dataset_name: str, X: np.ndarray, y: np.ndarray) -> list[dict]:
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    rows = []

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipe.fit(X_train, y_train)
    holdout_acc = accuracy_score(y_test, pipe.predict(X_test))
    rows.append(_row(dataset_name, "sklearn_LDA_holdout_7_3", holdout_acc, X.shape, np.unique(y).size))

    kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    kfold_acc = cross_val_score(pipe, X, y, cv=kfold, scoring="accuracy").mean()
    rows.append(_row(dataset_name, "sklearn_LDA_10_fold", kfold_acc, X.shape, np.unique(y).size))

    loo_acc = cross_val_score(pipe, X, y, cv=LeaveOneOut(), scoring="accuracy").mean()
    rows.append(_row(dataset_name, "sklearn_LDA_leave_one_out", loo_acc, X.shape, np.unique(y).size))

    if np.unique(y).size == 2:
        fisher_accs = []
        for train_idx, test_idx in kfold.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = BinaryFisherLDA().fit(X_train, y[train_idx])
            fisher_accs.append(accuracy_score(y[test_idx], clf.predict(X_test)))
        rows.append(_row(dataset_name, "custom_Fisher_LDA_10_fold", float(np.mean(fisher_accs)), X.shape, 2))
    return rows


def _row(dataset: str, method: str, acc: float, shape: tuple[int, int], classes: int) -> dict:
    return {
        "experiment": "lda",
        "dataset": dataset,
        "method": method,
        "samples": shape[0],
        "features": shape[1],
        "classes": classes,
        "ACC": round(float(acc), 6),
    }


def _sonar_feature_curve(X: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for k in [5, 10, 20, 40, X.shape[1]]:
        pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        acc = cross_val_score(pipe, X[:, :k], y, cv=cv, scoring="accuracy").mean()
        rows.append(
            {
                "experiment": "lda_sonar_feature_curve",
                "dataset": "Sonar",
                "selected_features": k,
                "ACC": round(float(acc), 6),
            }
        )
    return rows


def _plot_feature_curve(rows: list[dict], path: Path) -> None:
    xs = [row["selected_features"] for row in rows]
    ys = [row["ACC"] for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel("10-fold ACC")
    ax.set_title("Sonar LDA feature-count curve")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_iris_projection(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))
    Z = pipe.fit_transform(X, y)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for label in np.unique(y):
        mask = y == label
        ax.scatter(Z[mask, 0], Z[mask, 1], label=f"class {int(label)}", alpha=0.8)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title("Iris LDA projection")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
