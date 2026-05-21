from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import RANDOM_STATE, ensure_dir, load_feature_label_mat, plot_confusion_matrix


def run(experiment_root: str | Path, output_dir: str | Path) -> list[dict]:
    output_dir = ensure_dir(output_dir)
    figures = ensure_dir(output_dir / "figures")
    data_dir = (
        Path(experiment_root)
        / "第四次实验-神经网络"
        / "第四次实验-神经网络"
        / "datasets"
    )
    settings = {
        "MNIST": {
            "path": data_dir / "MNIST.mat",
            "hidden": (128,),
            "max_iter": 120,
        },
        "Yale": {
            "path": data_dir / "Yale.mat",
            "hidden": (64,),
            "max_iter": 200,
        },
        "lung": {
            "path": data_dir / "lung.mat",
            "hidden": (48,),
            "max_iter": 200,
        },
    }

    rows = []
    for name, cfg in settings.items():
        X, y = load_feature_label_mat(cfg["path"])
        X = X.astype(float)
        y = y.astype(int)
        if X.max() > 10:
            X = X / 255.0

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=cfg["hidden"],
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=min(64, X_train.shape[0]),
                learning_rate_init=1e-3,
                max_iter=cfg["max_iter"],
                early_stopping=True,
                n_iter_no_change=12,
                random_state=RANDOM_STATE,
            ),
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        mlp = clf.named_steps["mlpclassifier"]
        rows.append(
            {
                "experiment": "neural_network",
                "dataset": name,
                "samples": X.shape[0],
                "features": X.shape[1],
                "classes": int(np.unique(y).size),
                "hidden_layers": str(cfg["hidden"]),
                "iterations": int(mlp.n_iter_),
                "loss": round(float(mlp.loss_), 6),
                "test_ACC": round(float(acc), 6),
            }
        )

        labels = sorted(np.unique(y).tolist())
        plot_confusion_matrix(
            y_test,
            pred,
            labels=labels,
            title=f"{name} MLP confusion matrix",
            path=figures / f"nn_{name.lower()}_confusion_matrix.png",
        )
        if name in {"MNIST", "Yale"}:
            _plot_examples(name, X_test, y_test, pred, figures / f"nn_{name.lower()}_examples.png")
    _plot_summary(rows, figures / "nn_accuracy_summary.png")
    return rows


def _plot_examples(dataset: str, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    count = min(20, X.shape[0])
    side = 28 if dataset == "MNIST" else 32
    fig, axes = plt.subplots(4, 5, figsize=(9, 7), constrained_layout=True)
    for ax, image, truth, pred in zip(axes.ravel(), X[:count], y_true[:count], y_pred[:count]):
        ax.imshow(image.reshape(side, side), cmap="gray")
        color = "black" if truth == pred else "red"
        ax.set_title(f"T:{truth} P:{pred}", color=color, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{dataset} neural-network predictions")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    names = [row["dataset"] for row in rows]
    accs = [row["test_ACC"] for row in rows]
    ax.bar(names, accs, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test ACC")
    ax.set_title("Neural network accuracy")
    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
