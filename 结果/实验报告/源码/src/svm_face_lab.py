from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import RANDOM_STATE, ensure_dir, load_feature_label_mat, plot_confusion_matrix


def run(experiment_root: str | Path, output_dir: str | Path) -> tuple[list[dict], str]:
    output_dir = ensure_dir(output_dir)
    figures = ensure_dir(output_dir / "figures")
    path = (
        Path(experiment_root)
        / "第四次实验-神经网络"
        / "第四次实验-神经网络"
        / "datasets"
        / "Yale.mat"
    )
    X, y = load_feature_label_mat(path)
    X = X.astype(float) / 255.0
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Keep the PCA dimension below the smallest fold size in 3-fold CV.
    n_components = min(50, X_train.shape[0] - 1, X_train.shape[1])
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE),
        SVC(kernel="rbf", class_weight="balanced"),
    )
    grid = GridSearchCV(
        pipe,
        {
            "svc__C": [1, 5, 10, 50],
            "svc__gamma": ["scale", 0.001, 0.01],
        },
        cv=3,
        scoring="accuracy",
        n_jobs=None,
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    labels = sorted(np.unique(y).tolist())
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title="SVM face recognition confusion matrix",
        path=figures / "svm_yale_confusion_matrix.png",
    )
    _plot_prediction_examples(X_test, y_test, y_pred, figures / "svm_yale_predictions.png")

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    rows = [
        {
            "experiment": "svm_face_recognition",
            "dataset": "Yale",
            "samples": X.shape[0],
            "features": X.shape[1],
            "classes": len(labels),
            "pca_components": n_components,
            "best_params": str(grid.best_params_),
            "cv_best_ACC": round(float(grid.best_score_), 6),
            "test_ACC": round(float(acc), 6),
        }
    ]
    return rows, report


def _plot_prediction_examples(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    count = min(20, X.shape[0])
    fig, axes = plt.subplots(4, 5, figsize=(9, 7), constrained_layout=True)
    for ax, image, truth, pred in zip(axes.ravel(), X[:count], y_true[:count], y_pred[:count]):
        ax.imshow(image.reshape(32, 32), cmap="gray")
        color = "black" if truth == pred else "red"
        ax.set_title(f"T:{truth} P:{pred}", color=color, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Yale face recognition examples")
    fig.savefig(path, dpi=180)
    plt.close(fig)
