from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .common import RANDOM_STATE, ensure_dir


class RidgeClosedForm:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeClosedForm":
        X_aug = np.c_[np.ones(X.shape[0]), X]
        penalty = np.eye(X_aug.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(X_aug.T @ X_aug + self.alpha * penalty, X_aug.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_aug = np.c_[np.ones(X.shape[0]), X]
        return X_aug @ self.coef_


def run(output_dir: str | Path) -> list[dict]:
    output_dir = ensure_dir(output_dir)
    figures = ensure_dir(output_dir / "figures")

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "OLS_closed_form": RidgeClosedForm(alpha=0.0),
        "Ridge_alpha_0.1": RidgeClosedForm(alpha=0.1),
        "Ridge_alpha_1": RidgeClosedForm(alpha=1.0),
        "Ridge_alpha_10": RidgeClosedForm(alpha=10.0),
    }

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rows.append(
            {
                "experiment": "linear_regression",
                "dataset": "sklearn_diabetes",
                "model": name,
                "samples": len(data.target),
                "features": data.data.shape[1],
                "MSE": round(mean_squared_error(y_test, y_pred), 6),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6),
                "MAE": round(mean_absolute_error(y_test, y_pred), 6),
                "R2": round(r2_score(y_test, y_pred), 6),
            }
        )

    best_name = min(rows, key=lambda row: row["MSE"])["model"]
    _plot_prediction(y_test, predictions[best_name], best_name, figures / "linear_regression_prediction.png")
    _plot_ridge_path(X_train, y_train, figures / "linear_regression_ridge_coefficients.png")
    return rows


def _plot_prediction(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(y_true, y_pred, alpha=0.75)
    low = min(float(y_true.min()), float(y_pred.min()))
    high = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([low, high], [low, high], "r--", linewidth=1)
    ax.set_xlabel("True target")
    ax.set_ylabel("Predicted target")
    ax.set_title(f"Linear regression prediction: {model_name}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ridge_path(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    alphas = np.logspace(-3, 3, 30)
    coefs = []
    for alpha in alphas:
        model = RidgeClosedForm(alpha=alpha).fit(X, y)
        coefs.append(model.coef_[1:])
    coefs = np.asarray(coefs)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    ax.set_title("Ridge coefficient path")
    fig.savefig(path, dpi=180)
    plt.close(fig)
