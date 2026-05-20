from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.data_utils import iter_datasets
from src.evaluation import cluster_and_score
from src.feature_selection import (
    build_knn_graph,
    graph_sparse_regression_ranking,
    laplacian_score_ranking,
    standardize,
    variance_ranking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unsupervised feature selection on the assignment datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../../../dataset/dataset"),
        help="Directory containing .mat datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory used for result CSV/JSON/figures.",
    )
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplcache"))

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    rankings_for_json: dict[str, dict[str, list[int]]] = {}

    for dataset in iter_datasets(args.dataset_dir):
        X_raw, y = dataset.X, dataset.y
        X = standardize(X_raw)
        n_samples, n_features = X.shape
        n_classes = int(np.unique(y).size)
        counts = sorted({k for k in args.feature_counts if 1 <= k <= n_features})

        print(f"[{dataset.name}] samples={n_samples}, features={n_features}, classes={n_classes}")
        graph = build_knn_graph(X, n_neighbors=args.knn)
        proposed = graph_sparse_regression_ranking(
            X,
            graph,
            n_components=n_classes,
            alpha=args.alpha,
            beta=args.beta,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        rankings = {
            "SparseGraphRegression": proposed.ranking,
            "LaplacianScore": laplacian_score_ranking(X, graph),
            "Variance": variance_ranking(X),
        }
        rankings_for_json[dataset.name] = {
            name: [int(idx) for idx in ranking[: min(200, n_features)]]
            for name, ranking in rankings.items()
        }

        summaries.append(
            {
                "dataset": dataset.name,
                "samples": n_samples,
                "features": n_features,
                "classes": n_classes,
                "graph_edges": int(graph.nnz),
                "proposed_iterations": proposed.n_iter,
                "final_objective": proposed.objectives[-1] if proposed.objectives else np.nan,
            }
        )

        all_feature_metrics = cluster_and_score(X, y, n_clusters=n_classes, seed=args.seed)
        rows.append(
            _row(
                dataset.name,
                "AllFeatures",
                n_features,
                n_samples,
                n_features,
                n_classes,
                all_feature_metrics,
            )
        )

        for method, ranking in rankings.items():
            for k in counts:
                selected = ranking[:k]
                metrics = cluster_and_score(X[:, selected], y, n_clusters=n_classes, seed=args.seed)
                rows.append(_row(dataset.name, method, k, n_samples, n_features, n_classes, metrics))

    _write_csv(args.output_dir / "results.csv", rows)
    _write_csv(args.output_dir / "dataset_summary.csv", summaries)
    (args.output_dir / "selected_feature_rankings.json").write_text(
        json.dumps(rankings_for_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_results(args.output_dir / "results.csv", args.output_dir / "acc_by_dataset.png")
    print(f"Done. Results written to {args.output_dir}")


def _row(
    dataset: str,
    method: str,
    k: int,
    n_samples: int,
    n_features: int,
    n_classes: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "selected_features": k,
        "samples": n_samples,
        "features": n_features,
        "classes": n_classes,
        **{key: round(value, 6) for key, value in metrics.items()},
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(csv_path: Path, out_path: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv(csv_path)
    proposed = data[data["method"].isin(["AllFeatures", "SparseGraphRegression"])]
    if proposed.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method, group in proposed.groupby("method"):
        pivot = group.pivot_table(index="dataset", columns="selected_features", values="ACC")
        pivot.plot(kind="bar", ax=axes[0], alpha=0.85, title=f"{method} ACC")
    axes[0].set_ylabel("ACC")
    axes[0].legend(title="features", fontsize=8)

    best = data.sort_values("ACC").groupby(["dataset", "method"]).tail(1)
    best.pivot(index="dataset", columns="method", values="ACC").plot(
        kind="bar", ax=axes[1], title="Best ACC by method", alpha=0.85
    )
    axes[1].set_ylabel("ACC")
    axes[1].legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
