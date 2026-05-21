from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.common import ensure_dir, save_rows
from src import lda_lab, linear_regression_lab, neural_network_lab, svm_face_lab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all four machine-learning labs.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("../../../实验"),
        help="Directory containing the original four experiment folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))

    summary: dict[str, object] = {}

    print("Running experiment 1: linear/ridge regression")
    linear_rows = linear_regression_lab.run(out)
    save_rows(linear_rows, out / "linear_regression_metrics.csv")
    summary["linear_regression"] = linear_rows

    print("Running experiment 2: LDA")
    lda_rows, lda_feature_rows = lda_lab.run(args.experiment_root, out)
    save_rows(lda_rows, out / "lda_metrics.csv")
    save_rows(lda_feature_rows, out / "lda_sonar_feature_curve.csv")
    summary["lda"] = lda_rows
    summary["lda_sonar_feature_curve"] = lda_feature_rows

    print("Running experiment 3: SVM face recognition")
    svm_rows, svm_report = svm_face_lab.run(args.experiment_root, out)
    save_rows(svm_rows, out / "svm_face_metrics.csv")
    (out / "svm_face_classification_report.txt").write_text(svm_report, encoding="utf-8")
    summary["svm_face_recognition"] = svm_rows

    print("Running experiment 4: neural network classifier")
    nn_rows = neural_network_lab.run(args.experiment_root, out)
    save_rows(nn_rows, out / "neural_network_metrics.csv")
    summary["neural_network"] = nn_rows

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Outputs written to {out}")


if __name__ == "__main__":
    main()
