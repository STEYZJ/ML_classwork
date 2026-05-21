from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parent / "outputs" / ".mplcache"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

from src.common import ensure_dir, save_rows
from src.svm_face_lab import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lab 3: SVM face recognition.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("../../../../实验"),
        help="Directory containing the original four experiment folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)

    rows, report = run(args.experiment_root, out)
    save_rows(rows, out / "svm_face_metrics.csv")
    (out / "svm_face_classification_report.txt").write_text(report, encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"svm_face_recognition": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs written to {out}")


if __name__ == "__main__":
    main()
