from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from visionsearch_fg.retrieval import evaluate_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval distance metrics on the same saved embeddings."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["cosine", "euclidean"],
        default=["cosine", "euclidean"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/similarity_metric_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = np.load(args.embeddings)
    records = read_records_csv(args.records)
    labels = np.array([record["label"] for record in records], dtype=np.int64)

    if embeddings.shape[0] != len(records):
        raise ValueError("embeddings and records must contain the same number of samples")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in args.metrics:
        metric_values = evaluate_retrieval(
            embeddings=embeddings,
            labels=labels,
            recall_ks=(1, 5, 10),
            metric=metric,
        )
        rows.append(
            {
                "metric": metric,
                "num_samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                **metric_values,
            }
        )

    summary = {
        "embeddings": str(args.embeddings),
        "records": str(args.records),
        "controlled_variable": "same saved embeddings",
        "changed_variable": "retrieval distance metric",
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.output_dir / "summary.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def read_records_csv(records_path: Path) -> list[dict[str, Any]]:
    with records_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        records = []
        for row in reader:
            records.append(
                {
                    "image_id": int(row["image_id"]),
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "path": row["path"],
                }
            )
    return records


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "metric",
        "num_samples",
        "embedding_dim",
        "recall@1",
        "recall@5",
        "recall@10",
        "mAP",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
