from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from visionsearch_fg.retrieval import (
    dot_product_similarity_matrix,
    l2_normalize,
    mean_average_precision,
    rank_gallery_for_queries,
    recall_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw and L2-normalized embeddings for retrieval."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/embedding_normalization"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = np.load(args.embeddings)
    records = read_records_csv(args.records)
    labels = np.array([record["label"] for record in records], dtype=np.int64)

    if embeddings.shape[0] != len(records):
        raise ValueError("embeddings and records must contain the same number of samples")

    rows = [
        evaluate_variant(
            name="raw_dot_product",
            embeddings=embeddings,
            labels=labels,
            description="Raw embedding with dot-product similarity.",
        ),
        evaluate_variant(
            name="l2_normalized_cosine",
            embeddings=l2_normalize(embeddings),
            labels=labels,
            description=(
                "L2-normalized embedding with dot-product similarity, equivalent to cosine."
            ),
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "embeddings": str(args.embeddings),
        "records": str(args.records),
        "controlled_variable": "same saved embeddings and same query/gallery protocol",
        "changed_variable": "whether embeddings are L2-normalized before similarity",
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.output_dir / "summary.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def evaluate_variant(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    description: str,
) -> dict[str, Any]:
    similarity = dot_product_similarity_matrix(embeddings)
    ranked_indices = rank_gallery_for_queries(similarity, exclude_self=True)
    return {
        "variant": name,
        "description": description,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "recall@1": recall_at_k(ranked_indices, labels, k=1),
        "recall@5": recall_at_k(ranked_indices, labels, k=5),
        "recall@10": recall_at_k(ranked_indices, labels, k=10),
        "mAP": mean_average_precision(ranked_indices, labels),
    }


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
        "variant",
        "num_samples",
        "embedding_dim",
        "recall@1",
        "recall@5",
        "recall@10",
        "mAP",
        "description",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
