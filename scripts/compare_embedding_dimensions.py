from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean
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

FLOAT32_BYTES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PCA-compressed embedding dimensions for retrieval."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[512, 256, 128, 64])
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/embedding_dimension"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = np.load(args.embeddings).astype(np.float32)
    records = read_records_csv(args.records)
    labels = np.array([record["label"] for record in records], dtype=np.int64)

    if embeddings.shape[0] != len(records):
        raise ValueError("embeddings and records must contain the same number of samples")
    if args.latency_repeats < 1:
        raise ValueError("--latency-repeats must be greater than or equal to 1")

    pca = fit_pca(embeddings)
    rows = []
    for dimension in args.dimensions:
        if dimension < 1 or dimension > embeddings.shape[1]:
            raise ValueError(f"dimension must be in [1, {embeddings.shape[1]}]: {dimension}")

        if dimension == embeddings.shape[1]:
            variant_embeddings = embeddings
            projection = "original"
        else:
            variant_embeddings = transform_pca(embeddings, pca, n_components=dimension)
            projection = "pca"

        rows.append(
            evaluate_dimension(
                embeddings=variant_embeddings,
                labels=labels,
                projection=projection,
                explained_variance_ratio=explained_variance_ratio(pca, dimension),
                latency_repeats=args.latency_repeats,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "embeddings": str(args.embeddings),
        "records": str(args.records),
        "controlled_variable": "same saved embeddings, PCA reducer, L2-normalized cosine retrieval",
        "changed_variable": "embedding dimension",
        "latency_repeats": args.latency_repeats,
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.output_dir / "summary.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def fit_pca(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    mean_vector = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean_vector
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    return {
        "mean": mean_vector.astype(np.float32),
        "components": components.astype(np.float32),
        "singular_values": singular_values.astype(np.float64),
    }


def transform_pca(
    embeddings: np.ndarray,
    pca: dict[str, np.ndarray],
    n_components: int,
) -> np.ndarray:
    centered = embeddings - pca["mean"]
    return (centered @ pca["components"][:n_components].T).astype(np.float32)


def explained_variance_ratio(pca: dict[str, np.ndarray], n_components: int) -> float:
    singular_values = pca["singular_values"]
    eigenvalues = singular_values * singular_values
    return float(eigenvalues[:n_components].sum() / eigenvalues.sum())


def evaluate_dimension(
    embeddings: np.ndarray,
    labels: np.ndarray,
    projection: str,
    explained_variance_ratio: float,
    latency_repeats: int,
) -> dict[str, Any]:
    normalized_embeddings = l2_normalize(embeddings.astype(np.float32))
    similarity = dot_product_similarity_matrix(normalized_embeddings)
    ranked_indices = rank_gallery_for_queries(similarity, exclude_self=True)
    latency = benchmark_search(normalized_embeddings, latency_repeats)
    storage_bytes = int(
        normalized_embeddings.shape[0] * normalized_embeddings.shape[1] * FLOAT32_BYTES
    )

    return {
        "dimension": int(normalized_embeddings.shape[1]),
        "projection": projection,
        "num_samples": int(normalized_embeddings.shape[0]),
        "embedding_dim": int(normalized_embeddings.shape[1]),
        "explained_variance_ratio": explained_variance_ratio,
        "storage_bytes_float32": storage_bytes,
        "storage_mib_float32": storage_bytes / (1024 * 1024),
        "search_latency_ms_total": latency["search_latency_ms_total"],
        "search_latency_ms_per_query": latency["search_latency_ms_per_query"],
        "recall@1": recall_at_k(ranked_indices, labels, k=1),
        "recall@5": recall_at_k(ranked_indices, labels, k=5),
        "recall@10": recall_at_k(ranked_indices, labels, k=10),
        "mAP": mean_average_precision(ranked_indices, labels),
    }


def benchmark_search(embeddings: np.ndarray, repeats: int) -> dict[str, float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        similarity = dot_product_similarity_matrix(embeddings)
        rank_gallery_for_queries(similarity, exclude_self=True)
        durations.append((time.perf_counter() - start) * 1000)

    total_latency_ms = mean(durations)
    return {
        "search_latency_ms_total": total_latency_ms,
        "search_latency_ms_per_query": total_latency_ms / embeddings.shape[0],
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
        "dimension",
        "projection",
        "num_samples",
        "embedding_dim",
        "explained_variance_ratio",
        "storage_bytes_float32",
        "storage_mib_float32",
        "search_latency_ms_total",
        "search_latency_ms_per_query",
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
