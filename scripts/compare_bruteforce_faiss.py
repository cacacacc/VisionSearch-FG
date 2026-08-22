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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare brute-force exact retrieval with FAISS exact search."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/faiss_exact_search"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.latency_repeats < 1:
        raise ValueError("--latency-repeats must be greater than or equal to 1")

    embeddings = np.load(args.embeddings).astype(np.float32)
    records = read_records_csv(args.records)
    labels = np.array([record["label"] for record in records], dtype=np.int64)

    if embeddings.shape[0] != len(records):
        raise ValueError("embeddings and records must contain the same number of samples")

    normalized_embeddings = np.ascontiguousarray(l2_normalize(embeddings).astype(np.float32))
    brute_force = evaluate_brute_force(normalized_embeddings, labels, args.latency_repeats)
    faiss_exact = evaluate_faiss_exact(normalized_embeddings, labels, args.latency_repeats)

    summary = {
        "embeddings": str(args.embeddings),
        "records": str(args.records),
        "controlled_variable": "same L2-normalized embeddings and same exact inner-product search",
        "changed_variable": "search backend",
        "latency_repeats": args.latency_repeats,
        "rankings_identical": bool(
            np.array_equal(brute_force.pop("ranked_indices"), faiss_exact.pop("ranked_indices"))
        ),
        "metric_delta": metric_delta(brute_force, faiss_exact),
        "results": [brute_force, faiss_exact],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(summary["results"], args.output_dir / "summary.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def evaluate_brute_force(
    embeddings: np.ndarray,
    labels: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    similarity = dot_product_similarity_matrix(embeddings)
    ranked_indices = rank_gallery_for_queries(similarity, exclude_self=True)
    latency = benchmark_brute_force(embeddings, repeats)
    metrics = retrieval_metrics(ranked_indices, labels)
    return {
        "method": "brute_force_numpy",
        "indexing_time_ms": 0.0,
        "search_latency_ms_total": latency["search_latency_ms_total"],
        "search_latency_ms_per_query": latency["search_latency_ms_per_query"],
        "ranked_indices": ranked_indices,
        **metrics,
    }


def evaluate_faiss_exact(
    embeddings: np.ndarray,
    labels: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    faiss = import_faiss()

    index_start = time.perf_counter()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    indexing_time_ms = (time.perf_counter() - index_start) * 1000

    _, raw_indices = index.search(embeddings, embeddings.shape[0])
    ranked_indices = remove_query_self(raw_indices)
    latency = benchmark_faiss_search(index, embeddings, repeats)
    metrics = retrieval_metrics(ranked_indices, labels)
    return {
        "method": "faiss_index_flat_ip",
        "indexing_time_ms": indexing_time_ms,
        "search_latency_ms_total": latency["search_latency_ms_total"],
        "search_latency_ms_per_query": latency["search_latency_ms_per_query"],
        "ranked_indices": ranked_indices,
        **metrics,
    }


def benchmark_brute_force(embeddings: np.ndarray, repeats: int) -> dict[str, float]:
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


def benchmark_faiss_search(index: Any, embeddings: np.ndarray, repeats: int) -> dict[str, float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        index.search(embeddings, embeddings.shape[0])
        durations.append((time.perf_counter() - start) * 1000)

    total_latency_ms = mean(durations)
    return {
        "search_latency_ms_total": total_latency_ms,
        "search_latency_ms_per_query": total_latency_ms / embeddings.shape[0],
    }


def retrieval_metrics(ranked_indices: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked_indices, labels, k=1),
        "recall@5": recall_at_k(ranked_indices, labels, k=5),
        "recall@10": recall_at_k(ranked_indices, labels, k=10),
        "mAP": mean_average_precision(ranked_indices, labels),
    }


def remove_query_self(raw_indices: np.ndarray) -> np.ndarray:
    query_indices = np.arange(raw_indices.shape[0])[:, None]
    return raw_indices[raw_indices != query_indices].reshape(raw_indices.shape[0], -1)


def metric_delta(brute_force: dict[str, Any], faiss_exact: dict[str, Any]) -> dict[str, float]:
    metric_names = ["recall@1", "recall@5", "recall@10", "mAP"]
    return {
        metric_name: float(faiss_exact[metric_name] - brute_force[metric_name])
        for metric_name in metric_names
    }


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise ImportError(
            "FAISS is not installed. Install it with: pip install faiss-cpu"
        ) from error
    return faiss


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
        "method",
        "indexing_time_ms",
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
