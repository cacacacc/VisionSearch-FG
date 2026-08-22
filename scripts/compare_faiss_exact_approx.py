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

from visionsearch_fg.retrieval import l2_normalize

RECALL_KS = (1, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FAISS exact and approximate retrieval backends."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--ivf-nlist", type=int, default=32)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/faiss_exact_vs_approx"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.latency_repeats < 1:
        raise ValueError("--latency-repeats must be greater than or equal to 1")

    faiss = import_faiss()
    embeddings = np.load(args.embeddings).astype(np.float32)
    records = read_records_csv(args.records)
    labels = np.array([record["label"] for record in records], dtype=np.int64)

    if embeddings.shape[0] != len(records):
        raise ValueError("embeddings and records must contain the same number of samples")

    normalized_embeddings = np.ascontiguousarray(l2_normalize(embeddings).astype(np.float32))
    max_k = max(RECALL_KS)
    exact_index, exact_indexing_ms = build_flat_index(faiss, normalized_embeddings)
    exact_ranked = search_ranked(
        index=exact_index,
        embeddings=normalized_embeddings,
        search_k=max_k,
    )

    rows = [
        evaluate_index(
            faiss=faiss,
            name="flat_exact",
            index=exact_index,
            indexing_time_ms=exact_indexing_ms,
            embeddings=normalized_embeddings,
            labels=labels,
            exact_ranked_indices=exact_ranked,
            latency_repeats=args.latency_repeats,
            search_k=max_k,
            parameters={"index": "IndexFlatIP"},
        )
    ]

    for nprobe in (1, 4, 8):
        index, indexing_ms = build_ivf_index(
            faiss=faiss,
            embeddings=normalized_embeddings,
            nlist=args.ivf_nlist,
            nprobe=nprobe,
        )
        rows.append(
            evaluate_index(
                faiss=faiss,
                name=f"ivf_nlist{args.ivf_nlist}_nprobe{nprobe}",
                index=index,
                indexing_time_ms=indexing_ms,
                embeddings=normalized_embeddings,
                labels=labels,
                exact_ranked_indices=exact_ranked,
                latency_repeats=args.latency_repeats,
                search_k=max_k,
                parameters={"index": "IndexIVFFlat", "nlist": args.ivf_nlist, "nprobe": nprobe},
            )
        )

    for ef_search in (16, 32, 64):
        index, indexing_ms = build_hnsw_index(
            faiss=faiss,
            embeddings=normalized_embeddings,
            hnsw_m=args.hnsw_m,
            ef_search=ef_search,
        )
        rows.append(
            evaluate_index(
                faiss=faiss,
                name=f"hnsw_m{args.hnsw_m}_ef{ef_search}",
                index=index,
                indexing_time_ms=indexing_ms,
                embeddings=normalized_embeddings,
                labels=labels,
                exact_ranked_indices=exact_ranked,
                latency_repeats=args.latency_repeats,
                search_k=max_k,
                parameters={"index": "IndexHNSWFlat", "M": args.hnsw_m, "efSearch": ef_search},
            )
        )

    summary = {
        "embeddings": str(args.embeddings),
        "records": str(args.records),
        "controlled_variable": "same L2-normalized embeddings and same query/gallery protocol",
        "changed_variable": "FAISS search index",
        "latency_repeats": args.latency_repeats,
        "search_recall_reference": "flat_exact top-k neighbors after excluding query itself",
        "results": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(rows, args.output_dir / "summary.csv")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")


def build_flat_index(faiss: Any, embeddings: np.ndarray) -> tuple[Any, float]:
    start = time.perf_counter()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, (time.perf_counter() - start) * 1000


def build_ivf_index(
    faiss: Any,
    embeddings: np.ndarray,
    nlist: int,
    nprobe: int,
) -> tuple[Any, float]:
    start = time.perf_counter()
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = nprobe
    return index, (time.perf_counter() - start) * 1000


def build_hnsw_index(
    faiss: Any,
    embeddings: np.ndarray,
    hnsw_m: int,
    ef_search: int,
) -> tuple[Any, float]:
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(embeddings.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.add(embeddings)
    index.hnsw.efSearch = ef_search
    return index, (time.perf_counter() - start) * 1000


def evaluate_index(
    faiss: Any,
    name: str,
    index: Any,
    indexing_time_ms: float,
    embeddings: np.ndarray,
    labels: np.ndarray,
    exact_ranked_indices: np.ndarray,
    latency_repeats: int,
    search_k: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    ranked_indices = search_ranked(index=index, embeddings=embeddings, search_k=search_k)
    latency = benchmark_search(
        index=index,
        embeddings=embeddings,
        search_k=search_k,
        repeats=latency_repeats,
    )
    row = {
        "method": name,
        "parameters": parameters,
        "indexing_time_ms": indexing_time_ms,
        "search_latency_ms_total": latency["search_latency_ms_total"],
        "search_latency_ms_per_query": latency["search_latency_ms_per_query"],
        "index_memory_bytes": serialized_index_size_bytes(faiss, index),
        "index_memory_mib": serialized_index_size_bytes(faiss, index) / (1024 * 1024),
    }

    for k in RECALL_KS:
        row[f"search_recall@{k}"] = search_recall_at_k(
            exact_ranked_indices=exact_ranked_indices,
            approximate_ranked_indices=ranked_indices,
            k=k,
        )
        row[f"retrieval_recall@{k}"] = retrieval_recall_at_k(
            ranked_indices=ranked_indices,
            labels=labels,
            k=k,
        )
    return row


def search_ranked(index: Any, embeddings: np.ndarray, search_k: int) -> np.ndarray:
    _, raw_indices = index.search(embeddings, search_k + 1)
    return remove_query_self_and_trim(raw_indices=raw_indices, search_k=search_k)


def remove_query_self_and_trim(raw_indices: np.ndarray, search_k: int) -> np.ndarray:
    cleaned = np.full((raw_indices.shape[0], search_k), fill_value=-1, dtype=np.int64)
    for query_index, row in enumerate(raw_indices):
        neighbors = [int(index) for index in row if index >= 0 and index != query_index]
        cleaned[query_index, : min(search_k, len(neighbors))] = neighbors[:search_k]
    return cleaned


def benchmark_search(
    index: Any,
    embeddings: np.ndarray,
    search_k: int,
    repeats: int,
) -> dict[str, float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        index.search(embeddings, search_k + 1)
        durations.append((time.perf_counter() - start) * 1000)

    total_latency_ms = mean(durations)
    return {
        "search_latency_ms_total": total_latency_ms,
        "search_latency_ms_per_query": total_latency_ms / embeddings.shape[0],
    }


def search_recall_at_k(
    exact_ranked_indices: np.ndarray,
    approximate_ranked_indices: np.ndarray,
    k: int,
) -> float:
    recalls = []
    paired_rows = zip(
        exact_ranked_indices[:, :k],
        approximate_ranked_indices[:, :k],
        strict=True,
    )
    for exact_row, approximate_row in paired_rows:
        exact_neighbors = {int(index) for index in exact_row if index >= 0}
        approximate_neighbors = {int(index) for index in approximate_row if index >= 0}
        recalls.append(len(exact_neighbors & approximate_neighbors) / max(1, len(exact_neighbors)))
    return float(np.mean(recalls))


def retrieval_recall_at_k(ranked_indices: np.ndarray, labels: np.ndarray, k: int) -> float:
    hits = []
    for query_index, row in enumerate(ranked_indices[:, :k]):
        valid_indices = row[row >= 0]
        if valid_indices.size == 0:
            hits.append(False)
        else:
            hits.append(bool(np.any(labels[valid_indices] == labels[query_index])))
    return float(np.mean(hits))


def serialized_index_size_bytes(faiss: Any, index: Any) -> int:
    return int(faiss.serialize_index(index).size)


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
        "index_memory_bytes",
        "index_memory_mib",
        "search_recall@1",
        "search_recall@5",
        "search_recall@10",
        "retrieval_recall@1",
        "retrieval_recall@5",
        "retrieval_recall@10",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
