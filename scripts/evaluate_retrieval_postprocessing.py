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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from evaluate_ce_retrieval import build_model, build_top_results
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from visionsearch_fg.data import CUB200Dataset, read_image_ids
from visionsearch_fg.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from visionsearch_fg.retrieval import (
    cosine_similarity_matrix,
    l2_normalize,
    rank_gallery_for_queries,
)
from visionsearch_fg.utils import load_yaml_config

FLOAT32_BYTES = 4
VIEW_CHOICES = ("bbox", "bbox_flip", "original", "original_flip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval post-processing with TTA averaging and query expansion."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--feature", choices=["embedding", "projection"], default="embedding")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEW_CHOICES,
        default=["bbox"],
        help="Views to average. Each view is L2-normalized before averaging.",
    )
    parser.add_argument("--qe-top-k", nargs="*", type=int, default=[0])
    parser.add_argument("--qe-alpha", nargs="*", type=float, default=[0.0])
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/retrieval_postprocessing"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    device = resolve_device(args.device)

    model = build_model(model_config=config["model"], feature=args.feature).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    view_embeddings, records = extract_view_embeddings(
        config=config,
        model=model,
        split=args.split,
        ids_path=args.ids_path,
        views=args.views,
        device=device,
        feature=args.feature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    averaged_embeddings = average_view_embeddings(view_embeddings)
    labels = np.array([record["label"] for record in records], dtype=np.int64)

    postprocess_results = evaluate_postprocessing_grid(
        embeddings=averaged_embeddings,
        labels=labels,
        qe_top_ks=args.qe_top_k,
        qe_alphas=args.qe_alpha,
        latency_repeats=args.latency_repeats,
    )

    run_id = args.checkpoint.parent.name
    view_name = "_".join(args.views)
    output_dir = args.output_dir / run_id / view_name / args.feature
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", averaged_embeddings)
    write_records_csv(records, output_dir / "records.csv")
    top_results = build_top_results(
        embeddings=averaged_embeddings,
        records=records,
        top_k=10,
        metric="cosine",
    )
    (output_dir / "top_results.json").write_text(
        json.dumps(top_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    storage_bytes = int(averaged_embeddings.shape[0] * averaged_embeddings.shape[1] * FLOAT32_BYTES)
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": args.split,
        "ids_path": str(args.ids_path),
        "feature": args.feature,
        "views": args.views,
        "metric": "cosine",
        "num_samples": len(records),
        "embedding_dim": int(averaged_embeddings.shape[1]),
        "storage_bytes_float32": storage_bytes,
        "storage_mib_float32": storage_bytes / (1024 * 1024),
        "postprocessing_results": postprocess_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


@torch.no_grad()
def extract_view_embeddings(
    config: dict[str, Any],
    model: torch.nn.Module,
    split: str,
    ids_path: Path,
    views: list[str],
    device: torch.device,
    feature: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    embeddings_by_view: list[np.ndarray] = []
    reference_records: list[dict[str, Any]] | None = None

    for view in views:
        embeddings, records = extract_single_view_embeddings(
            config=config,
            model=model,
            split=split,
            ids_path=ids_path,
            view=view,
            device=device,
            feature=feature,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if reference_records is None:
            reference_records = records
        else:
            ensure_same_records(reference_records, records)
        embeddings_by_view.append(embeddings)

    if reference_records is None:
        raise ValueError("At least one view is required.")
    return embeddings_by_view, reference_records


@torch.no_grad()
def extract_single_view_embeddings(
    config: dict[str, Any],
    model: torch.nn.Module,
    split: str,
    ids_path: Path,
    view: str,
    device: torch.device,
    feature: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data_config = config["data"]
    effective_batch_size = batch_size or config["training"]["batch_size"]
    effective_num_workers = (
        num_workers if num_workers is not None else int(data_config.get("num_workers", 0))
    )
    dataset = CUB200Dataset(
        root=data_config["root"],
        split=split,
        image_ids=read_image_ids(ids_path),
        crop_mode=view_crop_mode(view),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        transform=build_view_transform(
            image_size=data_config["image_size"],
            flip=view.endswith("_flip"),
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for batch in tqdm(dataloader, desc=f"extract:{view}", leave=False):
        images = batch["image"].to(device)
        output = model(images)
        if not hasattr(output, feature):
            raise AttributeError(f"Model output does not contain feature: {feature}")
        embeddings.append(getattr(output, feature).cpu().numpy())

        batch_size = len(batch["label"])
        for index in range(batch_size):
            records.append(
                {
                    "image_id": int(batch["image_id"][index]),
                    "label": int(batch["label"][index]),
                    "class_name": batch["class_name"][index],
                    "path": batch["path"][index],
                }
            )

    return np.concatenate(embeddings, axis=0), records


def build_view_transform(image_size: int, flip: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if flip:
        steps.append(transforms.RandomHorizontalFlip(p=1.0))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def view_crop_mode(view: str) -> str:
    if view in {"bbox", "bbox_flip"}:
        return "bbox"
    if view in {"original", "original_flip"}:
        return "none"
    raise ValueError(f"Unsupported view: {view}")


def average_view_embeddings(view_embeddings: list[np.ndarray]) -> np.ndarray:
    if not view_embeddings:
        raise ValueError("view_embeddings must not be empty")
    base_shape = view_embeddings[0].shape
    if any(embeddings.shape != base_shape for embeddings in view_embeddings):
        raise ValueError("All view embeddings must have the same shape")

    normalized_views = [
        l2_normalize(embeddings.astype(np.float32)) for embeddings in view_embeddings
    ]
    averaged = np.mean(np.stack(normalized_views, axis=0), axis=0)
    return l2_normalize(averaged).astype(np.float32)


def evaluate_postprocessing_grid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    qe_top_ks: list[int],
    qe_alphas: list[float],
    latency_repeats: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    normalized_embeddings = l2_normalize(embeddings.astype(np.float32))
    base_ranked_indices = rank_gallery_for_queries(
        cosine_similarity_matrix(normalized_embeddings),
        exclude_self=True,
    )

    for top_k in qe_top_ks:
        if top_k < 0:
            raise ValueError("query expansion top-k must be greater than or equal to 0")
        for alpha in qe_alphas:
            if alpha < 0:
                raise ValueError("query expansion alpha must be greater than or equal to 0")
            if top_k == 0 or alpha == 0:
                query_embeddings = normalized_embeddings
                method = "tta_average"
            else:
                query_embeddings = query_expand_embeddings(
                    embeddings=normalized_embeddings,
                    ranked_indices=base_ranked_indices,
                    top_k=top_k,
                    alpha=alpha,
                )
                method = "query_expansion"

            metrics = evaluate_query_to_gallery(
                query_embeddings=query_embeddings,
                gallery_embeddings=normalized_embeddings,
                labels=labels,
            )
            latency = benchmark_query_to_gallery(
                query_embeddings=query_embeddings,
                gallery_embeddings=normalized_embeddings,
                repeats=latency_repeats,
            )
            results.append(
                {
                    "method": method,
                    "qe_top_k": top_k,
                    "qe_alpha": alpha,
                    **metrics,
                    **latency,
                }
            )
    return results


def query_expand_embeddings(
    embeddings: np.ndarray,
    ranked_indices: np.ndarray,
    top_k: int,
    alpha: float,
) -> np.ndarray:
    neighbor_indices = ranked_indices[:, :top_k]
    neighbor_mean = embeddings[neighbor_indices].mean(axis=1)
    expanded = embeddings + (alpha * neighbor_mean)
    return l2_normalize(expanded.astype(np.float32))


def evaluate_query_to_gallery(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    ranked_indices = rank_query_to_gallery(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        exclude_self=True,
    )
    return {
        "recall@1": recall_at_k(ranked_indices=ranked_indices, labels=labels, k=1),
        "recall@5": recall_at_k(ranked_indices=ranked_indices, labels=labels, k=5),
        "recall@10": recall_at_k(ranked_indices=ranked_indices, labels=labels, k=10),
        "mAP": mean_average_precision(ranked_indices=ranked_indices, labels=labels),
    }


def rank_query_to_gallery(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    exclude_self: bool,
) -> np.ndarray:
    if query_embeddings.shape != gallery_embeddings.shape:
        raise ValueError("query and gallery embeddings must have the same shape")
    similarity = l2_normalize(query_embeddings) @ l2_normalize(gallery_embeddings).T
    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)
        ranked_indices = np.argsort(-similarity, axis=1)
        query_indices = np.arange(similarity.shape[0])[:, None]
        return ranked_indices[ranked_indices != query_indices].reshape(similarity.shape[0], -1)
    return np.argsort(-similarity, axis=1)


def recall_at_k(ranked_indices: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = ranked_indices[:, :k]
    hits = labels[top_k] == labels[:, None]
    return float(hits.any(axis=1).mean())


def mean_average_precision(ranked_indices: np.ndarray, labels: np.ndarray) -> float:
    average_precisions = []
    for query_index, query_ranked_indices in enumerate(ranked_indices):
        relevant = labels[query_ranked_indices] == labels[query_index]
        if not relevant.any():
            average_precisions.append(0.0)
            continue
        relevant_ranks = np.flatnonzero(relevant) + 1
        precision_at_relevant = np.arange(1, relevant_ranks.size + 1) / relevant_ranks
        average_precisions.append(float(precision_at_relevant.mean()))
    return float(np.mean(average_precisions))


def benchmark_query_to_gallery(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    repeats: int,
) -> dict[str, float]:
    if repeats < 1:
        raise ValueError("latency repeats must be greater than or equal to 1")
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        rank_query_to_gallery(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            exclude_self=True,
        )
        durations.append((time.perf_counter() - start) * 1000)
    total_latency_ms = mean(durations)
    return {
        "search_latency_ms_total": total_latency_ms,
        "search_latency_ms_per_query": total_latency_ms / query_embeddings.shape[0],
    }


def ensure_same_records(
    primary_records: list[dict[str, Any]],
    secondary_records: list[dict[str, Any]],
) -> None:
    primary_ids = [record["image_id"] for record in primary_records]
    secondary_ids = [record["image_id"] for record in secondary_records]
    if primary_ids != secondary_ids:
        raise ValueError("View extraction produced different image order.")


def write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "label", "class_name", "path"])
        writer.writeheader()
        writer.writerows(records)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
