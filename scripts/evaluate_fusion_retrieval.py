from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_ce_retrieval import (
    benchmark_search,
    build_model,
    build_top_results,
    extract_embeddings,
)
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.retrieval import evaluate_retrieval, l2_normalize
from visionsearch_fg.utils import load_yaml_config

FLOAT32_BYTES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval with fused embeddings from two checkpoints."
    )
    parser.add_argument("--primary-config", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--secondary-config", type=Path, required=True)
    parser.add_argument("--secondary-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--feature", choices=["embedding", "projection"], default="embedding")
    parser.add_argument("--fusion", choices=["concat_l2", "average_l2"], default="concat_l2")
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/fusion_retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    primary_config = load_yaml_config(args.primary_config)
    secondary_config = load_yaml_config(args.secondary_config)

    primary_embeddings, primary_records = extract_checkpoint_embeddings(
        config=primary_config,
        checkpoint_path=args.primary_checkpoint,
        split=args.split,
        ids_path=args.ids_path,
        device=device,
        feature=args.feature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    secondary_embeddings, secondary_records = extract_checkpoint_embeddings(
        config=secondary_config,
        checkpoint_path=args.secondary_checkpoint,
        split=args.split,
        ids_path=args.ids_path,
        device=device,
        feature=args.feature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ensure_same_records(primary_records, secondary_records)

    fused_embeddings = fuse_embeddings(
        primary_embeddings=primary_embeddings,
        secondary_embeddings=secondary_embeddings,
        fusion=args.fusion,
    )
    labels = np.array([record["label"] for record in primary_records], dtype=np.int64)
    metrics = evaluate_retrieval(
        embeddings=fused_embeddings,
        labels=labels,
        recall_ks=(1, 5, 10),
        metric=args.metric,
    )
    latency = benchmark_search(
        embeddings=fused_embeddings,
        metric=args.metric,
        repeats=args.latency_repeats,
    )

    output_dir = make_output_dir(
        output_root=args.output_dir,
        primary_checkpoint=args.primary_checkpoint,
        secondary_checkpoint=args.secondary_checkpoint,
        fusion=args.fusion,
        metric=args.metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", fused_embeddings)
    write_records_csv(primary_records, output_dir / "records.csv")
    top_results = build_top_results(
        embeddings=fused_embeddings,
        records=primary_records,
        top_k=10,
        metric=args.metric,
    )
    (output_dir / "top_results.json").write_text(
        json.dumps(top_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    storage_bytes = int(fused_embeddings.shape[0] * fused_embeddings.shape[1] * FLOAT32_BYTES)
    summary = {
        "primary_checkpoint": str(args.primary_checkpoint),
        "primary_config": str(args.primary_config),
        "primary_crop_mode": primary_config["data"].get("crop_mode", "none"),
        "primary_bbox_margin": float(primary_config["data"].get("bbox_margin", 0.0)),
        "secondary_checkpoint": str(args.secondary_checkpoint),
        "secondary_config": str(args.secondary_config),
        "secondary_crop_mode": secondary_config["data"].get("crop_mode", "none"),
        "secondary_bbox_margin": float(secondary_config["data"].get("bbox_margin", 0.0)),
        "split": args.split,
        "ids_path": str(args.ids_path),
        "feature": args.feature,
        "fusion": args.fusion,
        "metric": args.metric,
        "num_samples": len(primary_records),
        "primary_embedding_dim": int(primary_embeddings.shape[1]),
        "secondary_embedding_dim": int(secondary_embeddings.shape[1]),
        "embedding_dim": int(fused_embeddings.shape[1]),
        "storage_bytes_float32": storage_bytes,
        "storage_mib_float32": storage_bytes / (1024 * 1024),
        "latency_repeats": args.latency_repeats,
        "search_latency_ms_total": latency["search_latency_ms_total"],
        "search_latency_ms_per_query": latency["search_latency_ms_per_query"],
        **metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


def extract_checkpoint_embeddings(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    ids_path: Path,
    device: torch.device,
    feature: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data_config = config["data"]
    model_config = config["model"]
    effective_batch_size = batch_size or config["training"]["batch_size"]
    effective_num_workers = (
        num_workers if num_workers is not None else int(data_config.get("num_workers", 0))
    )
    dataset = CUB200Dataset(
        root=data_config["root"],
        split=split,
        image_ids=read_image_ids(ids_path),
        crop_mode=data_config.get("crop_mode", "none"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        transform=build_classification_transform(
            image_size=data_config["image_size"],
            train=False,
            augmentation=data_config.get("augmentation", "hflip"),
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_config=model_config, feature=feature).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return extract_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        feature=feature,
    )


def fuse_embeddings(
    primary_embeddings: np.ndarray,
    secondary_embeddings: np.ndarray,
    fusion: str,
) -> np.ndarray:
    if primary_embeddings.shape[0] != secondary_embeddings.shape[0]:
        raise ValueError("primary and secondary embeddings must have the same number of samples")

    primary = l2_normalize(primary_embeddings.astype(np.float32, copy=False))
    secondary = l2_normalize(secondary_embeddings.astype(np.float32, copy=False))

    if fusion == "concat_l2":
        return l2_normalize(np.concatenate([primary, secondary], axis=1)).astype(np.float32)

    if fusion == "average_l2":
        if primary.shape[1] != secondary.shape[1]:
            raise ValueError("average_l2 requires embeddings with the same dimension")
        return l2_normalize((primary + secondary) / 2.0).astype(np.float32)

    raise ValueError(f"Unsupported fusion method: {fusion}")


def ensure_same_records(
    primary_records: list[dict[str, Any]],
    secondary_records: list[dict[str, Any]],
) -> None:
    if len(primary_records) != len(secondary_records):
        raise ValueError("primary and secondary records must have the same length")

    for index, (primary, secondary) in enumerate(zip(primary_records, secondary_records)):
        for key in ["image_id", "label", "class_name"]:
            if primary[key] != secondary[key]:
                raise ValueError(
                    f"Record mismatch at index {index}: "
                    f"primary[{key}]={primary[key]}, secondary[{key}]={secondary[key]}"
                )


def make_output_dir(
    output_root: Path,
    primary_checkpoint: Path,
    secondary_checkpoint: Path,
    fusion: str,
    metric: str,
) -> Path:
    primary_run = primary_checkpoint.parent.name
    secondary_run = secondary_checkpoint.parent.name
    return output_root / f"{primary_run}__{secondary_run}" / fusion / metric


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
