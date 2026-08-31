from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_ce_retrieval import (
    FLOAT32_BYTES,
    benchmark_search,
    build_top_results,
    write_records_csv,
)
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.models import build_swin_tiny_classifier
from visionsearch_fg.retrieval import evaluate_retrieval
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Swin-Tiny retrieval with global and top-response local token features."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--top-k-tokens", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/swin_local_token_retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(k < 1 for k in args.top_k_tokens):
        raise ValueError("--top-k-tokens values must be greater than or equal to 1")

    config = load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    device = resolve_device(args.device)
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else data_config["num_workers"]

    dataset = CUB200Dataset(
        root=data_config["root"],
        split=args.split,
        image_ids=read_image_ids(args.ids_path),
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    embeddings_by_variant, records = extract_global_and_local_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        top_k_tokens=sorted(set(args.top_k_tokens)),
    )

    run_id = args.checkpoint.parent.name
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.array([record["label"] for record in records], dtype=np.int64)
    rows = []
    for variant_name, embeddings in embeddings_by_variant.items():
        variant_dir = output_dir / variant_name / args.metric
        variant_dir.mkdir(parents=True, exist_ok=True)
        np.save(variant_dir / "embeddings.npy", embeddings)
        write_records_csv(records, variant_dir / "records.csv")

        metrics = evaluate_retrieval(
            embeddings=embeddings,
            labels=labels,
            recall_ks=(1, 5, 10),
            metric=args.metric,
        )
        latency = benchmark_search(
            embeddings=embeddings,
            metric=args.metric,
            repeats=args.latency_repeats,
        )
        top_results = build_top_results(
            embeddings=embeddings,
            records=records,
            top_k=10,
            metric=args.metric,
        )
        (variant_dir / "top_results.json").write_text(
            json.dumps(top_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        storage_bytes = int(embeddings.shape[0] * embeddings.shape[1] * FLOAT32_BYTES)
        summary = {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "split": args.split,
            "ids_path": str(args.ids_path),
            "backbone": model_config.get("backbone", "swin_tiny"),
            "variant": variant_name,
            "metric": args.metric,
            "crop_mode": data_config.get("crop_mode", "none"),
            "bbox_margin": float(data_config.get("bbox_margin", 0.0)),
            "num_samples": len(records),
            "embedding_dim": int(embeddings.shape[1]),
            "top_k_tokens": parse_variant_top_k(variant_name),
            "storage_bytes_float32": storage_bytes,
            "storage_mib_float32": storage_bytes / (1024 * 1024),
            "latency_repeats": args.latency_repeats,
            "search_latency_ms_total": latency["search_latency_ms_total"],
            "search_latency_ms_per_query": latency["search_latency_ms_per_query"],
            **metrics,
        }
        (variant_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(summary)

    write_summary_csv(rows, output_dir / "summary.csv")
    (output_dir / "summary.json").write_text(
        json.dumps({"results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output_dir), "results": rows}, ensure_ascii=False, indent=2))


@torch.no_grad()
def extract_global_and_local_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_k_tokens: list[int],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    chunks: dict[str, list[np.ndarray]] = {"global": []}
    for k_value in top_k_tokens:
        chunks[f"local_top{k_value}"] = []
        chunks[f"global_local_top{k_value}_concat_l2"] = []

    records: list[dict[str, Any]] = []
    for batch in tqdm(dataloader, desc="extract", leave=False):
        images = batch["image"].to(device)
        global_embedding, tokens = extract_swin_tokens(model, images)

        chunks["global"].append(global_embedding.cpu().numpy())
        num_tokens = tokens.shape[1]
        for k_value in top_k_tokens:
            effective_k = min(k_value, num_tokens)
            local_embedding = pool_topk_tokens(tokens=tokens, k=effective_k)
            fused_embedding = fuse_global_local(
                global_embedding=global_embedding,
                local_embedding=local_embedding,
            )
            chunks[f"local_top{k_value}"].append(local_embedding.cpu().numpy())
            chunks[f"global_local_top{k_value}_concat_l2"].append(fused_embedding.cpu().numpy())

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

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}, records


def extract_swin_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_features = model.backbone.features(images)
    tokens = flatten_swin_spatial_features(spatial_features)
    tokens = model.backbone.norm(tokens)
    global_embedding = tokens.mean(dim=1)
    return global_embedding, tokens


def flatten_swin_spatial_features(spatial_features: torch.Tensor) -> torch.Tensor:
    if spatial_features.ndim != 4:
        raise ValueError(f"Expected Swin spatial features with shape [B,H,W,C], got {tuple(spatial_features.shape)}")
    if spatial_features.shape[-1] >= spatial_features.shape[1]:
        batch_size, height, width, channels = spatial_features.shape
        return spatial_features.reshape(batch_size, height * width, channels)
    batch_size, channels, height, width = spatial_features.shape
    return spatial_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


def pool_topk_tokens(tokens: torch.Tensor, k: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch_size, num_tokens, channels], got {tuple(tokens.shape)}")
    if k < 1 or k > tokens.shape[1]:
        raise ValueError(f"k must be in [1, {tokens.shape[1]}], got {k}")
    token_scores = torch.linalg.vector_norm(tokens, ord=2, dim=-1)
    top_indices = torch.topk(token_scores, k=k, dim=1).indices
    gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(dim=1, index=gather_indices).mean(dim=1)


def fuse_global_local(
    global_embedding: torch.Tensor,
    local_embedding: torch.Tensor,
) -> torch.Tensor:
    if global_embedding.shape != local_embedding.shape:
        raise ValueError("global_embedding and local_embedding must have the same shape")
    global_normalized = F.normalize(global_embedding, p=2, dim=1)
    local_normalized = F.normalize(local_embedding, p=2, dim=1)
    return F.normalize(torch.cat([global_normalized, local_normalized], dim=1), p=2, dim=1)


def parse_variant_top_k(variant_name: str) -> int | None:
    for token in variant_name.split("_"):
        if token.startswith("top"):
            return int(token.removeprefix("top"))
    return None


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone not in {"swin_tiny", "swin_t"}:
        raise ValueError("Local token retrieval currently requires a Swin-Tiny config")
    return build_swin_tiny_classifier(
        num_classes=model_config["num_classes"],
        pretrained=False,
        freeze_backbone=model_config.get("freeze_backbone", False),
        fine_tune_mode=model_config.get("fine_tune_mode"),
    )


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "variant",
        "checkpoint",
        "config",
        "split",
        "ids_path",
        "backbone",
        "metric",
        "crop_mode",
        "bbox_margin",
        "num_samples",
        "embedding_dim",
        "top_k_tokens",
        "storage_bytes_float32",
        "storage_mib_float32",
        "latency_repeats",
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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
