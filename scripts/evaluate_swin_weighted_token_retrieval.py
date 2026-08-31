from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Literal

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
from evaluate_swin_local_token_retrieval import (
    extract_swin_tokens,
    fuse_global_local,
)
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.models import build_swin_tiny_classifier
from visionsearch_fg.retrieval import evaluate_retrieval
from visionsearch_fg.utils import load_yaml_config

TargetClassMode = Literal["predicted", "true"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Swin-Tiny retrieval with class-evidence weighted local token pooling."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--target-class", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/swin_weighted_token_retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(temperature <= 0 for temperature in args.temperatures):
        raise ValueError("--temperatures values must be positive")

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

    embeddings_by_variant, records = extract_weighted_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        temperatures=sorted(set(args.temperatures)),
        target_class=args.target_class,
    )

    run_id = args.checkpoint.parent.name
    output_dir = args.output_dir / run_id / args.target_class
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
            "target_class": args.target_class,
            "crop_mode": data_config.get("crop_mode", "none"),
            "bbox_margin": float(data_config.get("bbox_margin", 0.0)),
            "num_samples": len(records),
            "embedding_dim": int(embeddings.shape[1]),
            "temperature": parse_variant_temperature(variant_name),
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
def extract_weighted_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperatures: list[float],
    target_class: TargetClassMode,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    chunks: dict[str, list[np.ndarray]] = {"global": []}
    for temperature in temperatures:
        suffix = format_temperature(temperature)
        chunks[f"evidence_weighted_tau{suffix}"] = []
        chunks[f"global_evidence_weighted_tau{suffix}_concat_l2"] = []

    records: list[dict[str, Any]] = []
    classifier_weight = model.classifier.weight.detach()
    for batch in tqdm(dataloader, desc="extract", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        global_embedding, tokens = extract_swin_tokens(model, images)
        logits = model.classifier(global_embedding)
        if target_class == "predicted":
            target_labels = logits.argmax(dim=1)
        elif target_class == "true":
            target_labels = labels
        else:
            raise ValueError(f"Unsupported target_class: {target_class}")

        chunks["global"].append(global_embedding.cpu().numpy())
        token_scores = compute_token_class_evidence(
            tokens=tokens,
            classifier_weight=classifier_weight,
            target_labels=target_labels,
        )
        for temperature in temperatures:
            suffix = format_temperature(temperature)
            local_embedding = pool_weighted_tokens(
                tokens=tokens,
                scores=token_scores,
                temperature=temperature,
            )
            fused_embedding = fuse_global_local(
                global_embedding=global_embedding,
                local_embedding=local_embedding,
            )
            chunks[f"evidence_weighted_tau{suffix}"].append(local_embedding.cpu().numpy())
            chunks[f"global_evidence_weighted_tau{suffix}_concat_l2"].append(
                fused_embedding.cpu().numpy()
            )

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


def compute_token_class_evidence(
    tokens: torch.Tensor,
    classifier_weight: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch_size, num_tokens, channels], got {tuple(tokens.shape)}")
    target_weight = classifier_weight[target_labels]
    return torch.einsum("bnc,bc->bn", tokens, target_weight)


def pool_weighted_tokens(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch_size, num_tokens, channels], got {tuple(tokens.shape)}")
    if scores.shape != tokens.shape[:2]:
        raise ValueError("scores must have shape [batch_size, num_tokens]")
    weights = F.softmax(scores / temperature, dim=1)
    return torch.einsum("bn,bnc->bc", weights, tokens)


def format_temperature(temperature: float) -> str:
    text = f"{temperature:g}"
    return text.replace(".", "_")


def parse_variant_temperature(variant_name: str) -> float | None:
    if "tau" not in variant_name:
        return None
    suffix = variant_name.split("tau", maxsplit=1)[1]
    suffix = suffix.split("_concat", maxsplit=1)[0]
    return float(suffix.replace("_", "."))


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone not in {"swin_tiny", "swin_t"}:
        raise ValueError("Weighted token retrieval currently requires a Swin-Tiny config")
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
        "target_class",
        "crop_mode",
        "bbox_margin",
        "num_samples",
        "embedding_dim",
        "temperature",
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
