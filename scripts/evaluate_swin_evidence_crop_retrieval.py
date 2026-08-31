from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluate_ce_retrieval import FLOAT32_BYTES, benchmark_search, build_top_results
from evaluate_swin_local_token_retrieval import extract_swin_tokens
from evaluate_swin_weighted_token_retrieval import (
    compute_token_class_evidence,
    format_temperature,
    pool_weighted_tokens,
)
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, crop_image_to_bbox, read_image_ids
from visionsearch_fg.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from visionsearch_fg.models import build_swin_tiny_classifier
from visionsearch_fg.retrieval import evaluate_retrieval
from visionsearch_fg.utils import load_yaml_config

TargetClassMode = Literal["predicted", "true"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Swin-Tiny retrieval with automatic class-evidence local crops."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--target-class", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.6)
    parser.add_argument("--min-crop-ratio", type=float, default=0.35)
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/swin_evidence_crop_retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_crop_args(args)

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

    embeddings_by_variant, records = extract_evidence_crop_embeddings(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        image_size=int(data_config["image_size"]),
        crop_mode=data_config.get("crop_mode", "none"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        target_class=args.target_class,
        temperature=args.temperature,
        threshold_ratio=args.threshold_ratio,
        min_crop_ratio=args.min_crop_ratio,
        padding_ratio=args.padding_ratio,
    )

    run_id = args.checkpoint.parent.name
    suffix = f"{args.target_class}_tau{format_temperature(args.temperature)}"
    output_dir = args.output_dir / run_id / suffix
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
            "temperature": args.temperature,
            "threshold_ratio": args.threshold_ratio,
            "min_crop_ratio": args.min_crop_ratio,
            "padding_ratio": args.padding_ratio,
            "num_samples": len(records),
            "embedding_dim": int(embeddings.shape[1]),
            "mean_auto_crop_area_ratio": float(
                np.mean([record["auto_crop_area_ratio"] for record in records])
            ),
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
def extract_evidence_crop_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset: CUB200Dataset,
    device: torch.device,
    image_size: int,
    crop_mode: str,
    bbox_margin: float,
    target_class: TargetClassMode,
    temperature: float,
    threshold_ratio: float,
    min_crop_ratio: float,
    padding_ratio: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    chunks: dict[str, list[np.ndarray]] = {
        "global": [],
        f"evidence_weighted_tau{format_temperature(temperature)}": [],
        f"evidence_auto_crop_tau{format_temperature(temperature)}": [],
        f"global_evidence_weighted_tau{format_temperature(temperature)}_concat_l2": [],
        f"global_evidence_auto_crop_tau{format_temperature(temperature)}_concat_l2": [],
    }
    records: list[dict[str, Any]] = []
    sample_by_id = {sample.image_id: sample for sample in dataset.samples}
    auto_crop_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

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

        token_scores = compute_token_class_evidence(
            tokens=tokens,
            classifier_weight=classifier_weight,
            target_labels=target_labels,
        )
        weighted_local = pool_weighted_tokens(
            tokens=tokens,
            scores=token_scores,
            temperature=temperature,
        )
        auto_crop_images = []
        crop_metadata = []
        for item_index in range(images.shape[0]):
            image_id = int(batch["image_id"][item_index])
            sample = sample_by_id[image_id]
            global_pil = load_global_pil(
                image_path=sample.image_path,
                bbox=sample.bbox,
                crop_mode=crop_mode,
                bbox_margin=bbox_margin,
            )
            resized_global = global_pil.resize((image_size, image_size))
            crop_box = evidence_scores_to_crop_box(
                token_scores=token_scores[item_index].detach().cpu(),
                image_size=image_size,
                temperature=temperature,
                threshold_ratio=threshold_ratio,
                min_crop_ratio=min_crop_ratio,
                padding_ratio=padding_ratio,
            )
            auto_crop = resized_global.crop(crop_box)
            auto_crop_images.append(auto_crop_transform(auto_crop))
            crop_metadata.append(
                {
                    "auto_crop_left": crop_box[0],
                    "auto_crop_top": crop_box[1],
                    "auto_crop_right": crop_box[2],
                    "auto_crop_bottom": crop_box[3],
                    "auto_crop_area_ratio": crop_area_ratio(crop_box, image_size),
                }
            )

        auto_crop_batch = torch.stack(auto_crop_images).to(device)
        auto_crop_embedding = model(auto_crop_batch).embedding

        chunks["global"].append(global_embedding.cpu().numpy())
        chunks[f"evidence_weighted_tau{format_temperature(temperature)}"].append(
            weighted_local.cpu().numpy()
        )
        chunks[f"evidence_auto_crop_tau{format_temperature(temperature)}"].append(
            auto_crop_embedding.cpu().numpy()
        )
        chunks[f"global_evidence_weighted_tau{format_temperature(temperature)}_concat_l2"].append(
            fuse_embeddings([global_embedding, weighted_local]).cpu().numpy()
        )
        chunks[f"global_evidence_auto_crop_tau{format_temperature(temperature)}_concat_l2"].append(
            fuse_embeddings([global_embedding, auto_crop_embedding]).cpu().numpy()
        )

        for item_index in range(images.shape[0]):
            records.append(
                {
                    "image_id": int(batch["image_id"][item_index]),
                    "label": int(batch["label"][item_index]),
                    "class_name": batch["class_name"][item_index],
                    "path": batch["path"][item_index],
                    "target_label": int(target_labels[item_index].item()),
                    "pred_label": int(logits[item_index].argmax().item()),
                    **crop_metadata[item_index],
                }
            )

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}, records


def evidence_scores_to_crop_box(
    token_scores: torch.Tensor,
    image_size: int,
    temperature: float,
    threshold_ratio: float,
    min_crop_ratio: float,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    heatmap = token_scores_to_heatmap(token_scores, temperature=temperature)
    selected_y, selected_x = select_heatmap_cells(heatmap, threshold_ratio=threshold_ratio)
    side = heatmap.shape[0]
    cell_size = image_size / side
    left = float(selected_x.min()) * cell_size
    top = float(selected_y.min()) * cell_size
    right = float(selected_x.max() + 1) * cell_size
    bottom = float(selected_y.max() + 1) * cell_size

    pad = image_size * padding_ratio
    left -= pad
    top -= pad
    right += pad
    bottom += pad
    left, top, right, bottom = expand_box_to_min_size(
        box=(left, top, right, bottom),
        image_size=image_size,
        min_side=image_size * min_crop_ratio,
    )
    return (
        int(round(max(0, left))),
        int(round(max(0, top))),
        int(round(min(image_size, right))),
        int(round(min(image_size, bottom))),
    )


def token_scores_to_heatmap(token_scores: torch.Tensor, temperature: float) -> torch.Tensor:
    if token_scores.ndim != 1:
        raise ValueError("token_scores must have shape [num_tokens]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    num_tokens = token_scores.shape[0]
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected square token grid, got {num_tokens} tokens")
    weights = F.softmax(token_scores / temperature, dim=0)
    heatmap = weights.reshape(side, side)
    heatmap = heatmap - heatmap.min()
    return heatmap / heatmap.max().clamp_min(1e-12)


def select_heatmap_cells(
    heatmap: torch.Tensor,
    threshold_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if heatmap.ndim != 2:
        raise ValueError("heatmap must have shape [height, width]")
    if not 0 <= threshold_ratio <= 1:
        raise ValueError("threshold_ratio must be in [0, 1]")
    threshold = float(heatmap.max().item()) * threshold_ratio
    selected = torch.nonzero(heatmap >= threshold, as_tuple=False)
    if selected.numel() == 0:
        selected = torch.nonzero(heatmap == heatmap.max(), as_tuple=False)
    return selected[:, 0], selected[:, 1]


def expand_box_to_min_size(
    box: tuple[float, float, float, float],
    image_size: int,
    min_side: float,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    target_width = max(width, min_side)
    target_height = max(height, min_side)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    left = center_x - target_width / 2
    right = center_x + target_width / 2
    top = center_y - target_height / 2
    bottom = center_y + target_height / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_size:
        left -= right - image_size
        right = image_size
    if bottom > image_size:
        top -= bottom - image_size
        bottom = image_size
    return max(0, left), max(0, top), min(image_size, right), min(image_size, bottom)


def load_global_pil(
    image_path: Path,
    bbox: tuple[float, float, float, float] | None,
    crop_mode: str,
    bbox_margin: float,
) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if crop_mode == "none":
            return image.copy()
        if crop_mode == "bbox":
            if bbox is None:
                raise ValueError("crop_mode=bbox requires a bounding box")
            return crop_image_to_bbox(image, bbox, margin=bbox_margin)
    raise ValueError(f"Unsupported crop mode: {crop_mode}")


def crop_area_ratio(crop_box: tuple[int, int, int, int], image_size: int) -> float:
    left, top, right, bottom = crop_box
    return float((right - left) * (bottom - top) / (image_size * image_size))


def fuse_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
    if not embeddings:
        raise ValueError("At least one embedding tensor is required")
    normalized = [F.normalize(embedding, p=2, dim=1) for embedding in embeddings]
    return F.normalize(torch.cat(normalized, dim=1), p=2, dim=1)


def validate_crop_args(args: argparse.Namespace) -> None:
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 <= args.threshold_ratio <= 1:
        raise ValueError("--threshold-ratio must be in [0, 1]")
    if not 0 < args.min_crop_ratio <= 1:
        raise ValueError("--min-crop-ratio must be in (0, 1]")
    if args.padding_ratio < 0:
        raise ValueError("--padding-ratio must be non-negative")


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone not in {"swin_tiny", "swin_t"}:
        raise ValueError("Evidence crop retrieval currently requires a Swin-Tiny config")
    return build_swin_tiny_classifier(
        num_classes=model_config["num_classes"],
        pretrained=False,
        freeze_backbone=model_config.get("freeze_backbone", False),
        fine_tune_mode=model_config.get("fine_tune_mode"),
    )


def write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "image_id",
        "label",
        "class_name",
        "path",
        "target_label",
        "pred_label",
        "auto_crop_left",
        "auto_crop_top",
        "auto_crop_right",
        "auto_crop_bottom",
        "auto_crop_area_ratio",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


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
        "temperature",
        "threshold_ratio",
        "min_crop_ratio",
        "padding_ratio",
        "num_samples",
        "embedding_dim",
        "mean_auto_crop_area_ratio",
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
