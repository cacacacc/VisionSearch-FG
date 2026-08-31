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
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_ce_retrieval import (
    FLOAT32_BYTES,
    benchmark_search,
    build_top_results,
)
from evaluate_part_heatmap_alignment import (
    BBox,
    DEFAULT_PART_GROUPS,
    PartLocation,
    read_bboxes,
    read_part_locations,
    read_part_names,
)
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, crop_image_to_bbox, read_image_ids
from visionsearch_fg.models import build_swin_tiny_classifier
from visionsearch_fg.retrieval import evaluate_retrieval
from visionsearch_fg.utils import load_yaml_config

DEFAULT_VIEW_GROUPS = ("head", "wing", "body")
DEFAULT_VARIANTS = (
    "global",
    "part_head",
    "part_wing",
    "part_body",
    "global_head_concat_l2",
    "global_head_wing_concat_l2",
    "global_head_wing_body_concat_l2",
)
REQUIRED_FUSION_GROUPS = {"head", "wing", "body"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle part-guided crop retrieval with CUB part annotations."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--view-groups", nargs="+", default=list(DEFAULT_VIEW_GROUPS))
    parser.add_argument("--crop-scale", type=float, default=2.0)
    parser.add_argument("--min-crop-ratio", type=float, default=0.25)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embeddings/part_guided_crop_retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crop_scale <= 0:
        raise ValueError("--crop-scale must be positive")
    if not 0 < args.min_crop_ratio <= 1:
        raise ValueError("--min-crop-ratio must be in (0, 1]")
    unknown_groups = sorted(set(args.view_groups) - set(DEFAULT_PART_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown part groups: {unknown_groups}")
    missing_fusion_groups = sorted(REQUIRED_FUSION_GROUPS - set(args.view_groups))
    if missing_fusion_groups:
        raise ValueError(f"--view-groups must include {missing_fusion_groups} for default fusion variants")

    config = load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    device = resolve_device(args.device)
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else data_config["num_workers"]

    dataset = PartGuidedCropDataset(
        cub_root=Path(data_config["root"]),
        split=args.split,
        image_ids=read_image_ids(args.ids_path),
        image_size=int(data_config["image_size"]),
        augmentation=data_config.get("augmentation", "hflip"),
        global_crop_mode=data_config.get("crop_mode", "none"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        view_groups=tuple(args.view_groups),
        crop_scale=args.crop_scale,
        min_crop_ratio=args.min_crop_ratio,
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

    embeddings_by_variant, records = extract_part_guided_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        view_groups=tuple(args.view_groups),
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
        write_part_records_csv(records, variant_dir / "records.csv")

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
            "global_crop_mode": data_config.get("crop_mode", "none"),
            "bbox_margin": float(data_config.get("bbox_margin", 0.0)),
            "view_groups": list(args.view_groups),
            "crop_scale": args.crop_scale,
            "min_crop_ratio": args.min_crop_ratio,
            "num_samples": len(records),
            "embedding_dim": int(embeddings.shape[1]),
            "missing_part_counts": count_missing_parts(records, tuple(args.view_groups)),
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


class PartGuidedCropDataset(Dataset):
    def __init__(
        self,
        cub_root: Path,
        split: str,
        image_ids: set[int],
        image_size: int,
        augmentation: str,
        global_crop_mode: str,
        bbox_margin: float,
        view_groups: tuple[str, ...],
        crop_scale: float,
        min_crop_ratio: float,
    ) -> None:
        self.cub_root = cub_root
        self.image_size = image_size
        self.global_crop_mode = global_crop_mode
        self.bbox_margin = bbox_margin
        self.view_groups = view_groups
        self.crop_scale = crop_scale
        self.min_crop_ratio = min_crop_ratio
        self.transform = build_classification_transform(
            image_size=image_size,
            train=False,
            augmentation=augmentation,
        )
        self.samples = CUB200Dataset(
            root=cub_root,
            split=split,
            image_ids=image_ids,
            crop_mode="none",
        ).samples
        self.part_names = read_part_names(cub_root / "parts" / "parts.txt")
        self.part_locations = read_part_locations(cub_root / "parts" / "part_locs.txt")
        self.bboxes = read_bboxes(cub_root / "bounding_boxes.txt")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            original = image.convert("RGB")

        bbox = self.bboxes.get(sample.image_id)
        global_image = self._make_global_image(original, bbox)
        output: dict[str, Any] = {
            "global": self.transform(global_image),
            "image_id": sample.image_id,
            "label": sample.label,
            "class_name": sample.class_name,
            "path": str(sample.image_path),
        }

        for group_name in self.view_groups:
            crop, visible = crop_image_to_part_group(
                image=original,
                image_id=sample.image_id,
                group_name=group_name,
                part_names=self.part_names,
                part_locations=self.part_locations,
                crop_scale=self.crop_scale,
                min_crop_ratio=self.min_crop_ratio,
            )
            if crop is None:
                crop = global_image
            output[f"part_{group_name}"] = self.transform(crop)
            output[f"{group_name}_visible"] = visible

        return output

    def _make_global_image(self, image: Image.Image, bbox: BBox | None) -> Image.Image:
        if self.global_crop_mode == "none":
            return image
        if self.global_crop_mode == "bbox":
            if bbox is None:
                raise ValueError("global_crop_mode=bbox requires bounding_boxes.txt")
            return crop_image_to_bbox(image, bbox, margin=self.bbox_margin)
        raise ValueError(f"Unsupported global crop mode: {self.global_crop_mode}")


@torch.no_grad()
def extract_part_guided_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    view_groups: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    chunks: dict[str, list[np.ndarray]] = {"global": []}
    for group_name in view_groups:
        chunks[f"part_{group_name}"] = []
    chunks["global_head_concat_l2"] = []
    chunks["global_head_wing_concat_l2"] = []
    chunks["global_head_wing_body_concat_l2"] = []

    records: list[dict[str, Any]] = []
    for batch in tqdm(dataloader, desc="extract", leave=False):
        view_embeddings = {"global": embed_view(model, batch["global"].to(device))}
        for group_name in view_groups:
            view_embeddings[f"part_{group_name}"] = embed_view(
                model,
                batch[f"part_{group_name}"].to(device),
            )

        chunks["global"].append(view_embeddings["global"].cpu().numpy())
        for group_name in view_groups:
            chunks[f"part_{group_name}"].append(view_embeddings[f"part_{group_name}"].cpu().numpy())

        chunks["global_head_concat_l2"].append(
            fuse_view_embeddings([view_embeddings["global"], view_embeddings["part_head"]])
            .cpu()
            .numpy()
        )
        chunks["global_head_wing_concat_l2"].append(
            fuse_view_embeddings(
                [
                    view_embeddings["global"],
                    view_embeddings["part_head"],
                    view_embeddings["part_wing"],
                ]
            )
            .cpu()
            .numpy()
        )
        chunks["global_head_wing_body_concat_l2"].append(
            fuse_view_embeddings(
                [
                    view_embeddings["global"],
                    view_embeddings["part_head"],
                    view_embeddings["part_wing"],
                    view_embeddings["part_body"],
                ]
            )
            .cpu()
            .numpy()
        )

        batch_size = len(batch["label"])
        for index in range(batch_size):
            record = {
                "image_id": int(batch["image_id"][index]),
                "label": int(batch["label"][index]),
                "class_name": batch["class_name"][index],
                "path": batch["path"][index],
            }
            for group_name in view_groups:
                record[f"{group_name}_visible"] = bool(batch[f"{group_name}_visible"][index])
            records.append(record)

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}, records


def embed_view(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if not hasattr(output, "embedding"):
        raise AttributeError("Model output does not contain embedding")
    return output.embedding


def fuse_view_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
    if not embeddings:
        raise ValueError("At least one embedding tensor is required")
    batch_size = embeddings[0].shape[0]
    for embedding in embeddings:
        if embedding.ndim != 2:
            raise ValueError("Each embedding tensor must have shape [batch_size, dim]")
        if embedding.shape[0] != batch_size:
            raise ValueError("All embedding tensors must share the same batch size")
    normalized = [F.normalize(embedding, p=2, dim=1) for embedding in embeddings]
    return F.normalize(torch.cat(normalized, dim=1), p=2, dim=1)


def crop_image_to_part_group(
    image: Image.Image,
    image_id: int,
    group_name: str,
    part_names: dict[int, str],
    part_locations: dict[int, dict[int, PartLocation]],
    crop_scale: float,
    min_crop_ratio: float,
) -> tuple[Image.Image | None, bool]:
    points = visible_group_points(
        image_id=image_id,
        group_name=group_name,
        part_names=part_names,
        part_locations=part_locations,
    )
    if not points:
        return None, False
    crop_box = compute_part_crop_box(
        points=points,
        image_size=image.size,
        crop_scale=crop_scale,
        min_crop_ratio=min_crop_ratio,
    )
    return image.crop(crop_box), True


def visible_group_points(
    image_id: int,
    group_name: str,
    part_names: dict[int, str],
    part_locations: dict[int, dict[int, PartLocation]],
) -> list[tuple[float, float]]:
    if group_name not in DEFAULT_PART_GROUPS:
        raise ValueError(f"Unknown part group: {group_name}")
    allowed_names = set(DEFAULT_PART_GROUPS[group_name])
    points = []
    for part_id, (x_coord, y_coord, visible) in part_locations.get(image_id, {}).items():
        if visible and part_names[part_id] in allowed_names:
            points.append((float(x_coord), float(y_coord)))
    return points


def compute_part_crop_box(
    points: list[tuple[float, float]],
    image_size: tuple[int, int],
    crop_scale: float,
    min_crop_ratio: float,
) -> tuple[int, int, int, int]:
    if not points:
        raise ValueError("points must not be empty")
    if crop_scale <= 0:
        raise ValueError("crop_scale must be positive")
    if not 0 < min_crop_ratio <= 1:
        raise ValueError("min_crop_ratio must be in (0, 1]")

    image_width, image_height = image_size
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    span = max(x_max - x_min, y_max - y_min)
    min_side = min(image_width, image_height) * min_crop_ratio
    side = max(span * crop_scale, min_side)
    side = min(side, image_width, image_height)

    left = center_x - side / 2
    top = center_y - side / 2
    right = center_x + side / 2
    bottom = center_y + side / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid part crop box for points={points}, image_size={image_size}")
    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def count_missing_parts(records: list[dict[str, Any]], view_groups: tuple[str, ...]) -> dict[str, int]:
    return {
        group_name: sum(not bool(record.get(f"{group_name}_visible")) for record in records)
        for group_name in view_groups
    }


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone not in {"swin_tiny", "swin_t"}:
        raise ValueError("Part-guided crop retrieval currently requires a Swin-Tiny config")
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
        "global_crop_mode",
        "bbox_margin",
        "view_groups",
        "crop_scale",
        "min_crop_ratio",
        "num_samples",
        "embedding_dim",
        "missing_part_counts",
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


def write_part_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(records[0].keys()) if records else []
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
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
