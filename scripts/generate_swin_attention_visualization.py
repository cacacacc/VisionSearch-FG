from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from visionsearch_fg.data import (
    CUB200Dataset,
    build_classification_transform,
    crop_image_to_bbox,
    read_image_ids,
)
from visionsearch_fg.models import build_swin_tiny_classifier
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Swin-Tiny attention-style visualization maps."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--ids-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-correct", type=int, default=12)
    parser.add_argument("--num-wrong", type=int, default=12)
    parser.add_argument("--max-scan-batches", type=int, default=None)
    parser.add_argument(
        "--target-class",
        choices=["predicted", "true"],
        default="predicted",
        help="Class used for gradient-weighted Swin feature visualization.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/explainability/swin_attention_phase6"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_correct < 0 or args.num_wrong < 0:
        raise ValueError("--num-correct and --num-wrong must be non-negative")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0, 1]")

    config = load_yaml_config(args.config)
    device = resolve_device(args.device)
    data_config = config["data"]
    model_config = config["model"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else data_config["num_workers"]

    dataset = CUB200Dataset(
        root=data_config["root"],
        split=args.split,
        image_ids=read_image_ids(args.ids_path) if args.ids_path is not None else None,
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

    model = build_model(model_config=model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = load_class_names(Path(data_config["root"]))
    records = collect_prediction_records(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        device=device,
        max_scan_batches=args.max_scan_batches,
    )
    selected_records = select_records(
        records=records,
        num_correct=args.num_correct,
        num_wrong=args.num_wrong,
    )

    run_id = args.checkpoint.parent.name
    output_dir = args.output_dir / run_id
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    sample_by_id = {sample.image_id: sample for sample in dataset.samples}
    visualizer = SwinFeatureVisualizer(model=model)
    output_records = []
    try:
        for index, record in enumerate(
            tqdm(selected_records, desc="attention", leave=False),
            start=1,
        ):
            sample = sample_by_id[int(record["image_id"])]
            target_label = (
                int(record["pred_label"])
                if args.target_class == "predicted"
                else int(record["true_label"])
            )
            image_tensor = dataset_transform_image(
                image_path=sample.image_path,
                image_size=data_config["image_size"],
                augmentation=data_config.get("augmentation", "hflip"),
                bbox=sample.bbox,
                crop_mode=data_config.get("crop_mode", "none"),
                bbox_margin=float(data_config.get("bbox_margin", 0.0)),
            ).unsqueeze(0)
            heatmap = visualizer.generate(
                image=image_tensor.to(device),
                target_label=target_label,
            )
            image_name = make_image_name(index=index, record=record)
            original_path = image_dir / f"{image_name}_original.jpg"
            overlay_path = image_dir / f"{image_name}_attention.jpg"
            heatmap_path = image_dir / f"{image_name}_attention_heatmap.npy"
            np.save(heatmap_path, heatmap.astype(np.float32))
            save_original_and_overlay(
                image_path=sample.image_path,
                heatmap=heatmap,
                image_size=data_config["image_size"],
                bbox=sample.bbox,
                crop_mode=data_config.get("crop_mode", "none"),
                bbox_margin=float(data_config.get("bbox_margin", 0.0)),
                original_path=original_path,
                overlay_path=overlay_path,
                alpha=args.alpha,
            )
            output_records.append(
                {
                    **record,
                    "target_class_mode": args.target_class,
                    "target_label": target_label,
                    "target_class": class_names[target_label],
                    "original_path": str(original_path),
                    "attention_path": str(overlay_path),
                    "heatmap_path": str(heatmap_path),
                    "manual_focus_local_detail": "",
                    "manual_focus_bird_body": "",
                    "manual_focus_background": "",
                    "manual_notes": "",
                }
            )
    finally:
        visualizer.close()

    summary = summarize_records(records=records, selected_records=output_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "attention_records.json").write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_records_csv(output_records, output_dir / "attention_records.csv")
    write_html_report(output_records, output_dir / "index.html")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


@torch.no_grad()
def collect_prediction_records(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device,
    max_scan_batches: int | None,
) -> list[dict[str, Any]]:
    records = []
    total = len(dataloader) if max_scan_batches is None else min(max_scan_batches, len(dataloader))

    for batch_index, batch in enumerate(tqdm(dataloader, desc="scan", total=total, leave=False)):
        if max_scan_batches is not None and batch_index >= max_scan_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        output = model(images)
        probabilities = torch.softmax(output.logits, dim=1)
        pred_probs, pred_labels = probabilities.max(dim=1)

        for index in range(images.shape[0]):
            true_label = int(labels[index].item())
            pred_label = int(pred_labels[index].item())
            records.append(
                {
                    "image_id": int(batch["image_id"][index]),
                    "path": batch["path"][index],
                    "true_label": true_label,
                    "true_class": batch["class_name"][index],
                    "pred_label": pred_label,
                    "pred_class": class_names[pred_label],
                    "pred_probability": float(pred_probs[index].item()),
                    "correct": pred_label == true_label,
                }
            )

    return records


def select_records(
    records: list[dict[str, Any]],
    num_correct: int,
    num_wrong: int,
) -> list[dict[str, Any]]:
    correct = [record for record in records if record["correct"]]
    wrong = [record for record in records if not record["correct"]]
    correct = sorted(correct, key=lambda record: float(record["pred_probability"]), reverse=True)
    wrong = sorted(wrong, key=lambda record: float(record["pred_probability"]), reverse=True)
    return correct[:num_correct] + wrong[:num_wrong]


class SwinFeatureVisualizer:
    """Gradient-weighted visualization over Swin's final spatial feature map.

    Torchvision's Swin implementation does not expose attention weights directly.
    This class visualizes class-relevant final-stage token features as an
    attention-style proxy for qualitative analysis.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = self.model.backbone.features.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = self.model.backbone.features.register_full_backward_hook(
            self._save_gradients
        )

    def generate(self, image: torch.Tensor, target_label: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        score = output.logits[:, target_label].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Swin hooks did not capture activations and gradients")

        activations = to_channel_first(self.activations.detach()[0])
        gradients = to_channel_first(self.gradients.detach()[0])
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=0))
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max().clamp_min(1e-12)
        return heatmap.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activations(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

    def _save_gradients(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0]


def to_channel_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D feature tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] > 16:
        return tensor.permute(2, 0, 1)
    return tensor


def dataset_transform_image(
    image_path: Path,
    image_size: int,
    augmentation: str,
    bbox: tuple[float, float, float, float] | None,
    crop_mode: str,
    bbox_margin: float,
) -> torch.Tensor:
    transform = build_classification_transform(
        image_size=image_size,
        train=False,
        augmentation=augmentation,
    )
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if crop_mode == "bbox":
            if bbox is None:
                raise ValueError(f"Missing bounding box for {image_path}")
            image = crop_image_to_bbox(image, bbox, margin=bbox_margin)
        return transform(image)


def save_original_and_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    image_size: int,
    bbox: tuple[float, float, float, float] | None,
    crop_mode: str,
    bbox_margin: float,
    original_path: Path,
    overlay_path: Path,
    alpha: float,
) -> None:
    with Image.open(image_path) as image:
        original = image.convert("RGB")
        if crop_mode == "bbox":
            if bbox is None:
                raise ValueError(f"Missing bounding box for {image_path}")
            original = crop_image_to_bbox(original, bbox, margin=bbox_margin)
        original = original.resize((image_size, image_size))

    original_path.parent.mkdir(parents=True, exist_ok=True)
    original.save(original_path, quality=95)

    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    heatmap_image = heatmap_image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    heatmap_rgb = colorize_heatmap(np.asarray(heatmap_image, dtype=np.float32) / 255.0)
    original_rgb = np.asarray(original, dtype=np.float32)
    overlay = (1.0 - alpha) * original_rgb + alpha * heatmap_rgb
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path, quality=95)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 * heatmap, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * heatmap - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - heatmap), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1) * 255.0


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone not in {"swin_tiny", "swin_t"}:
        raise ValueError("Swin attention visualization requires a Swin-Tiny config")
    return build_swin_tiny_classifier(
        num_classes=model_config["num_classes"],
        pretrained=False,
        freeze_backbone=model_config.get("freeze_backbone", False),
        fine_tune_mode=model_config.get("fine_tune_mode"),
    )


def summarize_records(
    records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    correct_count = sum(1 for record in records if record["correct"])
    return {
        "num_scanned": len(records),
        "num_correct": correct_count,
        "num_wrong": len(records) - correct_count,
        "accuracy": correct_count / len(records) if records else 0.0,
        "num_selected": len(selected_records),
        "num_selected_correct": sum(1 for record in selected_records if record["correct"]),
        "num_selected_wrong": sum(1 for record in selected_records if not record["correct"]),
        "visualization_type": "gradient_weighted_swin_feature_map",
    }


def write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "image_id",
        "path",
        "true_label",
        "true_class",
        "pred_label",
        "pred_class",
        "pred_probability",
        "correct",
        "target_class_mode",
        "target_label",
        "target_class",
        "original_path",
        "attention_path",
        "heatmap_path",
        "manual_focus_local_detail",
        "manual_focus_bird_body",
        "manual_focus_background",
        "manual_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_html_report(records: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for record in records:
        status = "正确" if record["correct"] else "错误"
        rows.append(
            f"""
            <article class="card">
              <h2>{status} | Image {record["image_id"]}</h2>
              <p>True: {html.escape(record["true_class"])}</p>
              <p>Pred: {html.escape(record["pred_class"])} ({record["pred_probability"]:.3f})</p>
              <p>Visualization target: {html.escape(record["target_class"])}</p>
              <div class="images">
                <img src="{html.escape(Path(record["original_path"]).relative_to(output_path.parent).as_posix())}" alt="original">
                <img src="{html.escape(Path(record["attention_path"]).relative_to(output_path.parent).as_posix())}" alt="attention">
              </div>
              <p class="note">人工标注：Local detail / Bird body / Background</p>
            </article>
            """
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Swin Attention Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    p {{ margin: 4px 0; }}
    .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }}
    img {{ width: 100%; height: auto; display: block; }}
    .note {{ color: #5f6368; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>Swin Attention Visualization</h1>
  <p>Visualization type: gradient-weighted final-stage Swin feature map.</p>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def make_image_name(index: int, record: dict[str, Any]) -> str:
    status = "correct" if record["correct"] else "wrong"
    return f"{index:03d}_{status}_image{record['image_id']}"


def load_class_names(root: Path) -> list[str]:
    class_names = []
    for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, class_name = line.split(maxsplit=1)
        class_names.append(class_name)
    return class_names


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
