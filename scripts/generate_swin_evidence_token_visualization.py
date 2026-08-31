from __future__ import annotations

import argparse
import csv
import html
import json
import math
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_swin_local_token_retrieval import extract_swin_tokens
from evaluate_swin_weighted_token_retrieval import compute_token_class_evidence
from generate_swin_attention_visualization import (
    colorize_heatmap,
    dataset_transform_image,
    load_class_names,
    make_image_name,
    save_original_and_overlay,
    select_records,
)
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.models import build_swin_tiny_classifier
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate class-evidence token heatmaps for Swin-Tiny."
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
        help="Class used to compute token-class evidence.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/explainability/swin_evidence_token_phase8"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_correct < 0 or args.num_wrong < 0:
        raise ValueError("--num-correct and --num-wrong must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
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
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
    output_dir = args.output_dir / run_id / f"{args.target_class}_tau{format_temperature(args.temperature)}"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    sample_by_id = {sample.image_id: sample for sample in dataset.samples}
    output_records = []
    for index, record in enumerate(tqdm(selected_records, desc="evidence", leave=False), start=1):
        sample = sample_by_id[int(record["image_id"])]
        target_label = int(record["pred_label"]) if args.target_class == "predicted" else int(record["true_label"])
        image_tensor = dataset_transform_image(
            image_path=sample.image_path,
            image_size=data_config["image_size"],
            augmentation=data_config.get("augmentation", "hflip"),
            bbox=sample.bbox,
            crop_mode=data_config.get("crop_mode", "none"),
            bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        ).unsqueeze(0)
        heatmap = generate_evidence_heatmap(
            model=model,
            image=image_tensor.to(device),
            target_label=target_label,
            temperature=args.temperature,
        )
        image_name = make_image_name(index=index, record=record)
        original_path = image_dir / f"{image_name}_original.jpg"
        overlay_path = image_dir / f"{image_name}_evidence.jpg"
        heatmap_path = image_dir / f"{image_name}_evidence_heatmap.npy"
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
                "temperature": args.temperature,
                "original_path": str(original_path),
                "evidence_path": str(overlay_path),
                "heatmap_path": str(heatmap_path),
                "manual_focus_local_detail": "",
                "manual_focus_bird_body": "",
                "manual_focus_background": "",
                "manual_notes": "",
            }
        )

    summary = summarize_records(
        records=records,
        selected_records=output_records,
        target_class=args.target_class,
        temperature=args.temperature,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evidence_records.json").write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_records_csv(output_records, output_dir / "evidence_records.csv")
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
        global_embedding, _tokens = extract_swin_tokens(model, images)
        logits = model.classifier(global_embedding)
        probabilities = torch.softmax(logits, dim=1)
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


@torch.no_grad()
def generate_evidence_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_label: int,
    temperature: float,
) -> np.ndarray:
    _global_embedding, tokens = extract_swin_tokens(model, image)
    target_labels = torch.tensor([target_label], dtype=torch.long, device=image.device)
    token_scores = compute_token_class_evidence(
        tokens=tokens,
        classifier_weight=model.classifier.weight.detach(),
        target_labels=target_labels,
    )
    return token_scores_to_heatmap(token_scores[0], temperature=temperature).cpu().numpy()


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


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone not in {"swin_tiny", "swin_t"}:
        raise ValueError("Evidence token visualization requires a Swin-Tiny config")
    return build_swin_tiny_classifier(
        num_classes=model_config["num_classes"],
        pretrained=False,
        freeze_backbone=model_config.get("freeze_backbone", False),
        fine_tune_mode=model_config.get("fine_tune_mode"),
    )


def summarize_records(
    records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    target_class: str,
    temperature: float,
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
        "visualization_type": "class_evidence_weighted_swin_token_map",
        "target_class": target_class,
        "temperature": temperature,
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
        "temperature",
        "original_path",
        "evidence_path",
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
              <p>Evidence target: {html.escape(record["target_class"])}</p>
              <p>Temperature: {record["temperature"]}</p>
              <div class="images">
                <img src="{html.escape(Path(record["original_path"]).relative_to(output_path.parent).as_posix())}" alt="original">
                <img src="{html.escape(Path(record["evidence_path"]).relative_to(output_path.parent).as_posix())}" alt="evidence">
              </div>
            </article>
            """
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Swin Evidence Token Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    p {{ margin: 4px 0; }}
    .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }}
    img {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>Swin Evidence Token Visualization</h1>
  <p>Visualization type: class-evidence weighted final-stage Swin token map.</p>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def format_temperature(temperature: float) -> str:
    return f"{temperature:g}".replace(".", "_")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
