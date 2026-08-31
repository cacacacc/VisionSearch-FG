from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.engine.metrics import accuracy, macro_f1_score, top_k_accuracy
from visionsearch_fg.models import build_swin_head_aware_classifier
from visionsearch_fg.utils import load_yaml_config

HEAD_PARTS = {"beak", "crown", "forehead", "left eye", "right eye", "nape", "throat"}


@dataclass(frozen=True)
class EpochStats:
    loss: float
    ce_loss: float
    selector_loss: float
    accuracy: float
    top5_accuracy: float
    macro_f1: float
    num_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin-Tiny with a head-aware soft local selector.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config["project"]["seed"])

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]
    runtime_config = config.get("runtime", {})
    device = resolve_device(args.device)
    if data_config.get("augmentation", "basic") != "basic":
        raise ValueError("Head selector training requires deterministic augmentation=basic")

    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else training_config["batch_size"]
    max_train_batches = args.max_train_batches if args.max_train_batches is not None else runtime_config.get("max_train_batches")
    max_val_batches = args.max_val_batches if args.max_val_batches is not None else runtime_config.get("max_val_batches")
    experiment_name = args.experiment_name or config.get("experiment", {}).get("name", "swin_head_selector")

    train_loader = build_head_selector_dataloader(
        root=Path(data_config["root"]),
        split=data_config.get("train_split", "train"),
        image_ids_path=data_config.get("train_ids_path"),
        image_size=int(data_config["image_size"]),
        augmentation=data_config.get("augmentation", "hflip"),
        crop_mode=data_config.get("crop_mode", "bbox"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        heatmap_size=int(training_config.get("selector_heatmap_size", 7)),
        heatmap_sigma=float(training_config.get("selector_heatmap_sigma", 1.0)),
        batch_size=batch_size,
        num_workers=int(data_config["num_workers"]),
        train=True,
    )
    val_loader = build_head_selector_dataloader(
        root=Path(data_config["root"]),
        split=data_config.get("val_split", "train"),
        image_ids_path=data_config.get("val_ids_path"),
        image_size=int(data_config["image_size"]),
        augmentation=data_config.get("augmentation", "hflip"),
        crop_mode=data_config.get("crop_mode", "bbox"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        heatmap_size=int(training_config.get("selector_heatmap_size", 7)),
        heatmap_sigma=float(training_config.get("selector_heatmap_sigma", 1.0)),
        batch_size=batch_size,
        num_workers=int(data_config["num_workers"]),
        train=False,
    )

    model = build_swin_head_aware_classifier(
        num_classes=model_config["num_classes"],
        pretrained=model_config.get("pretrained", True),
        freeze_backbone=model_config.get("freeze_backbone", False),
        fine_tune_mode=model_config.get("fine_tune_mode"),
        selector_temperature=float(model_config.get("selector_temperature", 1.0)),
    ).to(device)

    init_checkpoint = args.init_checkpoint or model_config.get("init_checkpoint")
    load_report = None
    if init_checkpoint:
        load_report = load_compatible_checkpoint(model, Path(init_checkpoint), device)

    optimizer = torch.optim.AdamW(
        build_optimizer_parameter_groups(
            model=model,
            backbone_lr=training_config.get("backbone_learning_rate"),
            selector_lr=training_config.get("selector_learning_rate"),
            classifier_lr=training_config.get("classifier_learning_rate"),
            default_lr=training_config["learning_rate"],
        ),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    ce_criterion = nn.CrossEntropyLoss()
    selector_criterion = nn.BCEWithLogitsLoss()
    selector_loss_weight = float(training_config.get("selector_loss_weight", 0.2))

    run_dirs = create_run_dirs(
        checkpoint_root=Path(output_config["checkpoint_dir"]),
        log_root=Path(output_config["log_dir"]),
        experiment_name=experiment_name,
    )
    shutil.copy2(args.config, run_dirs["log_dir"] / "config.yaml")

    metadata = {
        "experiment_name": experiment_name,
        "run_id": run_dirs["run_id"],
        "config_path": str(args.config),
        "device": str(device),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "load_report": load_report,
        "epochs": epochs,
        "batch_size": batch_size,
        "selector_loss_weight": selector_loss_weight,
        "selector_temperature": float(model_config.get("selector_temperature", 1.0)),
        "fine_tune_mode": model_config.get("fine_tune_mode"),
        "crop_mode": data_config.get("crop_mode", "bbox"),
        "bbox_margin": float(data_config.get("bbox_margin", 0.0)),
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
    }
    (run_dirs["log_dir"] / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"checkpoint_dir: {run_dirs['checkpoint_dir']}")
    print(f"log_dir: {run_dirs['log_dir']}")

    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = training_config.get("early_stopping_patience")
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = run_epoch(
            model=model,
            dataloader=train_loader,
            ce_criterion=ce_criterion,
            selector_criterion=selector_criterion,
            selector_loss_weight=selector_loss_weight,
            device=device,
            optimizer=optimizer,
            max_batches=max_train_batches,
            train=True,
        )
        val_stats = run_epoch(
            model=model,
            dataloader=val_loader,
            ce_criterion=ce_criterion,
            selector_criterion=selector_criterion,
            selector_loss_weight=selector_loss_weight,
            device=device,
            optimizer=None,
            max_batches=max_val_batches,
            train=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_ce_loss": train_stats.ce_loss,
            "train_selector_loss": train_stats.selector_loss,
            "train_accuracy": train_stats.accuracy,
            "val_loss": val_stats.loss,
            "val_ce_loss": val_stats.ce_loss,
            "val_selector_loss": val_stats.selector_loss,
            "val_accuracy": val_stats.accuracy,
            "val_top5_accuracy": val_stats.top5_accuracy,
            "val_macro_f1": val_stats.macro_f1,
            "epoch_time_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_stats.accuracy,
            "val_top5_accuracy": val_stats.top5_accuracy,
            "val_macro_f1": val_stats.macro_f1,
            "config": config,
            "metadata": metadata,
        }
        torch.save(checkpoint, run_dirs["checkpoint_dir"] / "last.pt")
        if val_stats.accuracy > best_accuracy:
            best_accuracy = val_stats.accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, run_dirs["checkpoint_dir"] / "best.pt")
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"early_stopping: best_epoch={best_epoch}, patience={patience}")
            break

    log_path = run_dirs["log_dir"] / "history.json"
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_log: {log_path}")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    ce_criterion: nn.Module,
    selector_criterion: nn.Module,
    selector_loss_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    train: bool,
) -> EpochStats:
    model.train(mode=train)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_selector_loss = 0.0
    total_correct = 0.0
    total_top5_correct = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    desc = "train" if train else "valid"
    progress = tqdm(limit_batches(dataloader, max_batches), desc=desc, leave=False)

    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        head_targets = batch["head_heatmap"].to(device).flatten(start_dim=1)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            output = model(images)
            ce_loss = ce_criterion(output.logits, labels)
            selector_loss = selector_criterion(output.selector_logits, head_targets)
            loss = ce_loss + selector_loss_weight * selector_loss
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        batch_accuracy = accuracy(output.logits.detach(), labels)
        batch_top5 = top_k_accuracy(output.logits.detach(), labels, k=5)
        total_loss += float(loss.item()) * batch_size
        total_ce_loss += float(ce_loss.item()) * batch_size
        total_selector_loss += float(selector_loss.item()) * batch_size
        total_correct += batch_accuracy * batch_size
        total_top5_correct += batch_top5 * batch_size
        total_samples += batch_size
        all_predictions.append(output.logits.detach().argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return EpochStats(
        loss=total_loss / total_samples,
        ce_loss=total_ce_loss / total_samples,
        selector_loss=total_selector_loss / total_samples,
        accuracy=total_correct / total_samples,
        top5_accuracy=total_top5_correct / total_samples,
        macro_f1=macro_f1_score(torch.cat(all_predictions), torch.cat(all_labels)),
        num_samples=total_samples,
    )


class HeadHeatmapCUBDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        image_ids: set[int] | None,
        image_size: int,
        augmentation: str,
        crop_mode: str,
        bbox_margin: float,
        heatmap_size: int,
        heatmap_sigma: float,
        train: bool,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.bbox_margin = bbox_margin
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.transform = build_classification_transform(
            image_size=image_size,
            train=train,
            augmentation=augmentation,
        )
        dataset = CUB200Dataset(root=root, split=split, image_ids=image_ids, crop_mode="bbox" if crop_mode == "bbox" else "none", bbox_margin=bbox_margin)
        self.samples = dataset.samples
        self.part_names = read_part_names(root / "parts" / "parts.txt")
        self.part_locations = read_part_locations(root / "parts" / "part_locs.txt")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            original = image.convert("RGB")
        head_points = transform_head_points_to_input(
            image_id=sample.image_id,
            part_names=self.part_names,
            part_locations=self.part_locations,
            original_size=original.size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            bbox=sample.bbox,
            bbox_margin=self.bbox_margin,
        )
        if self.crop_mode == "bbox":
            if sample.bbox is None:
                raise ValueError(f"Missing bounding box for image id {sample.image_id}")
            from visionsearch_fg.data import crop_image_to_bbox

            original = crop_image_to_bbox(original, sample.bbox, margin=self.bbox_margin)
        return {
            "image": self.transform(original),
            "head_heatmap": make_gaussian_heatmap(
                points=head_points,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.heatmap_sigma,
            ),
            "label": sample.label,
            "image_id": sample.image_id,
            "path": str(sample.image_path),
            "class_name": sample.class_name,
        }


def transform_head_points_to_input(
    image_id: int,
    part_names: dict[int, str],
    part_locations: dict[int, dict[int, tuple[float, float, bool]]],
    original_size: tuple[int, int],
    image_size: int,
    crop_mode: str,
    bbox: tuple[float, float, float, float] | None,
    bbox_margin: float,
) -> list[tuple[float, float]]:
    image_width, image_height = original_size
    crop_box = compute_bbox_crop_box(original_size, bbox, bbox_margin) if crop_mode == "bbox" else None
    points = []
    for part_id, (x_coord, y_coord, visible) in part_locations.get(image_id, {}).items():
        if not visible or part_names[part_id] not in HEAD_PARTS:
            continue
        if crop_box is None:
            points.append((x_coord * image_size / image_width, y_coord * image_size / image_height))
            continue
        x_min, y_min, x_max, y_max = crop_box
        if not (x_min <= x_coord <= x_max and y_min <= y_coord <= y_max):
            continue
        points.append(
            (
                (x_coord - x_min) * image_size / (x_max - x_min),
                (y_coord - y_min) * image_size / (y_max - y_min),
            )
        )
    return [(float(x_coord), float(y_coord)) for x_coord, y_coord in points]


def make_gaussian_heatmap(
    points: list[tuple[float, float]],
    image_size: int,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    if heatmap_size < 1:
        raise ValueError("heatmap_size must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    ys = (torch.arange(heatmap_size, dtype=torch.float32) + 0.5) * image_size / heatmap_size
    xs = (torch.arange(heatmap_size, dtype=torch.float32) + 0.5) * image_size / heatmap_size
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    heatmap = torch.zeros((heatmap_size, heatmap_size), dtype=torch.float32)
    if not points:
        return heatmap
    sigma_pixels = sigma * image_size / heatmap_size
    for point_x, point_y in points:
        heatmap = torch.maximum(
            heatmap,
            torch.exp(-((grid_x - point_x) ** 2 + (grid_y - point_y) ** 2) / (2 * sigma_pixels**2)),
        )
    return heatmap / heatmap.max().clamp_min(1e-12)


def compute_bbox_crop_box(
    original_size: tuple[int, int],
    bbox: tuple[float, float, float, float] | None,
    margin: float,
) -> tuple[float, float, float, float]:
    if bbox is None:
        raise ValueError("bbox crop requires a bounding box")
    x_coord, y_coord, width, height = bbox
    image_width, image_height = original_size
    x_min = max(0, int(round(x_coord - width * margin)))
    y_min = max(0, int(round(y_coord - height * margin)))
    x_max = min(image_width, int(round(x_coord + width + width * margin)))
    y_max = min(image_height, int(round(y_coord + height + height * margin)))
    return float(x_min), float(y_min), float(x_max), float(y_max)


def build_head_selector_dataloader(
    root: Path,
    split: str,
    image_ids_path: str | None,
    image_size: int,
    augmentation: str,
    crop_mode: str,
    bbox_margin: float,
    heatmap_size: int,
    heatmap_sigma: float,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    image_ids = read_image_ids(image_ids_path) if image_ids_path is not None else None
    dataset = HeadHeatmapCUBDataset(
        root=root,
        split=split,
        image_ids=image_ids,
        image_size=image_size,
        augmentation=augmentation,
        crop_mode=crop_mode,
        bbox_margin=bbox_margin,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def read_part_names(path: Path) -> dict[int, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        part_id, name = line.split(maxsplit=1)
        rows[int(part_id)] = name
    return rows


def read_part_locations(path: Path) -> dict[int, dict[int, tuple[float, float, bool]]]:
    rows: dict[int, dict[int, tuple[float, float, bool]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_id, part_id, x_coord, y_coord, visible = line.split()
        rows.setdefault(int(image_id), {})[int(part_id)] = (
            float(x_coord),
            float(y_coord),
            bool(int(visible)),
        )
    return rows


def load_compatible_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape and not key.startswith("classifier.")
    }
    skipped = sorted(set(source_state) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(compatible),
        "skipped_keys": skipped,
    }


def build_optimizer_parameter_groups(
    model: nn.Module,
    backbone_lr: float | None,
    selector_lr: float | None,
    classifier_lr: float | None,
    default_lr: float,
) -> list[dict[str, Any]]:
    groups = []
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    selector_parameters = [p for p in model.selector.parameters() if p.requires_grad]
    classifier_parameters = [p for p in model.classifier.parameters() if p.requires_grad]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_lr or default_lr})
    if selector_parameters:
        groups.append({"params": selector_parameters, "lr": selector_lr or default_lr})
    if classifier_parameters:
        groups.append({"params": classifier_parameters, "lr": classifier_lr or default_lr})
    if not groups:
        raise RuntimeError("No trainable parameters found")
    return groups


def limit_batches(dataloader: DataLoader, max_batches: int | None):
    if max_batches is None:
        yield from dataloader
        return
    if max_batches < 1:
        raise ValueError("max_batches must be greater than or equal to 1")
    for index, batch in enumerate(dataloader):
        if index >= max_batches:
            break
        yield batch


def create_run_dirs(checkpoint_root: Path, log_root: Path, experiment_name: str) -> dict[str, Path | str]:
    safe_name = sanitize_name(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{safe_name}"
    checkpoint_dir = checkpoint_root / safe_name / run_id
    log_dir = log_root / safe_name / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    return {"run_id": run_id, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir}


def sanitize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()).strip("._-")
    if not normalized:
        raise ValueError("experiment name must contain at least one valid character")
    return normalized


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
