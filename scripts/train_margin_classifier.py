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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from visionsearch_fg.data import (
    CUB200Dataset,
    build_classification_transform,
    build_two_view_transform,
    read_image_ids,
)
from visionsearch_fg.engine import validate
from visionsearch_fg.engine.metrics import accuracy
from visionsearch_fg.losses import SupConLoss
from visionsearch_fg.models import (
    AngularMarginClassifier,
    build_resnet18_classifier,
    build_swin_tiny_classifier,
    default_margin_for_type,
)
from visionsearch_fg.utils import load_yaml_config


@dataclass(frozen=True)
class MarginTrainStats:
    loss: float
    ce_loss: float
    supcon_loss: float
    accuracy: float
    num_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ArcFace/CosFace classifiers.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrained", choices=["true", "false"], default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config["project"]["seed"]))

    device = resolve_device(args.device)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]
    runtime_config = config.get("runtime", {})

    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training_config["batch_size"])
    pretrained = (
        args.pretrained.lower() == "true"
        if args.pretrained is not None
        else bool(model_config.get("pretrained", True))
    )
    max_train_batches = (
        args.max_train_batches
        if args.max_train_batches is not None
        else runtime_config.get("max_train_batches")
    )
    max_val_batches = (
        args.max_val_batches
        if args.max_val_batches is not None
        else runtime_config.get("max_val_batches")
    )
    experiment_name = args.experiment_name or config.get("experiment", {}).get(
        "name", "margin_classifier"
    )

    ce_weight = float(training_config.get("ce_weight", 1.0))
    supcon_weight = float(training_config.get("supcon_weight", 0.0))
    two_view = bool(training_config.get("two_view", supcon_weight > 0.0))
    supcon_feature = training_config.get("supcon_feature", "embedding")
    selection_metric = training_config.get("selection_metric", "val_accuracy")

    if ce_weight <= 0:
        raise ValueError("Angular margin training requires ce_weight > 0")
    if supcon_weight < 0:
        raise ValueError("supcon_weight must be non-negative")
    if supcon_weight > 0 and not two_view:
        raise ValueError("ArcFace + SupCon requires two_view=true")
    if supcon_feature not in {"embedding", "projection"}:
        raise ValueError("supcon_feature must be one of: embedding, projection")

    train_loader = build_dataloader(
        root=data_config["root"],
        split=data_config.get("train_split", "train"),
        image_ids_path=data_config.get("train_ids_path"),
        image_size=int(data_config["image_size"]),
        augmentation=data_config.get("augmentation", "hflip"),
        crop_mode=data_config.get("crop_mode", "none"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        batch_size=batch_size,
        num_workers=int(data_config["num_workers"]),
        train=True,
        two_view=two_view,
    )
    val_loader = build_dataloader(
        root=data_config["root"],
        split=data_config.get("val_split", "train"),
        image_ids_path=data_config.get("val_ids_path"),
        image_size=int(data_config["image_size"]),
        augmentation=data_config.get("augmentation", "hflip"),
        crop_mode=data_config.get("crop_mode", "none"),
        bbox_margin=float(data_config.get("bbox_margin", 0.0)),
        batch_size=batch_size,
        num_workers=int(data_config["num_workers"]),
        train=False,
        two_view=False,
    )

    model = build_model(model_config=model_config, pretrained=pretrained).to(device)
    parameter_counts = count_parameters(model)
    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=float(training_config.get("temperature", 0.07)))
    optimizer = torch.optim.AdamW(
        build_optimizer_parameter_groups(
            model=model,
            backbone_lr=training_config.get("backbone_learning_rate"),
            classifier_lr=training_config.get("classifier_learning_rate"),
            projection_lr=training_config.get("projection_learning_rate"),
            default_lr=float(training_config["learning_rate"]),
        ),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

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
        "backbone": model_config.get("backbone", "swin_tiny"),
        "epochs": epochs,
        "batch_size": batch_size,
        "pretrained": pretrained,
        "fine_tune_mode": model_config.get("fine_tune_mode", "full"),
        "margin_type": model.margin_type,
        "margin": model.margin,
        "scale": model.scale,
        "projection_dim": model.projection_dim,
        "total_parameters": parameter_counts["total_parameters"],
        "trainable_parameters": parameter_counts["trainable_parameters"],
        "ce_weight": ce_weight,
        "supcon_weight": supcon_weight,
        "supcon_feature": supcon_feature,
        "temperature": training_config.get("temperature", 0.07),
        "two_view": two_view,
        "selection_metric": selection_metric,
        "augmentation": data_config.get("augmentation", "hflip"),
        "crop_mode": data_config.get("crop_mode", "none"),
        "bbox_margin": float(data_config.get("bbox_margin", 0.0)),
        "train_ids_path": data_config.get("train_ids_path"),
        "val_ids_path": data_config.get("val_ids_path"),
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
    }
    (run_dirs["log_dir"] / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"log_dir: {run_dirs['log_dir']}")
    print(f"checkpoint_dir: {run_dirs['checkpoint_dir']}")

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = training_config.get("early_stopping_patience")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch_margin(
            model=model,
            dataloader=train_loader,
            ce_criterion=ce_criterion,
            supcon_criterion=supcon_criterion,
            optimizer=optimizer,
            device=device,
            ce_weight=ce_weight,
            supcon_weight=supcon_weight,
            supcon_feature=supcon_feature,
            max_batches=max_train_batches,
        )
        val_stats = validate(
            model=model,
            dataloader=val_loader,
            criterion=ce_criterion,
            device=device,
            max_batches=max_val_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_ce_loss": train_stats.ce_loss,
            "train_supcon_loss": train_stats.supcon_loss,
            "train_accuracy": train_stats.accuracy,
            "val_loss": val_stats.loss,
            "val_accuracy": val_stats.accuracy,
            "val_top5_accuracy": val_stats.top5_accuracy,
            "val_macro_f1": val_stats.macro_f1,
            "epoch_time_seconds": time.perf_counter() - epoch_start,
            "train_samples": train_stats.num_samples,
            "val_samples": val_stats.num_samples,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_stats.loss,
            "train_ce_loss": train_stats.ce_loss,
            "train_supcon_loss": train_stats.supcon_loss,
            "train_accuracy": train_stats.accuracy,
            "val_accuracy": val_stats.accuracy,
            "val_top5_accuracy": val_stats.top5_accuracy,
            "val_macro_f1": val_stats.macro_f1,
            "config": config,
            "metadata": metadata,
        }
        torch.save(checkpoint, run_dirs["checkpoint_dir"] / "last.pt")

        current_score = score_row(row=row, selection_metric=selection_metric)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, run_dirs["checkpoint_dir"] / "best.pt")
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and selection_metric == "val_accuracy"
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "early_stopping: "
                f"best_epoch={best_epoch}, patience={early_stopping_patience}"
            )
            break

    log_path = run_dirs["log_dir"] / "history.json"
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_log: {log_path}")


def build_dataloader(
    root: str,
    split: str,
    image_ids_path: str | None,
    image_size: int,
    augmentation: str,
    crop_mode: str,
    bbox_margin: float,
    batch_size: int,
    num_workers: int,
    train: bool,
    two_view: bool,
) -> DataLoader:
    transform = (
        build_two_view_transform(image_size=image_size, augmentation=augmentation)
        if two_view
        else build_classification_transform(
            image_size=image_size,
            train=train,
            augmentation=augmentation,
        )
    )
    dataset = CUB200Dataset(
        root=root,
        split=split,
        image_ids=read_image_ids(image_ids_path) if image_ids_path is not None else None,
        crop_mode=crop_mode,
        bbox_margin=bbox_margin,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(model_config: dict, pretrained: bool) -> AngularMarginClassifier:
    backbone = model_config.get("backbone", "swin_tiny")
    if backbone == "resnet18":
        base_classifier = build_resnet18_classifier(
            num_classes=int(model_config["num_classes"]),
            pretrained=pretrained,
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
            fine_tune_mode=model_config.get("fine_tune_mode"),
            trainable_backbone_layers=model_config.get("trainable_backbone_layers"),
        )
    elif backbone in {"swin_tiny", "swin_t"}:
        base_classifier = build_swin_tiny_classifier(
            num_classes=int(model_config["num_classes"]),
            pretrained=pretrained,
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
            fine_tune_mode=model_config.get("fine_tune_mode"),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    margin_type = model_config.get("margin_type", "arcface")
    return AngularMarginClassifier(
        base_classifier=base_classifier,
        margin_type=margin_type,
        scale=float(model_config.get("scale", 30.0)),
        margin=float(model_config.get("margin", default_margin_for_type(margin_type))),
        projection_dim=model_config.get("projection_dim"),
        projection_hidden_dim=model_config.get("projection_hidden_dim"),
    )


def train_one_epoch_margin(
    model: AngularMarginClassifier,
    dataloader: DataLoader,
    ce_criterion: nn.Module,
    supcon_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_weight: float,
    supcon_weight: float,
    supcon_feature: str,
    max_batches: int | None,
) -> MarginTrainStats:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_supcon_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress = tqdm(limit_batches(dataloader, max_batches), desc="train", leave=False)
    for batch in progress:
        labels = batch["label"].to(device)
        view_1, view_2 = split_views(batch["image"])
        if view_2 is None:
            model_images = view_1.to(device)
            ce_labels = labels
        else:
            model_images = torch.cat([view_1.to(device), view_2.to(device)], dim=0)
            ce_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad(set_to_none=True)
        output = model(model_images, labels=ce_labels)
        ce_loss = ce_criterion(output.logits, ce_labels)

        if supcon_weight > 0:
            if view_2 is None:
                raise ValueError("SupCon loss requires two views in the training batch")
            batch_size = labels.shape[0]
            supcon_output = output.embedding if supcon_feature == "embedding" else output.projection
            if supcon_output is None:
                raise ValueError("supcon_feature=projection requires model.projection_dim")
            features = torch.stack(
                [supcon_output[:batch_size], supcon_output[batch_size:]],
                dim=1,
            )
            supcon_loss = supcon_criterion(features, labels)
        else:
            supcon_loss = output.logits.sum() * 0.0

        loss = (ce_weight * ce_loss) + (supcon_weight * supcon_loss)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        logits_for_accuracy = output.logits[:batch_size] if view_2 is not None else output.logits
        batch_accuracy = accuracy(logits_for_accuracy.detach(), labels)

        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        total_supcon_loss += supcon_loss.item() * batch_size
        total_correct += batch_accuracy * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            ce=f"{total_ce_loss / total_samples:.4f}",
            supcon=f"{total_supcon_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return MarginTrainStats(
        loss=total_loss / total_samples,
        ce_loss=total_ce_loss / total_samples,
        supcon_loss=total_supcon_loss / total_samples,
        accuracy=total_correct / total_samples,
        num_samples=total_samples,
    )


def split_views(images) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(images, (list, tuple)):
        if len(images) != 2:
            raise ValueError("two-view batches must contain exactly two image tensors")
        return images[0], images[1]
    return images, None


def score_row(row: dict, selection_metric: str) -> float:
    if selection_metric not in row:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    return float(row[selection_metric])


def build_optimizer_parameter_groups(
    model: AngularMarginClassifier,
    backbone_lr: float | None,
    classifier_lr: float | None,
    projection_lr: float | None,
    default_lr: float,
) -> list[dict]:
    parameter_groups: list[dict] = []
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [p for p in model.margin_head.parameters() if p.requires_grad]
    projection_parameters = (
        []
        if model.projection_head is None
        else [p for p in model.projection_head.parameters() if p.requires_grad]
    )
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": backbone_lr or default_lr})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": classifier_lr or default_lr})
    if projection_parameters:
        parameter_groups.append({"params": projection_parameters, "lr": projection_lr or default_lr})
    if not parameter_groups:
        raise RuntimeError("No trainable parameters found for optimizer")
    return parameter_groups


def limit_batches(dataloader: DataLoader, max_batches: int | None):
    if max_batches is None:
        yield from dataloader
        return
    if max_batches < 1:
        raise ValueError("max_batches must be greater than or equal to 1")
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        yield batch


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def create_run_dirs(checkpoint_root: Path, log_root: Path, experiment_name: str) -> dict:
    safe_name = sanitize_name(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = make_unique_run_id(checkpoint_root, log_root, safe_name, timestamp)
    checkpoint_dir = checkpoint_root / safe_name / run_id
    log_dir = log_root / safe_name / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    return {"run_id": run_id, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir}


def make_unique_run_id(
    checkpoint_root: Path,
    log_root: Path,
    experiment_name: str,
    timestamp: str,
) -> str:
    base_run_id = f"{timestamp}_{experiment_name}"
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        if not (checkpoint_root / experiment_name / run_id).exists() and not (
            log_root / experiment_name / run_id
        ).exists():
            return run_id
    raise RuntimeError(f"Could not create a unique run id for experiment: {experiment_name}")


def sanitize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("experiment name must contain at least one valid character")
    return normalized


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
