from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.engine import train_one_epoch, validate
from visionsearch_fg.models import build_resnet18_classifier
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear probe on a frozen encoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config["project"]["seed"])

    device = resolve_device(args.device)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]
    runtime_config = config.get("runtime", {})

    epochs = args.epochs if args.epochs is not None else int(training_config.get("probe_epochs", 20))
    batch_size = args.batch_size if args.batch_size is not None else training_config["batch_size"]
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(training_config.get("probe_learning_rate", 0.001))
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
    experiment_name = args.experiment_name or f"{config['experiment']['name']}_linear_probe"

    train_loader = build_dataloader(
        root=data_config["root"],
        split=data_config.get("train_split", "train"),
        image_ids_path=data_config.get("train_ids_path"),
        image_size=data_config["image_size"],
        augmentation=data_config.get("probe_augmentation", "hflip"),
        batch_size=batch_size,
        num_workers=data_config["num_workers"],
        train=True,
    )
    val_loader = build_dataloader(
        root=data_config["root"],
        split=data_config.get("val_split", "train"),
        image_ids_path=data_config.get("val_ids_path"),
        image_size=data_config["image_size"],
        augmentation=data_config.get("probe_augmentation", "hflip"),
        batch_size=batch_size,
        num_workers=data_config["num_workers"],
        train=False,
    )

    model = build_resnet18_classifier(
        num_classes=model_config["num_classes"],
        pretrained=False,
        freeze_backbone=False,
        fine_tune_mode="full",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.classifier.reset_parameters()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=float(training_config.get("probe_weight_decay", 0.0001)),
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
        "source_checkpoint": str(args.checkpoint),
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "frozen_encoder": True,
        "reset_classifier": True,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
    }
    (run_dirs["log_dir"] / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    best_accuracy = -1.0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )
        val_stats = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
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

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_stats.accuracy,
            "val_top5_accuracy": val_stats.top5_accuracy,
            "val_macro_f1": val_stats.macro_f1,
            "config": config,
            "metadata": metadata,
        }
        torch.save(checkpoint_payload, run_dirs["checkpoint_dir"] / "last.pt")
        if val_stats.accuracy > best_accuracy:
            best_accuracy = val_stats.accuracy
            torch.save(checkpoint_payload, run_dirs["checkpoint_dir"] / "best.pt")

    log_path = run_dirs["log_dir"] / "history.json"
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_log: {log_path}")
    print(f"checkpoint_dir: {run_dirs['checkpoint_dir']}")


def build_dataloader(
    root: str,
    split: str,
    image_ids_path: str | None,
    image_size: int,
    augmentation: str,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    image_ids = read_image_ids(image_ids_path) if image_ids_path is not None else None
    dataset = CUB200Dataset(
        root=root,
        split=split,
        image_ids=image_ids,
        transform=build_classification_transform(
            image_size=image_size,
            train=train,
            augmentation=augmentation,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def create_run_dirs(
    checkpoint_root: Path,
    log_root: Path,
    experiment_name: str,
) -> dict[str, Path | str]:
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
