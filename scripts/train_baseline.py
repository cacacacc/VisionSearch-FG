from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from visionsearch_fg.data import CUB200Dataset, build_classification_transform
from visionsearch_fg.engine import train_one_epoch, validate
from visionsearch_fg.models import build_resnet18_classifier
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ResNet-18 CUB baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_resnet18.yaml"),
        help="Path to the training config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--pretrained",
        choices=["true", "false"],
        default=None,
        help="Override whether to use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the ResNet backbone and only train the classifier head.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches for CPU smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation batches for CPU smoke tests.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config["project"]["seed"])

    device = resolve_device(args.device)
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]
    output_config = config["outputs"]

    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else training_config["batch_size"]
    pretrained = (
        args.pretrained.lower() == "true"
        if args.pretrained is not None
        else model_config["pretrained"]
    )

    train_loader = build_dataloader(
        root=data_config["root"],
        split="train",
        image_size=data_config["image_size"],
        batch_size=batch_size,
        num_workers=data_config["num_workers"],
        train=True,
    )
    val_loader = build_dataloader(
        root=data_config["root"],
        split="test",
        image_size=data_config["image_size"],
        batch_size=batch_size,
        num_workers=data_config["num_workers"],
        train=False,
    )

    model = build_resnet18_classifier(
        num_classes=model_config["num_classes"],
        pretrained=pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    log_dir = Path(output_config["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    history: list[dict] = []
    print(f"device: {device}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"pretrained: {pretrained}")
    print(f"freeze_backbone: {args.freeze_backbone}")

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )
        val_stats = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_accuracy": train_stats.accuracy,
            "val_loss": val_stats.loss,
            "val_accuracy": val_stats.accuracy,
            "val_top5_accuracy": val_stats.top5_accuracy,
            "train_samples": train_stats.num_samples,
            "val_samples": val_stats.num_samples,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        if val_stats.accuracy > best_accuracy:
            best_accuracy = val_stats.accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_stats.accuracy,
                    "val_top5_accuracy": val_stats.top5_accuracy,
                    "config": config,
                },
                checkpoint_dir / "baseline_resnet18_best.pt",
            )

    log_path = log_dir / "baseline_resnet18_history.json"
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_log: {log_path}")


def build_dataloader(
    root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    dataset = CUB200Dataset(
        root=root,
        split=split,
        transform=build_classification_transform(image_size=image_size, train=train),
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
