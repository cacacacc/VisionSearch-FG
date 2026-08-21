from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from itertools import islice
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from visionsearch_fg.analysis import build_prediction_records, summarize_prediction_records
from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.models import build_resnet18_classifier
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CUB baseline predictions.")
    parser.add_argument("--config", type=Path, required=True, help="Training config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--ids-path",
        type=Path,
        default=None,
        help="Optional image id file used to filter the selected CUB split.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/baseline_error_analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    device = resolve_device(args.device)

    data_config = config["data"]
    model_config = config["model"]
    batch_size = args.batch_size or config["training"]["batch_size"]

    dataset = CUB200Dataset(
        root=data_config["root"],
        split=args.split,
        image_ids=read_image_ids(args.ids_path) if args.ids_path is not None else None,
        transform=build_classification_transform(image_size=data_config["image_size"], train=False),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18_classifier(
        num_classes=model_config["num_classes"],
        pretrained=False,
        freeze_backbone=model_config.get("freeze_backbone", False),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = load_class_names(Path(data_config["root"]))
    records = collect_records(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        device=device,
        max_batches=args.max_batches,
    )
    summary = summarize_prediction_records(records)

    output_dir = args.output_dir / args.checkpoint.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_predictions_csv(records, output_dir / "predictions.csv")

    wrong_records = [record for record in records if not record["correct"]]
    high_confidence_wrong = sorted(
        wrong_records,
        key=lambda record: float(record["pred_probability"]),
        reverse=True,
    )
    grid_path = output_dir / "high_confidence_wrong_examples.jpg"
    if high_confidence_wrong:
        save_prediction_grid(
            records=high_confidence_wrong[: args.num_examples],
            output_path=grid_path,
            title="High-confidence wrong predictions",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


@torch.no_grad()
def collect_records(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device,
    max_batches: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    batches = dataloader if max_batches is None else islice(dataloader, max_batches)

    for batch in tqdm(batches, desc="analyze", total=total_batches, leave=False):

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        image_ids = batch["image_id"]
        paths = list(batch["path"])
        true_class_names = list(batch["class_name"])

        output = model(images)
        records.extend(
            build_prediction_records(
                logits=output.logits.cpu(),
                labels=labels.cpu(),
                image_ids=image_ids.cpu(),
                paths=paths,
                true_class_names=true_class_names,
                class_names=class_names,
                top_k=5,
            )
        )

    return records


def write_predictions_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "image_id",
        "path",
        "true_label",
        "true_class",
        "pred_label",
        "pred_class",
        "correct",
        "top_k_correct",
        "true_probability",
        "pred_probability",
        "top_k_labels",
        "top_k_classes",
        "top_k_probabilities",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["top_k_labels"] = ";".join(str(label) for label in record["top_k_labels"])
            row["top_k_classes"] = ";".join(record["top_k_classes"])
            row["top_k_probabilities"] = ";".join(
                f"{probability:.6f}" for probability in record["top_k_probabilities"]
            )
            writer.writerow(row)


def save_prediction_grid(records: list[dict[str, Any]], output_path: Path, title: str) -> None:
    columns = 4
    image_size = 180
    label_height = 78
    title_height = 34
    padding = 12
    rows = (len(records) + columns - 1) // columns

    canvas_width = columns * image_size + (columns + 1) * padding
    canvas_height = title_height + rows * (image_size + label_height) + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), title, fill=(20, 20, 20))

    for index, record in enumerate(records):
        row = index // columns
        column = index % columns
        x = padding + column * (image_size + padding)
        y = title_height + padding + row * (image_size + label_height + padding)

        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((image_size, image_size))
            tile = Image.new("RGB", (image_size, image_size), (245, 245, 245))
            paste_x = (image_size - image.width) // 2
            paste_y = (image_size - image.height) // 2
            tile.paste(image, (paste_x, paste_y))

        canvas.paste(tile, (x, y))
        text = format_record_label(record)
        draw.multiline_text((x, y + image_size + 4), text, fill=(20, 20, 20), spacing=3)

    canvas.save(output_path, quality=95)


def format_record_label(record: dict[str, Any]) -> str:
    true_class = textwrap.shorten(record["true_class"], width=32, placeholder="...")
    pred_class = textwrap.shorten(record["pred_class"], width=32, placeholder="...")
    probability = float(record["pred_probability"])
    return f"id {record['image_id']}\nT: {true_class}\nP: {pred_class}\np={probability:.3f}"


def load_class_names(root: Path) -> list[str]:
    class_path = root / "classes.txt"
    class_names: list[str] = []
    for line in class_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, class_name = line.split(maxsplit=1)
        zero_based_label = int(class_id) - 1
        while len(class_names) <= zero_based_label:
            class_names.append("")
        class_names[zero_based_label] = class_name
    return class_names


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
