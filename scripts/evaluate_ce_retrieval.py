from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from visionsearch_fg.data import CUB200Dataset, build_classification_transform, read_image_ids
from visionsearch_fg.models import build_resnet18_classifier, build_swin_tiny_classifier
from visionsearch_fg.retrieval import (
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    evaluate_retrieval,
    rank_gallery_by_distance,
    rank_gallery_for_queries,
)
from visionsearch_fg.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CE-trained classifier embeddings for retrieval."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--ids-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/embeddings/ce_retrieval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    device = resolve_device(args.device)

    data_config = config["data"]
    model_config = config["model"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else data_config["num_workers"]

    dataset = CUB200Dataset(
        root=data_config["root"],
        split=args.split,
        image_ids=read_image_ids(args.ids_path),
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

    embeddings, records = extract_embeddings(model=model, dataloader=dataloader, device=device)
    labels = np.array([record["label"] for record in records], dtype=np.int64)
    metrics = evaluate_retrieval(
        embeddings=embeddings,
        labels=labels,
        recall_ks=(1, 5, 10),
        metric=args.metric,
    )

    run_id = args.checkpoint.parent.name
    output_dir = args.output_dir / run_id / args.metric
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    write_records_csv(records, output_dir / "records.csv")

    top_results = build_top_results(
        embeddings=embeddings,
        records=records,
        top_k=10,
        metric=args.metric,
    )
    (output_dir / "top_results.json").write_text(
        json.dumps(top_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": args.split,
        "ids_path": str(args.ids_path),
        "backbone": model_config.get("backbone", "resnet18"),
        "metric": args.metric,
        "num_samples": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        **metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    embeddings: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="extract", leave=False):
        images = batch["image"].to(device)
        output = model(images)
        embeddings.append(output.embedding.cpu().numpy())

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

    return np.concatenate(embeddings, axis=0), records


def build_top_results(
    embeddings: np.ndarray,
    records: list[dict[str, Any]],
    top_k: int,
    metric: str,
) -> list[dict[str, Any]]:
    if metric == "cosine":
        score_matrix = cosine_similarity_matrix(embeddings)
        ranked_indices = rank_gallery_for_queries(score_matrix, exclude_self=True)
        score_name = "similarity"
    elif metric == "euclidean":
        score_matrix = euclidean_distance_matrix(embeddings)
        ranked_indices = rank_gallery_by_distance(score_matrix, exclude_self=True)
        score_name = "distance"
    else:
        raise ValueError(f"Unsupported retrieval metric: {metric}")

    results: list[dict[str, Any]] = []
    for query_index, query_record in enumerate(records):
        neighbors = []
        for gallery_index in ranked_indices[query_index, :top_k]:
            gallery_record = records[int(gallery_index)]
            neighbors.append(
                {
                    "image_id": gallery_record["image_id"],
                    "label": gallery_record["label"],
                    "class_name": gallery_record["class_name"],
                    "path": gallery_record["path"],
                    score_name: float(score_matrix[query_index, gallery_index]),
                    "same_class": gallery_record["label"] == query_record["label"],
                }
            )
        results.append({"query": query_record, "neighbors": neighbors})
    return results


def build_model(model_config: dict) -> torch.nn.Module:
    backbone = model_config.get("backbone", "resnet18")
    if backbone == "resnet18":
        return build_resnet18_classifier(
            num_classes=model_config["num_classes"],
            pretrained=False,
            freeze_backbone=model_config.get("freeze_backbone", False),
            fine_tune_mode=model_config.get("fine_tune_mode"),
            trainable_backbone_layers=model_config.get("trainable_backbone_layers"),
        )
    if backbone in {"swin_tiny", "swin_t"}:
        return build_swin_tiny_classifier(
            num_classes=model_config["num_classes"],
            pretrained=False,
            freeze_backbone=model_config.get("freeze_backbone", False),
            fine_tune_mode=model_config.get("fine_tune_mode"),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "label", "class_name", "path"])
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
