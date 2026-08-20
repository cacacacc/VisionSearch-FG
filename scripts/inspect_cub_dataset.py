from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from visionsearch_fg.data import CUB200Dataset, build_classification_transform
from visionsearch_fg.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CUB-200-2011 dataset loading.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_resnet18.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="train",
        help="Dataset split to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_config = config["data"]

    transform = build_classification_transform(
        image_size=data_config["image_size"],
        train=args.split == "train",
    )
    dataset = CUB200Dataset(
        root=data_config["root"],
        split=args.split,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"split: {args.split}")
    print(f"num_samples: {len(dataset)}")
    print(f"image_batch_shape: {tuple(batch['image'].shape)}")
    print(f"label_batch_shape: {tuple(batch['label'].shape)}")
    print(f"first_image_id: {batch['image_id'][0].item()}")
    print(f"first_class_name: {batch['class_name'][0]}")


if __name__ == "__main__":
    main()
