from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionsearch_fg.data import (
    CUB200Dataset,
    stratified_train_val_split,
    write_image_ids,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed CUB train/val/test split files.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/CUB_200_2011"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_train = CUB200Dataset(root=args.root, split="train")
    official_test = CUB200Dataset(root=args.root, split="test")

    train_ids, val_ids = stratified_train_val_split(
        samples=official_train.samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    test_ids = sorted(sample.image_id for sample in official_test.samples)

    train_path = args.output_dir / f"cub_train_ids_seed{args.seed}.txt"
    val_path = args.output_dir / f"cub_val_ids_seed{args.seed}.txt"
    test_path = args.output_dir / "cub_test_ids.txt"
    manifest_path = args.output_dir / f"cub_split_manifest_seed{args.seed}.json"

    write_image_ids(train_path, train_ids)
    write_image_ids(val_path, val_ids)
    write_image_ids(test_path, test_ids)

    manifest = {
        "dataset": "CUB-200-2011",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "official_train_samples": len(official_train),
        "official_test_samples": len(official_test),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "train_ids_path": str(train_path),
        "val_ids_path": str(val_path),
        "test_ids_path": str(test_path),
    }
    write_split_manifest(manifest_path, manifest)

    print(f"train_samples: {len(train_ids)}")
    print(f"val_samples: {len(val_ids)}")
    print(f"test_samples: {len(test_ids)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
