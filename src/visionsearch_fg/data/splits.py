from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from visionsearch_fg.data.cub import CUBSample


def stratified_train_val_split(
    samples: list[CUBSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """在保留每个类别的前提下划分官方训练样本。

    划分会按 label 独立进行，并使用局部 seeded generator 打乱。返回值是排序后的
    image ID，排序可以让生成文件可复现、易于 diff，且不改变成员关系。官方 CUB
    test set 不会传入该函数，因此不会被意外混入 validation。
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    ids_by_label: dict[int, list[int]] = defaultdict(list)
    for sample in samples:
        ids_by_label[sample.label].append(sample.image_id)

    rng = random.Random(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []

    for label, image_ids in sorted(ids_by_label.items()):
        shuffled_ids = sorted(image_ids)
        rng.shuffle(shuffled_ids)
        val_count = round(len(shuffled_ids) * val_ratio)
        val_count = max(1, val_count)
        if val_count >= len(shuffled_ids):
            raise ValueError(f"Class {label} does not have enough samples to split")

        val_ids.extend(shuffled_ids[:val_count])
        train_ids.extend(shuffled_ids[val_count:])

    return sorted(train_ids), sorted(val_ids)


def read_image_ids(path: str | Path) -> list[int]:
    """按行读取 image ID，并容忍空行和 UTF-8 BOM。"""
    image_ids: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if stripped:
            image_ids.append(int(stripped))
    return image_ids


def write_image_ids(path: str | Path, image_ids: list[int]) -> None:
    """创建父目录，并写入确定性的 ID 列表。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(image_id) for image_id in image_ids) + "\n",
        encoding="utf-8",
    )


def write_split_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """将 split 来源信息写成缩进 JSON，便于后续实验审计。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
