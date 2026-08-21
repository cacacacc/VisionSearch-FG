from __future__ import annotations

from pathlib import Path

from visionsearch_fg.data import read_image_ids, stratified_train_val_split, write_image_ids
from visionsearch_fg.data.cub import CUBSample


def test_stratified_train_val_split_keeps_each_class_in_both_sets() -> None:
    samples = [
        CUBSample(i, Path(f"{i}.jpg"), label=0, class_name="a", is_train=True)
        for i in range(1, 6)
    ] + [
        CUBSample(i, Path(f"{i}.jpg"), label=1, class_name="b", is_train=True)
        for i in range(6, 11)
    ]

    train_ids, val_ids = stratified_train_val_split(samples=samples, val_ratio=0.2, seed=42)

    assert len(train_ids) == 8
    assert len(val_ids) == 2
    assert set(train_ids).isdisjoint(val_ids)
    assert {sample.label for sample in samples if sample.image_id in train_ids} == {0, 1}
    assert {sample.label for sample in samples if sample.image_id in val_ids} == {0, 1}


def test_image_id_files_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"

    write_image_ids(path, [3, 1, 2])

    assert read_image_ids(path) == [3, 1, 2]
