from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from visionsearch_fg.data import CUB200Dataset, build_classification_transform


def _write_fake_cub(root: Path) -> None:
    image_dir = root / "images" / "001.Black_footed_Albatross"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_dir / "image_0001.jpg")
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_dir / "image_0002.jpg")

    (root / "images.txt").write_text(
        "\n".join(
            [
                "1 001.Black_footed_Albatross/image_0001.jpg",
                "2 001.Black_footed_Albatross/image_0002.jpg",
            ]
        ),
        encoding="utf-8",
    )
    (root / "image_class_labels.txt").write_text("1 1\n2 1\n", encoding="utf-8")
    (root / "train_test_split.txt").write_text("1 1\n2 0\n", encoding="utf-8")
    (root / "classes.txt").write_text("1 001.Black_footed_Albatross\n", encoding="utf-8")


def test_cub_dataset_reads_train_split_and_zero_indexes_labels(tmp_path: Path) -> None:
    _write_fake_cub(tmp_path)

    transform = build_classification_transform(image_size=64, train=False)
    dataset = CUB200Dataset(root=tmp_path, split="train", transform=transform)

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["label"] == 0
    assert sample["image_id"] == 1
    assert sample["class_name"] == "001.Black_footed_Albatross"
    assert tuple(sample["image"].shape) == (3, 64, 64)


def test_cub_dataset_reads_test_split(tmp_path: Path) -> None:
    _write_fake_cub(tmp_path)

    dataset = CUB200Dataset(root=tmp_path, split="test")

    assert len(dataset) == 1
    assert dataset[0]["image_id"] == 2


def test_cub_dataset_filters_by_image_ids_inside_selected_split(tmp_path: Path) -> None:
    _write_fake_cub(tmp_path)

    dataset = CUB200Dataset(root=tmp_path, split="all", image_ids={2})

    assert len(dataset) == 1
    assert dataset[0]["image_id"] == 2


def test_cub_dataset_applies_bbox_crop_before_transform(tmp_path: Path) -> None:
    _write_fake_cub(tmp_path)
    image_path = tmp_path / "images" / "001.Black_footed_Albatross" / "image_0001.jpg"
    image = Image.new("RGB", (40, 30), color=(255, 0, 0))
    for x in range(10, 30):
        for y in range(5, 15):
            image.putpixel((x, y), (0, 255, 0))
    image.save(image_path)
    (tmp_path / "bounding_boxes.txt").write_text("1 10 5 20 10\n2 0 0 32 32\n", encoding="utf-8")

    dataset = CUB200Dataset(root=tmp_path, split="train", crop_mode="bbox")
    sample = dataset[0]
    red, green, blue = sample["image"].getpixel((0, 0))

    assert sample["image"].size == (20, 10)
    assert green > red
    assert green > blue


def test_cub_dataset_requires_bbox_file_when_bbox_crop_is_enabled(tmp_path: Path) -> None:
    _write_fake_cub(tmp_path)

    with pytest.raises(FileNotFoundError):
        CUB200Dataset(root=tmp_path, split="train", crop_mode="bbox")
