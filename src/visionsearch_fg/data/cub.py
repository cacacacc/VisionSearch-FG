from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

Split = Literal["train", "test", "all"]
CropMode = Literal["none", "bbox"]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CUBSample:
    image_id: int
    image_path: Path
    label: int
    class_name: str
    is_train: bool
    bbox: BBox | None = None


class CUB200Dataset(Dataset):
    """CUB-200-2011 dataset reader for classification and embedding learning."""

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        transform: Callable | None = None,
        image_ids: Collection[int] | None = None,
        crop_mode: CropMode = "none",
        bbox_margin: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_ids = set(image_ids) if image_ids is not None else None
        self.crop_mode = crop_mode
        self.bbox_margin = bbox_margin

        if split not in {"train", "test", "all"}:
            msg = f"Unsupported split: {split}. Expected one of: train, test, all."
            raise ValueError(msg)
        if crop_mode not in {"none", "bbox"}:
            msg = f"Unsupported crop_mode: {crop_mode}. Expected one of: none, bbox."
            raise ValueError(msg)
        if bbox_margin < 0:
            raise ValueError("bbox_margin must be non-negative.")

        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")

        if self.crop_mode == "bbox":
            if sample.bbox is None:
                raise ValueError(f"Missing bounding box for image id {sample.image_id}.")
            image = crop_image_to_bbox(image, sample.bbox, margin=self.bbox_margin)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample.label,
            "image_id": sample.image_id,
            "path": str(sample.image_path),
            "class_name": sample.class_name,
            "is_train": sample.is_train,
        }

    def _load_samples(self) -> list[CUBSample]:
        images = _read_id_text_file(self.root / "images.txt")
        labels = _read_id_int_file(self.root / "image_class_labels.txt")
        split_flags = _read_id_int_file(self.root / "train_test_split.txt")
        class_names = _read_id_text_file(self.root / "classes.txt")
        bounding_boxes = (
            _read_id_bbox_file(self.root / "bounding_boxes.txt")
            if self.crop_mode == "bbox"
            else {}
        )

        samples: list[CUBSample] = []
        for image_id, relative_path in sorted(images.items()):
            if image_id not in labels:
                raise ValueError(f"Missing label for image id {image_id}.")
            if image_id not in split_flags:
                raise ValueError(f"Missing train/test split flag for image id {image_id}.")

            is_train = split_flags[image_id] == 1
            if self.split == "train" and not is_train:
                continue
            if self.split == "test" and is_train:
                continue
            if self.image_ids is not None and image_id not in self.image_ids:
                continue

            cub_class_id = labels[image_id]
            label = cub_class_id - 1
            class_name = class_names.get(cub_class_id, Path(relative_path).parent.name)
            image_path = self.root / "images" / relative_path
            bbox = bounding_boxes.get(image_id)

            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            if self.crop_mode == "bbox" and bbox is None:
                raise ValueError(f"Missing bounding box for image id {image_id}.")

            samples.append(
                CUBSample(
                    image_id=image_id,
                    image_path=image_path,
                    label=label,
                    class_name=class_name,
                    is_train=is_train,
                    bbox=bbox,
                )
            )

        if not samples:
            raise ValueError(f"No samples found for split '{self.split}' under {self.root}.")

        return samples


def crop_image_to_bbox(image: Image.Image, bbox: BBox, margin: float = 0.0) -> Image.Image:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounding box with non-positive size: {bbox}")

    image_width, image_height = image.size
    x_min = max(0, int(round(x - width * margin)))
    y_min = max(0, int(round(y - height * margin)))
    x_max = min(image_width, int(round(x + width + width * margin)))
    y_max = min(image_height, int(round(y + height + height * margin)))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Bounding box is outside image bounds: bbox={bbox}, image_size={image.size}")

    return image.crop((x_min, y_min, x_max, y_max))


def _read_id_text_file(path: Path) -> dict[int, str]:
    _ensure_file(path)
    rows: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item_id, value = line.split(maxsplit=1)
        rows[int(item_id)] = value
    return rows


def _read_id_int_file(path: Path) -> dict[int, int]:
    _ensure_file(path)
    rows: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item_id, value = line.split(maxsplit=1)
        rows[int(item_id)] = int(value)
    return rows


def _read_id_bbox_file(path: Path) -> dict[int, BBox]:
    _ensure_file(path)
    rows: dict[int, BBox] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item_id, x, y, width, height = line.split(maxsplit=4)
        rows[int(item_id)] = (float(x), float(y), float(width), float(height))
    return rows


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required CUB metadata file does not exist: {path}")
