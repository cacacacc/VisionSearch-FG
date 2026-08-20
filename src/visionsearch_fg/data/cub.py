from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "test", "all"]


@dataclass(frozen=True)
class CUBSample:
    image_id: int
    image_path: Path
    label: int
    class_name: str
    is_train: bool


class CUB200Dataset(Dataset):
    """CUB-200-2011 dataset reader for classification and embedding learning."""

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in {"train", "test", "all"}:
            msg = f"Unsupported split: {split}. Expected one of: train, test, all."
            raise ValueError(msg)

        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")

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

            cub_class_id = labels[image_id]
            label = cub_class_id - 1
            class_name = class_names.get(cub_class_id, Path(relative_path).parent.name)
            image_path = self.root / "images" / relative_path

            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

            samples.append(
                CUBSample(
                    image_id=image_id,
                    image_path=image_path,
                    label=label,
                    class_name=class_name,
                    is_train=is_train,
                )
            )

        if not samples:
            raise ValueError(f"No samples found for split '{self.split}' under {self.root}.")

        return samples


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


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required CUB metadata file does not exist: {path}")
