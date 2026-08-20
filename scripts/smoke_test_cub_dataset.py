from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from visionsearch_fg.data import CUB200Dataset, build_classification_transform


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
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

        transform = build_classification_transform(image_size=64, train=False)
        train_dataset = CUB200Dataset(root=root, split="train", transform=transform)
        test_dataset = CUB200Dataset(root=root, split="test", transform=transform)
        sample = train_dataset[0]

        print(f"train_samples: {len(train_dataset)}")
        print(f"test_samples: {len(test_dataset)}")
        print(f"sample_label: {sample['label']}")
        print(f"sample_image_shape: {tuple(sample['image'].shape)}")
        print(f"sample_class_name: {sample['class_name']}")


if __name__ == "__main__":
    main()
