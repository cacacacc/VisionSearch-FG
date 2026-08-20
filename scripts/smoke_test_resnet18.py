from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from visionsearch_fg.models import build_resnet18_classifier


def main() -> None:
    model = build_resnet18_classifier(num_classes=200, pretrained=False)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(images)

    print(f"input_shape: {tuple(images.shape)}")
    print(f"embedding_shape: {tuple(output.embedding.shape)}")
    print(f"logits_shape: {tuple(output.logits.shape)}")


if __name__ == "__main__":
    main()
