from __future__ import annotations

import pytest
import torch
from PIL import Image

from visionsearch_fg.data import build_classification_transform


@pytest.mark.parametrize(
    "augmentation",
    ["basic", "hflip", "random_resized_crop", "rrc_hflip_colorjitter"],
)
def test_classification_train_transforms_return_expected_shape(augmentation: str) -> None:
    image = Image.new("RGB", (96, 80), color=(128, 64, 32))
    transform = build_classification_transform(
        image_size=64,
        train=True,
        augmentation=augmentation,
    )

    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (3, 64, 64)


def test_classification_transform_rejects_unknown_augmentation() -> None:
    with pytest.raises(ValueError):
        build_classification_transform(image_size=64, train=True, augmentation="unknown")
