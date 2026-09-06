from __future__ import annotations

from typing import Literal

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AugmentationName = Literal["basic", "hflip", "random_resized_crop", "rrc_hflip_colorjitter"]


def build_classification_transform(
    image_size: int = 224,
    train: bool = True,
    augmentation: AugmentationName = "hflip",
) -> transforms.Compose:
    """Build the image pipeline used by classification and retrieval models.

    Training augmentations are deliberately selected by name so an experiment
    YAML file can change one augmentation policy without changing Python code.
    Validation and test images always use deterministic resizing. All branches
    finish with the ImageNet normalization expected by the pretrained models.
    """
    if train and augmentation == "basic":
        return _with_normalization([transforms.Resize((image_size, image_size))])

    if train and augmentation == "hflip":
        return _with_normalization(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    if train and augmentation == "random_resized_crop":
        return _with_normalization(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            ]
        )

    if train and augmentation == "rrc_hflip_colorjitter":
        return _with_normalization(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            ]
        )

    if train:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    return _with_normalization([transforms.Resize((image_size, image_size))])


class TwoViewTransform:
    """Create two independent views for supervised contrastive learning.

    The same transform object is called twice, but random torchvision
    transforms sample fresh randomness on each call. The two returned tensors
    therefore depict the same source image with potentially different views.
    """

    def __init__(self, base_transform: transforms.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


def build_two_view_transform(
    image_size: int = 224,
    augmentation: AugmentationName = "rrc_hflip_colorjitter",
) -> TwoViewTransform:
    """Build the default two-view augmentation pipeline."""
    return TwoViewTransform(
        build_classification_transform(
            image_size=image_size,
            train=True,
            augmentation=augmentation,
        )
    )


def _with_normalization(transform_steps: list) -> transforms.Compose:
    """Append tensor conversion and ImageNet normalization to a pipeline."""
    return transforms.Compose(
        [
            *transform_steps,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
