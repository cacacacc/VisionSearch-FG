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
    """构建分类和检索模型共用的图像处理流水线。

    训练增强通过名称选择，使实验 YAML 可以只改变数据增强策略而不修改
    Python 代码。验证和测试图像始终使用确定性的 resize。所有分支最后都会执行
    预训练模型所需的 ImageNet 归一化。
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
    """为监督式对比学习创建两个独立视图。

    同一个变换对象会被调用两次，但随机 torchvision 变换每次调用都会
    重新采样随机性。因此返回的两个张量来自同一张原图，但可能具有不同增强视角。
    """

    def __init__(self, base_transform: transforms.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


def build_two_view_transform(
    image_size: int = 224,
    augmentation: AugmentationName = "rrc_hflip_colorjitter",
) -> TwoViewTransform:
    """构建默认的双视图数据增强流水线。"""
    return TwoViewTransform(
        build_classification_transform(
            image_size=image_size,
            train=True,
            augmentation=augmentation,
        )
    )


def _with_normalization(transform_steps: list) -> transforms.Compose:
    """在流水线末尾追加张量转换和 ImageNet 归一化。"""
    return transforms.Compose(
        [
            *transform_steps,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
