from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

from visionsearch_fg.models.resnet import ModelOutput

FineTuneMode = Literal["frozen", "full"]


class SwinTinyClassifier(nn.Module):
    """暴露池化视觉嵌入的 Swin-Tiny 分类器。

    与 ResNet wrapper 类似，这里会移除预训练 ImageNet 分类头，并为 200 个 CUB
    类别接入新的分类器。池化后的 Swin 向量仍会保留给图像检索使用。
    """

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fine_tune_mode: FineTuneMode | None = None,
    ) -> None:
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        backbone = swin_t(weights=weights)

        # Swin 原始分类头会输出 ImageNet 逻辑值，并隐藏检索流水线需要的特征向量。
        embedding_dim = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(mode)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        """返回 CUB logits 和 pooled Swin 表征。"""
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        return ModelOutput(logits=logits, embedding=embedding)

    def set_fine_tune_mode(self, mode: FineTuneMode) -> None:
        """只冻结或解冻 backbone；新增分类器始终保持可训练。"""
        if mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return

        if mode == "frozen":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            return

        raise ValueError(f"Unsupported Swin-Tiny fine_tune_mode: {mode}")


def build_swin_tiny_classifier(
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    fine_tune_mode: FineTuneMode | None = None,
) -> SwinTinyClassifier:
    return SwinTinyClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        fine_tune_mode=fine_tune_mode,
    )
