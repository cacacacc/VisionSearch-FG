from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

FineTuneMode = Literal["frozen", "partial", "full"]


@dataclass(frozen=True)
class ModelOutput:
    """通用分类输出：类别 logits 加检索 embedding。"""

    logits: torch.Tensor
    embedding: torch.Tensor


class ResNet18Classifier(nn.Module):
    """同时暴露视觉 embedding 的 ResNet-18 分类器。

    原始 ImageNet 全连接层被替换为 identity 模块。得到的 512 维 backbone 输出会
    保留为 embedding，并送入新的 CUB 分类头。
    """

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fine_tune_mode: FineTuneMode | None = None,
        trainable_backbone_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # 移除原始 1000 类分类头，把 ResNet 转换为编码器。
        embedding_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(
            mode=mode,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    def forward(self, images: torch.Tensor) -> ModelOutput:
        """编码一个 batch 并完成分类，同时保留 embedding。"""
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        return ModelOutput(logits=logits, embedding=embedding)

    def set_fine_tune_mode(
        self,
        mode: FineTuneMode,
        trainable_backbone_layers: list[str] | None = None,
    ) -> None:
        """按 frozen、partial 或 full 模式设置 backbone 的可训练参数。"""
        if mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if mode == "frozen":
            return

        if mode != "partial":
            raise ValueError(f"Unsupported fine_tune_mode: {mode}")

        layers = trainable_backbone_layers or ["layer4"]
        for layer_name in layers:
            layer = getattr(self.backbone, layer_name, None)
            if layer is None:
                raise ValueError(f"Unknown ResNet-18 backbone layer: {layer_name}")
            for parameter in layer.parameters():
                parameter.requires_grad = True


def build_resnet18_classifier(
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    fine_tune_mode: FineTuneMode | None = None,
    trainable_backbone_layers: list[str] | None = None,
) -> ResNet18Classifier:
    return ResNet18Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        fine_tune_mode=fine_tune_mode,
        trainable_backbone_layers=trainable_backbone_layers,
    )
