from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

from visionsearch_fg.models.resnet import ModelOutput

FineTuneMode = Literal["frozen", "full"]


class SwinTinyClassifier(nn.Module):
    """Swin-Tiny classifier that exposes the pooled visual embedding."""

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

        embedding_dim = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(mode)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        return ModelOutput(logits=logits, embedding=embedding)

    def set_fine_tune_mode(self, mode: FineTuneMode) -> None:
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
