from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    embedding: torch.Tensor


class ResNet18Classifier(nn.Module):
    """ResNet-18 classifier that also exposes visual embeddings."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        embedding_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> ModelOutput:
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        return ModelOutput(logits=logits, embedding=embedding)


def build_resnet18_classifier(
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> ResNet18Classifier:
    return ResNet18Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
