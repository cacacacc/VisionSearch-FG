from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

FineTuneMode = Literal["frozen", "partial", "full"]


@dataclass(frozen=True)
class ModelOutput:
    """Common classifier result: class logits plus the retrieval embedding."""

    logits: torch.Tensor
    embedding: torch.Tensor


class ResNet18Classifier(nn.Module):
    """ResNet-18 classifier that also exposes visual embeddings.

    The original ImageNet fully connected layer is replaced with an identity
    module. The resulting 512-dimensional backbone output is retained as the
    embedding and sent to a new CUB classifier head.
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

        # Removing the original 1000-class head turns ResNet into an encoder.
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
        """Encode a batch and classify it without discarding the embedding."""
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        return ModelOutput(logits=logits, embedding=embedding)

    def set_fine_tune_mode(
        self,
        mode: FineTuneMode,
        trainable_backbone_layers: list[str] | None = None,
    ) -> None:
        """Set trainable backbone parameters for frozen, partial, or full tuning."""
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
