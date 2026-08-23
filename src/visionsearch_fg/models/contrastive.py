from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from visionsearch_fg.models.resnet import ResNet18Classifier


@dataclass(frozen=True)
class ContrastiveOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor


class ContrastiveClassifier(nn.Module):
    """Classifier with an additional projection head for contrastive learning."""

    def __init__(
        self,
        classifier: ResNet18Classifier,
        projection_dim: int = 128,
        projection_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if projection_dim < 1:
            raise ValueError("projection_dim must be greater than or equal to 1")

        embedding_dim = classifier.embedding_dim
        hidden_dim = projection_hidden_dim or embedding_dim
        self.backbone = classifier.backbone
        self.classifier = classifier.classifier
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_classes = classifier.num_classes
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> ContrastiveOutput:
        embedding = self.backbone(images)
        logits = self.classifier(embedding)
        projection = self.projection_head(embedding)
        return ContrastiveOutput(
            logits=logits,
            embedding=embedding,
            projection=projection,
        )


def build_contrastive_classifier(
    classifier: ResNet18Classifier,
    projection_dim: int = 128,
    projection_hidden_dim: int | None = None,
) -> ContrastiveClassifier:
    return ContrastiveClassifier(
        classifier=classifier,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
    )
