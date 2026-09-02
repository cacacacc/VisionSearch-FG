from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ContrastiveOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor


class ContrastiveClassifier(nn.Module):
    """Classifier with an additional projection head for contrastive learning."""

    def __init__(
        self,
        classifier: nn.Module,
        projection_head: str = "mlp",
        projection_dim: int = 128,
        projection_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if projection_head not in {"mlp", "identity"}:
            raise ValueError("projection_head must be one of: mlp, identity")
        if projection_dim < 1:
            raise ValueError("projection_dim must be greater than or equal to 1")

        embedding_dim = classifier.embedding_dim
        if projection_head == "identity" and projection_dim != embedding_dim:
            raise ValueError("identity projection_head requires projection_dim == embedding_dim")

        hidden_dim = projection_hidden_dim or embedding_dim
        self.backbone = classifier.backbone
        self.classifier = classifier.classifier
        self.embedding_dim = embedding_dim
        self.projection_head_type = projection_head
        self.projection_dim = projection_dim
        self.num_classes = classifier.num_classes
        if projection_head == "identity":
            self.projection_head = nn.Identity()
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            )

    def forward(self, images: torch.Tensor) -> ContrastiveOutput:
        embedding = self.backbone(images)
        if embedding.ndim == 4:
            embedding = embedding.mean(dim=(-2, -1))
        elif embedding.ndim > 2:
            embedding = torch.flatten(embedding, start_dim=1)
        logits = self.classifier(embedding)
        projection = self.projection_head(embedding)
        return ContrastiveOutput(
            logits=logits,
            embedding=embedding,
            projection=projection,
        )


def build_contrastive_classifier(
    classifier: nn.Module,
    projection_head: str = "mlp",
    projection_dim: int = 128,
    projection_hidden_dim: int | None = None,
) -> ContrastiveClassifier:
    return ContrastiveClassifier(
        classifier=classifier,
        projection_head=projection_head,
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
    )
