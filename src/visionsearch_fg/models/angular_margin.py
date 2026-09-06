from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

MarginType = Literal["arcface", "cosface"]


@dataclass(frozen=True)
class AngularMarginOutput:
    """Output bundle for margin-based classification and optional contrastive use."""

    logits: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor | None = None


class AngularMarginHead(nn.Module):
    """ArcFace/CosFace classification head over L2-normalized embeddings.

    Normalized class weights and embeddings make the linear output a cosine
    similarity. ArcFace adds the margin in angle space; CosFace subtracts it
    directly from the target cosine. The scale restores a useful logit range
    for cross-entropy after normalization.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin_type: MarginType = "arcface",
        scale: float = 30.0,
        margin: float = 0.5,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if margin_type not in {"arcface", "cosface"}:
            raise ValueError("margin_type must be one of: arcface, cosface")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be greater than or equal to 1")
        if num_classes < 2:
            raise ValueError("num_classes must be greater than or equal to 2")
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        if margin < 0:
            raise ValueError("margin must be non-negative")

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin_type = margin_type
        self.scale = scale
        self.margin = margin
        self.eps = eps
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Produce margin-adjusted logits when labels are supplied.

        Labels are optional so the same head can produce inference logits
        without modifying a target class. During training, only the target
        class logit receives the angular margin.
        """
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        if labels is None:
            return cosine * self.scale

        if labels.ndim != 1:
            raise ValueError("labels must have shape [batch_size]")
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels and embeddings must have the same batch size")

        target_logits = cosine[torch.arange(labels.shape[0], device=labels.device), labels]
        if self.margin_type == "arcface":
            target_logits = torch.cos(
                torch.acos(target_logits.clamp(-1.0 + self.eps, 1.0 - self.eps)) + self.margin
            )
        else:
            target_logits = target_logits - self.margin

        logits = cosine.clone()
        logits[torch.arange(labels.shape[0], device=labels.device), labels] = target_logits
        return logits * self.scale


class AngularMarginClassifier(nn.Module):
    """Wrap an embedding classifier with an ArcFace/CosFace head.

    The base classifier contributes its backbone, while the ordinary linear
    classification layer is replaced by the margin head. An optional projection
    head keeps this model compatible with joint margin and SupCon experiments.
    """

    def __init__(
        self,
        base_classifier: nn.Module,
        margin_type: MarginType = "arcface",
        scale: float = 30.0,
        margin: float = 0.5,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(base_classifier, "backbone"):
            raise ValueError("base_classifier must expose a backbone module")
        if not hasattr(base_classifier, "embedding_dim"):
            raise ValueError("base_classifier must expose embedding_dim")
        if not hasattr(base_classifier, "num_classes"):
            raise ValueError("base_classifier must expose num_classes")

        self.backbone = base_classifier.backbone
        self.embedding_dim = int(base_classifier.embedding_dim)
        self.num_classes = int(base_classifier.num_classes)
        self.margin_head = AngularMarginHead(
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            margin_type=margin_type,
            scale=scale,
            margin=margin,
        )
        self.margin_type = margin_type
        self.scale = scale
        self.margin = margin

        self.projection_dim = projection_dim
        if projection_dim is None:
            self.projection_head = None
        else:
            if projection_dim < 1:
                raise ValueError("projection_dim must be greater than or equal to 1")
            hidden_dim = projection_hidden_dim or self.embedding_dim
            self.projection_head = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> AngularMarginOutput:
        """Encode images and apply the selected margin head to their labels."""
        embedding = self.backbone(images)
        if embedding.ndim == 4:
            embedding = embedding.mean(dim=(-2, -1))
        elif embedding.ndim > 2:
            embedding = torch.flatten(embedding, start_dim=1)
        logits = self.margin_head(embedding, labels=labels)
        projection = self.projection_head(embedding) if self.projection_head is not None else None
        return AngularMarginOutput(logits=logits, embedding=embedding, projection=projection)


def default_margin_for_type(margin_type: MarginType) -> float:
    """Return the project defaults used for ArcFace and CosFace experiments."""
    if margin_type == "arcface":
        return 0.5
    if margin_type == "cosface":
        return 0.35
    raise ValueError(f"Unsupported margin_type: {margin_type}")


def degrees_from_radians(value: float) -> float:
    """Convert an angular margin from radians to degrees for readable reports."""
    return value * 180.0 / math.pi
