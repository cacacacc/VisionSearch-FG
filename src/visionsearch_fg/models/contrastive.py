from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ContrastiveOutput:
    """集中保存分类、检索和对比学习表征。"""

    logits: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor


class ContrastiveClassifier(nn.Module):
    """带额外 projection head 的对比学习分类器。

    分类器 embedding 会保留给检索使用。projection 是 SupCon 使用的独立空间，
    因此对比学习目标不会强制检索特征维度等于 projection 维度。
    """

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
        # 复用基础编码器和分类器，避免创建重复参数，并保留既定微调策略。
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
        """返回 logits、检索 embedding 和 SupCon projection。"""
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
