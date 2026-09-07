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
    """用于 margin 分类和可选对比学习的输出集合。"""

    logits: torch.Tensor
    embedding: torch.Tensor
    projection: torch.Tensor | None = None


class AngularMarginHead(nn.Module):
    """作用在 L2-normalized embedding 上的 ArcFace/CosFace 分类头。

    归一化后的类别权重和 embedding 会让线性输出变成 cosine similarity。
    ArcFace 在角度空间加入 margin；CosFace 直接从目标类别 cosine 中减去 margin。
    scale 用于在归一化后恢复适合 cross-entropy 的 logit 数值范围。
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
        """在提供 labels 时生成加入 margin 的 logits。

        labels 是可选的，因此同一个 head 可以在推理阶段生成不修改目标类别的 logits。
        训练阶段只有真实类别 logit 会加入 angular margin。
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
    """用 ArcFace/CosFace head 包装一个 embedding 分类器。

    base classifier 提供 backbone，普通线性分类层会被 margin head 替换。
    可选 projection head 让该模型兼容 margin 与 SupCon 的联合实验。
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
        """编码图像，并把所选 margin head 应用到对应 labels。"""
        embedding = self.backbone(images)
        if embedding.ndim == 4:
            embedding = embedding.mean(dim=(-2, -1))
        elif embedding.ndim > 2:
            embedding = torch.flatten(embedding, start_dim=1)
        logits = self.margin_head(embedding, labels=labels)
        projection = self.projection_head(embedding) if self.projection_head is not None else None
        return AngularMarginOutput(logits=logits, embedding=embedding, projection=projection)


def default_margin_for_type(margin_type: MarginType) -> float:
    """返回项目中 ArcFace 和 CosFace 实验使用的默认 margin。"""
    if margin_type == "arcface":
        return 0.5
    if margin_type == "cosface":
        return 0.35
    raise ValueError(f"Unsupported margin_type: {margin_type}")


def degrees_from_radians(value: float) -> float:
    """将弧度制 angular margin 转为角度，便于报告展示。"""
    return value * 180.0 / math.pi
