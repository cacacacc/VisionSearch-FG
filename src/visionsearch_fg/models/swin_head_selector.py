from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t

FineTuneMode = Literal["frozen", "full"]


@dataclass(frozen=True)
class HeadSelectorOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    global_embedding: torch.Tensor
    local_embedding: torch.Tensor
    selector_logits: torch.Tensor
    selector_weights: torch.Tensor


class SwinHeadAwareClassifier(nn.Module):
    """Swin-Tiny with a learned soft local selector over final-stage tokens."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fine_tune_mode: FineTuneMode | None = None,
        selector_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if selector_temperature <= 0:
            raise ValueError("selector_temperature must be positive")

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        backbone = swin_t(weights=weights)
        embedding_dim = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone
        self.selector = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Linear(embedding_dim * 2, num_classes)
        self.embedding_dim = embedding_dim * 2
        self.token_dim = embedding_dim
        self.num_classes = num_classes
        self.selector_temperature = selector_temperature

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(mode)

    def forward(self, images: torch.Tensor) -> HeadSelectorOutput:
        tokens = self.extract_tokens(images)
        global_embedding = tokens.mean(dim=1)
        selector_logits = self.selector(tokens).squeeze(-1)
        selector_weights = torch.softmax(selector_logits / self.selector_temperature, dim=1)
        local_embedding = torch.einsum("bn,bnc->bc", selector_weights, tokens)
        embedding = F.normalize(
            torch.cat(
                [
                    F.normalize(global_embedding, p=2, dim=1),
                    F.normalize(local_embedding, p=2, dim=1),
                ],
                dim=1,
            ),
            p=2,
            dim=1,
        )
        logits = self.classifier(embedding)
        return HeadSelectorOutput(
            logits=logits,
            embedding=embedding,
            global_embedding=global_embedding,
            local_embedding=local_embedding,
            selector_logits=selector_logits,
            selector_weights=selector_weights,
        )

    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        spatial_features = self.backbone.features(images)
        tokens = flatten_swin_spatial_features(spatial_features)
        return self.backbone.norm(tokens)

    def set_fine_tune_mode(self, mode: FineTuneMode) -> None:
        if mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return
        if mode == "frozen":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            return
        raise ValueError(f"Unsupported Swin head-aware fine_tune_mode: {mode}")


def flatten_swin_spatial_features(spatial_features: torch.Tensor) -> torch.Tensor:
    if spatial_features.ndim != 4:
        raise ValueError(f"Expected Swin spatial features with shape [B,H,W,C], got {tuple(spatial_features.shape)}")
    if spatial_features.shape[-1] >= spatial_features.shape[1]:
        batch_size, height, width, channels = spatial_features.shape
        return spatial_features.reshape(batch_size, height * width, channels)
    batch_size, channels, height, width = spatial_features.shape
    return spatial_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


def build_swin_head_aware_classifier(
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    fine_tune_mode: FineTuneMode | None = None,
    selector_temperature: float = 1.0,
) -> SwinHeadAwareClassifier:
    return SwinHeadAwareClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        fine_tune_mode=fine_tune_mode,
        selector_temperature=selector_temperature,
    )
