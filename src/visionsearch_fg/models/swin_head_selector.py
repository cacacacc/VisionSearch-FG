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
    """Expose every feature needed by training, retrieval, and selector analysis.

    ``logits`` is used by the classification loss. ``embedding`` is the final
    global-plus-local representation used for retrieval. The remaining fields
    keep the intermediate values visible so scripts can inspect where the
    selector is looking instead of reconstructing them from the model.
    """

    logits: torch.Tensor
    embedding: torch.Tensor
    global_embedding: torch.Tensor
    local_embedding: torch.Tensor
    selector_logits: torch.Tensor
    selector_weights: torch.Tensor


class SwinHeadAwareClassifier(nn.Module):
    """Swin-Tiny with a learned soft local selector over final-stage tokens.

    The normal Swin classifier reduces its final spatial feature map to one
    vector by global average pooling. This model keeps the spatial tokens and
    learns one scalar score for each token. A softmax turns those scores into
    weights, so the local feature is a differentiable weighted average rather
    than a hard top-k selection. This preserves gradient flow through both the
    selector and the backbone.
    """

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

        # Remove Swin's original ImageNet classification head. The backbone
        # will provide the final token features, while this module supplies a
        # new classifier that consumes the fused global/local representation.
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        backbone = swin_t(weights=weights)
        embedding_dim = backbone.head.in_features
        backbone.head = nn.Identity()

        self.backbone = backbone
        # Each token receives one scalar evidence score. The softmax in
        # forward() converts these scores into a probability distribution over
        # spatial locations for every image independently.
        self.selector = nn.Linear(embedding_dim, 1)
        # Global and local features are concatenated, therefore the classifier
        # input dimension is twice the Swin token dimension (768 -> 1536 for
        # torchvision's Swin-Tiny).
        self.classifier = nn.Linear(embedding_dim * 2, num_classes)
        self.embedding_dim = embedding_dim * 2
        self.token_dim = embedding_dim
        self.num_classes = num_classes
        self.selector_temperature = selector_temperature

        mode = fine_tune_mode or ("frozen" if freeze_backbone else "full")
        self.set_fine_tune_mode(mode)

    def forward(self, images: torch.Tensor) -> HeadSelectorOutput:
        """Classify images and return both retrieval features and evidence maps.

        For a standard 224x224 input, ``tokens`` has shape ``[B, 49, 768]``:
        49 is the 7x7 final spatial grid and 768 is the channel dimension.
        The selector operates along the 49-token axis, not across channels.
        """

        tokens = self.extract_tokens(images)

        # Averaging all final-stage tokens gives the conventional global Swin
        # representation and keeps the whole image in the feature.
        global_embedding = tokens.mean(dim=1)

        # Produce one unnormalized evidence score per token. Squeezing only
        # the last dimension changes [B, N, 1] into [B, N]. Dividing by the
        # temperature controls concentration: lower values make attention
        # more peaked, while higher values make it more uniform.
        selector_logits = self.selector(tokens).squeeze(-1)
        selector_weights = torch.softmax(selector_logits / self.selector_temperature, dim=1)

        # Weighted pooling creates a differentiable local representation. The
        # einsum notation means: for each batch item, sum token features after
        # multiplying each token by its scalar selector weight.
        local_embedding = torch.einsum("bn,bnc->bc", selector_weights, tokens)

        # Normalize the two branches separately so the local branch cannot
        # dominate merely because of feature magnitude. Normalize once more
        # after concatenation because the result is the representation exposed
        # to retrieval and contrastive objectives.
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
        """Return normalized final-stage Swin tokens as ``[B, N, C]``.

        ``torchvision`` exposes the Swin feature extractor as ``features``.
        Depending on the torchvision implementation, its output may be either
        channels-last ``[B, H, W, C]`` or channels-first ``[B, C, H, W]``.
        The helper below canonicalizes both layouts before Swin's final norm.
        """

        spatial_features = self.backbone.features(images)
        tokens = flatten_swin_spatial_features(spatial_features)
        return self.backbone.norm(tokens)

    def set_fine_tune_mode(self, mode: FineTuneMode) -> None:
        """Control whether gradients update the Swin backbone.

        The selector and classifier are intentionally untouched here. Thus a
        frozen run still trains the newly added heads while using the pretrained
        Swin features as a fixed encoder.
        """

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
    """Flatten a Swin feature map into ``[batch, tokens, channels]``.

    The project needs one canonical layout for the selector. The shape check
    prevents silently flattening an unexpected tensor, while the layout branch
    supports both feature conventions used by different torchvision versions.
    For a 7x7 feature map, ``H * W`` becomes 49 tokens.
    """

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
