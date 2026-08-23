from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss for two or more augmented views per sample."""

    def __init__(self, temperature: float = 0.07, eps: float = 1e-12) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "features must have shape [batch_size, num_views, projection_dim]"
            )
        if labels.ndim != 1:
            raise ValueError("labels must have shape [batch_size]")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same batch size")
        if features.shape[1] < 2:
            raise ValueError("SupConLoss requires at least two views per sample")

        batch_size, num_views, projection_dim = features.shape
        contrast_features = F.normalize(
            features.reshape(batch_size * num_views, projection_dim),
            dim=1,
        )
        labels = labels.reshape(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        positive_mask = positive_mask.repeat_interleave(num_views, dim=0).repeat_interleave(
            num_views,
            dim=1,
        )

        logits = contrast_features @ contrast_features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size * num_views, device=features.device)
        positive_mask = positive_mask * (1.0 - self_mask)
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor_mask = positives_per_anchor > 0
        if not torch.any(valid_anchor_mask):
            raise ValueError("SupConLoss requires at least one positive pair in the batch")

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(
            positives_per_anchor,
            min=1.0,
        )
        return -mean_log_prob_pos[valid_anchor_mask].mean()
