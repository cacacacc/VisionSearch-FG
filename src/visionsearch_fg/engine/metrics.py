from __future__ import annotations

import torch


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Compute top-k accuracy for classification logits."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, num_classes], got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape [batch], got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")

    max_k = min(k, logits.shape[1])
    predicted = logits.topk(max_k, dim=1).indices
    correct = predicted.eq(targets.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute ordinary top-1 accuracy."""
    return top_k_accuracy(logits=logits, targets=targets, k=1)
