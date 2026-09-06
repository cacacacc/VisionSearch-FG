from __future__ import annotations

import torch


def macro_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
) -> float:
    """Compute the unweighted mean of per-class F1 scores.

    When ``num_classes`` is omitted, only classes appearing in ``targets`` are
    averaged. Supplying it evaluates the complete class range and assigns zero
    to classes with no true positives, which is useful for fixed-class reports.
    """
    if predictions.ndim != 1:
        raise ValueError(
            f"predictions must have shape [num_samples], got {tuple(predictions.shape)}"
        )
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape [num_samples], got {tuple(targets.shape)}")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("predictions and targets must have the same number of samples")
    if targets.numel() == 0:
        raise ValueError("targets must contain at least one sample")

    if num_classes is None:
        class_ids = torch.unique(targets)
    else:
        if num_classes < 1:
            raise ValueError("num_classes must be greater than or equal to 1")
        class_ids = torch.arange(num_classes, device=targets.device)

    f1_scores = []
    for class_id in class_ids:
        # Count one-vs-rest outcomes for this class. The macro average later
        # gives each class equal importance regardless of its sample count.
        true_positive = ((predictions == class_id) & (targets == class_id)).sum().float()
        false_positive = ((predictions == class_id) & (targets != class_id)).sum().float()
        false_negative = ((predictions != class_id) & (targets == class_id)).sum().float()
        denominator = (2 * true_positive) + false_positive + false_negative
        if denominator > 0:
            f1_scores.append((2 * true_positive) / denominator)
        else:
            f1_scores.append(torch.tensor(0.0, device=targets.device))

    return float(torch.stack(f1_scores).mean().item())


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Compute the fraction whose target appears among the k largest logits."""
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
    """Compute ordinary top-1 accuracy through the shared Top-K implementation."""
    return top_k_accuracy(logits=logits, targets=targets, k=1)
