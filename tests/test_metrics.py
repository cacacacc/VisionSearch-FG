from __future__ import annotations

import torch
import pytest

from visionsearch_fg.engine import accuracy, top_k_accuracy


def test_accuracy_computes_top1_accuracy() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    targets = torch.tensor([1, 2, 2])

    assert accuracy(logits, targets) == pytest.approx(2 / 3)


def test_top_k_accuracy_counts_target_inside_top_k() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.1, 0.7],
            [0.2, 0.3, 0.5],
        ]
    )
    targets = torch.tensor([1, 2, 0])

    assert top_k_accuracy(logits, targets, k=2) == pytest.approx(2 / 3)
