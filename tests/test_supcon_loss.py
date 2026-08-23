from __future__ import annotations

import pytest
import torch

from visionsearch_fg.losses import SupConLoss


def test_supcon_loss_runs_and_backpropagates() -> None:
    features = torch.randn(4, 2, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    criterion = SupConLoss(temperature=0.07)

    loss = criterion(features, labels)
    loss.backward()

    assert loss.item() > 0
    assert features.grad is not None


def test_supcon_loss_handles_unique_labels_through_two_views() -> None:
    features = torch.randn(3, 2, 8)
    labels = torch.tensor([0, 1, 2])
    criterion = SupConLoss()

    loss = criterion(features, labels)

    assert loss.item() > 0


def test_supcon_loss_requires_two_views() -> None:
    features = torch.randn(4, 1, 16)
    labels = torch.tensor([0, 0, 1, 1])
    criterion = SupConLoss()

    with pytest.raises(ValueError):
        criterion(features, labels)
