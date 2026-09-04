from __future__ import annotations

import torch
from torch import nn

from visionsearch_fg.models import AngularMarginClassifier, AngularMarginHead


def test_angular_margin_head_returns_scaled_cosine_logits() -> None:
    head = AngularMarginHead(embedding_dim=4, num_classes=3, margin_type="arcface", scale=16.0)
    embeddings = torch.randn(2, 4)

    logits = head(embeddings)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_arcface_margin_changes_target_logit_only() -> None:
    head = AngularMarginHead(embedding_dim=4, num_classes=3, margin_type="arcface", scale=1.0)
    embeddings = torch.randn(2, 4)
    labels = torch.tensor([0, 2])

    eval_logits = head(embeddings)
    train_logits = head(embeddings, labels=labels)

    assert train_logits[0, 0] < eval_logits[0, 0]
    assert train_logits[1, 2] < eval_logits[1, 2]
    assert torch.allclose(train_logits[0, 1:], eval_logits[0, 1:])


def test_angular_margin_classifier_exposes_embedding_and_projection() -> None:
    base = _TinyClassifier()
    model = AngularMarginClassifier(base, projection_dim=5)

    output = model(torch.randn(2, 3, 8, 8), labels=torch.tensor([0, 1]))

    assert output.logits.shape == (2, 3)
    assert output.embedding.shape == (2, 4)
    assert output.projection is not None
    assert output.projection.shape == (2, 5)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
        self.embedding_dim = 4
        self.num_classes = 3
