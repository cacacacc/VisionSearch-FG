from __future__ import annotations

import torch

from visionsearch_fg.models import build_swin_tiny_classifier


def test_swin_tiny_classifier_outputs_logits_and_embedding_shapes() -> None:
    model = build_swin_tiny_classifier(num_classes=200, pretrained=False)
    model.eval()

    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(images)

    assert tuple(output.embedding.shape) == (1, 768)
    assert tuple(output.logits.shape) == (1, 200)


def test_swin_tiny_frozen_mode_freezes_backbone_only() -> None:
    model = build_swin_tiny_classifier(
        num_classes=200,
        pretrained=False,
        fine_tune_mode="frozen",
    )

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
