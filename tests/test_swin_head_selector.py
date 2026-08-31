from __future__ import annotations

import torch

from visionsearch_fg.models import build_swin_head_aware_classifier


def test_swin_head_aware_classifier_outputs_expected_shapes() -> None:
    model = build_swin_head_aware_classifier(num_classes=200, pretrained=False)
    model.eval()

    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(images)

    assert tuple(output.logits.shape) == (1, 200)
    assert tuple(output.embedding.shape) == (1, 1536)
    assert tuple(output.global_embedding.shape) == (1, 768)
    assert tuple(output.local_embedding.shape) == (1, 768)
    assert tuple(output.selector_logits.shape) == (1, 49)
    assert tuple(output.selector_weights.shape) == (1, 49)
    torch.testing.assert_close(output.selector_weights.sum(dim=1), torch.ones(1))


def test_swin_head_aware_frozen_mode_freezes_backbone_only() -> None:
    model = build_swin_head_aware_classifier(
        num_classes=200,
        pretrained=False,
        fine_tune_mode="frozen",
    )

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.selector.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
