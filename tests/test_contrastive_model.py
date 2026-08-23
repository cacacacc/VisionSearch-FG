from __future__ import annotations

import torch

from visionsearch_fg.models import build_contrastive_classifier, build_resnet18_classifier


def test_contrastive_classifier_outputs_projection() -> None:
    classifier = build_resnet18_classifier(num_classes=200, pretrained=False)
    model = build_contrastive_classifier(classifier=classifier, projection_dim=64)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(images)

    assert tuple(output.embedding.shape) == (2, 512)
    assert tuple(output.logits.shape) == (2, 200)
    assert tuple(output.projection.shape) == (2, 64)
