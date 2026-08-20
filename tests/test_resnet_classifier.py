from __future__ import annotations

import torch

from visionsearch_fg.models import build_resnet18_classifier


def test_resnet18_classifier_outputs_logits_and_embedding_shapes() -> None:
    model = build_resnet18_classifier(num_classes=200, pretrained=False)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(images)

    assert tuple(output.embedding.shape) == (2, 512)
    assert tuple(output.logits.shape) == (2, 200)
