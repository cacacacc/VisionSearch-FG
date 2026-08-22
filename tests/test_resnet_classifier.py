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


def test_resnet18_partial_fine_tuning_only_trains_layer4_and_classifier() -> None:
    model = build_resnet18_classifier(
        num_classes=200,
        pretrained=False,
        fine_tune_mode="partial",
        trainable_backbone_layers=["layer4"],
    )

    assert any(parameter.requires_grad for parameter in model.backbone.layer4.parameters())
    assert all(not parameter.requires_grad for parameter in model.backbone.layer3.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
