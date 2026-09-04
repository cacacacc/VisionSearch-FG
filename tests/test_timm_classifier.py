from __future__ import annotations

import sys
import types

import torch
from torch import nn

from visionsearch_fg.models import build_contrastive_classifier, build_timm_classifier


class _FakeTimmBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 16
        self.proj = nn.Linear(3, self.num_features)
        self.blocks = nn.ModuleList([nn.Linear(self.num_features, self.num_features)])
        self.norm = nn.LayerNorm(self.num_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1))
        return self.proj(pooled)


def test_timm_classifier_outputs_logits_and_embedding_shapes(monkeypatch) -> None:
    create_model_calls = []

    def fake_create_model(*args, **kwargs):
        create_model_calls.append((args, kwargs))
        return _FakeTimmBackbone()

    fake_timm = types.SimpleNamespace(
        create_model=fake_create_model
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    model = build_timm_classifier(
        model_name="convnextv2_nano.fake",
        num_classes=200,
        pretrained=False,
        model_kwargs={"img_size": 448},
    )
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(images)

    assert tuple(output.embedding.shape) == (2, 16)
    assert tuple(output.logits.shape) == (2, 200)
    assert create_model_calls[0][1]["img_size"] == 448


def test_timm_frozen_mode_freezes_backbone_only(monkeypatch) -> None:
    fake_timm = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: _FakeTimmBackbone()
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    model = build_timm_classifier(
        model_name="convnextv2_nano.fake",
        num_classes=200,
        pretrained=False,
        fine_tune_mode="frozen",
    )

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_timm_partial_mode_trains_selected_backbone_layers_only(monkeypatch) -> None:
    fake_timm = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: _FakeTimmBackbone()
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    model = build_timm_classifier(
        model_name="vit_small_patch14_dinov2.fake",
        num_classes=200,
        pretrained=False,
        fine_tune_mode="partial",
        trainable_backbone_layers=["blocks.0", "norm"],
    )

    assert all(not parameter.requires_grad for parameter in model.backbone.proj.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.norm.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_timm_contrastive_classifier_outputs_projection(monkeypatch) -> None:
    fake_timm = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: _FakeTimmBackbone()
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    classifier = build_timm_classifier(
        model_name="convnextv2_tiny.fake",
        num_classes=200,
        pretrained=False,
    )
    model = build_contrastive_classifier(
        classifier=classifier,
        projection_dim=8,
        projection_hidden_dim=16,
    )

    images = torch.randn(2, 3, 224, 224)
    output = model(images)

    assert tuple(output.embedding.shape) == (2, 16)
    assert tuple(output.logits.shape) == (2, 200)
    assert tuple(output.projection.shape) == (2, 8)
