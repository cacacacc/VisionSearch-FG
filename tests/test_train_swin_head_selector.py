from __future__ import annotations

from pathlib import Path

import torch

from scripts.train_swin_head_selector import (
    HEAD_PARTS,
    compute_bbox_crop_box,
    load_compatible_checkpoint,
    make_gaussian_heatmap,
    transform_head_points_to_input,
)
from visionsearch_fg.models import build_swin_head_aware_classifier, build_swin_tiny_classifier


def test_make_gaussian_heatmap_peaks_near_point() -> None:
    heatmap = make_gaussian_heatmap(
        points=[(112.0, 112.0)],
        image_size=224,
        heatmap_size=7,
        sigma=1.0,
    )

    assert tuple(heatmap.shape) == (7, 7)
    assert float(heatmap.max()) == 1.0
    assert torch.argmax(heatmap).item() == 24


def test_make_gaussian_heatmap_returns_zero_without_points() -> None:
    heatmap = make_gaussian_heatmap(points=[], image_size=224, heatmap_size=7, sigma=1.0)

    assert float(heatmap.sum()) == 0.0


def test_transform_head_points_to_input_with_bbox_crop() -> None:
    part_names = {1: "beak", 2: "left wing"}
    part_locations = {10: {1: (30.0, 20.0, True), 2: (50.0, 50.0, True)}}

    points = transform_head_points_to_input(
        image_id=10,
        part_names=part_names,
        part_locations=part_locations,
        original_size=(100, 100),
        image_size=224,
        crop_mode="bbox",
        bbox=(10.0, 10.0, 40.0, 40.0),
        bbox_margin=0.0,
    )

    assert "beak" in HEAD_PARTS
    assert points == [(112.0, 56.0)]


def test_compute_bbox_crop_box_applies_margin() -> None:
    box = compute_bbox_crop_box(
        original_size=(100, 100),
        bbox=(20.0, 30.0, 40.0, 20.0),
        margin=0.25,
    )

    assert box == (10.0, 25.0, 70.0, 55.0)


def test_load_compatible_checkpoint_skips_incompatible_classifier(tmp_path: Path) -> None:
    source = build_swin_tiny_classifier(num_classes=200, pretrained=False)
    target = build_swin_head_aware_classifier(num_classes=200, pretrained=False)
    checkpoint_path = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

    report = load_compatible_checkpoint(target, checkpoint_path, torch.device("cpu"))

    assert report["loaded_keys"] > 0
    assert "classifier.weight" in report["skipped_keys"]
    assert "classifier.bias" in report["skipped_keys"]
