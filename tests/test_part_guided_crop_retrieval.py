from __future__ import annotations

import torch

from scripts.evaluate_part_guided_crop_retrieval import (
    compute_part_crop_box,
    fuse_view_embeddings,
    visible_group_points,
)


def test_visible_group_points_keeps_only_visible_group_parts() -> None:
    part_names = {
        1: "back",
        2: "beak",
        3: "left eye",
        4: "right wing",
    }
    part_locations = {
        10: {
            1: (5.0, 5.0, True),
            2: (10.0, 20.0, True),
            3: (12.0, 18.0, False),
            4: (30.0, 40.0, True),
        }
    }

    points = visible_group_points(
        image_id=10,
        group_name="head",
        part_names=part_names,
        part_locations=part_locations,
    )

    assert points == [(10.0, 20.0)]


def test_compute_part_crop_box_expands_around_points_and_clips_to_image() -> None:
    box = compute_part_crop_box(
        points=[(5.0, 5.0)],
        image_size=(100, 80),
        crop_scale=2.0,
        min_crop_ratio=0.5,
    )

    assert box == (0, 0, 40, 40)


def test_compute_part_crop_box_uses_point_span_when_larger_than_minimum() -> None:
    box = compute_part_crop_box(
        points=[(20.0, 20.0), (80.0, 40.0)],
        image_size=(100, 100),
        crop_scale=1.5,
        min_crop_ratio=0.2,
    )

    assert box == (5, 0, 95, 90)


def test_fuse_view_embeddings_l2_normalizes_concat_output() -> None:
    first = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    second = torch.tensor([[0.0, 2.0], [0.0, 5.0]])

    fused = fuse_view_embeddings([first, second])

    assert fused.shape == (2, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(fused, ord=2, dim=1), torch.ones(2))
