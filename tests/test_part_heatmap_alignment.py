from __future__ import annotations

import math

import numpy as np

from scripts.evaluate_part_heatmap_alignment import (
    heat_mass_near_points,
    heatmap_peak_xy,
    nearest_distance,
    transform_visible_parts,
)


def test_heatmap_peak_xy_maps_cell_center_to_image_coordinates() -> None:
    heatmap = np.array([[0.0, 0.1], [0.2, 1.0]], dtype=np.float32)

    assert heatmap_peak_xy(heatmap, image_size=224) == (168.0, 168.0)


def test_transform_visible_parts_with_bbox_crop() -> None:
    part_names = {2: "beak"}
    part_locations = {1: {2: (30.0, 20.0, True)}}

    points = transform_visible_parts(
        image_id=1,
        part_names=part_names,
        part_locations=part_locations,
        original_size=(100, 100),
        image_size=224,
        crop_mode="bbox",
        bbox=(10.0, 10.0, 40.0, 40.0),
        bbox_margin=0.0,
    )

    assert points["beak"] == (112.0, 56.0)


def test_nearest_distance_returns_inf_without_candidates() -> None:
    assert math.isinf(nearest_distance((0.0, 0.0), []))


def test_heat_mass_near_points_counts_normalized_heatmap_mass() -> None:
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[1, 1] = 2.0
    heatmap[3, 3] = 2.0

    mass = heat_mass_near_points(
        heatmap=heatmap,
        points=[(84.0, 84.0)],
        image_size=224,
        radius_pixels=32.0,
    )

    assert mass == 0.5
