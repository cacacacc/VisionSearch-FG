from __future__ import annotations

import torch

from scripts.evaluate_swin_evidence_crop_retrieval import (
    crop_area_ratio,
    evidence_scores_to_crop_box,
    expand_box_to_min_size,
    fuse_embeddings,
    select_heatmap_cells,
    token_scores_to_heatmap,
)


def test_token_scores_to_heatmap_normalizes_square_token_grid() -> None:
    scores = torch.arange(4, dtype=torch.float32)

    heatmap = token_scores_to_heatmap(scores, temperature=1.0)

    assert tuple(heatmap.shape) == (2, 2)
    assert float(heatmap.min()) == 0.0
    assert float(heatmap.max()) == 1.0


def test_select_heatmap_cells_uses_relative_threshold() -> None:
    heatmap = torch.tensor([[0.1, 1.0], [0.7, 0.2]])

    selected_y, selected_x = select_heatmap_cells(heatmap, threshold_ratio=0.6)

    assert selected_y.tolist() == [0, 1]
    assert selected_x.tolist() == [1, 0]


def test_expand_box_to_min_size_clips_to_image_bounds() -> None:
    expanded = expand_box_to_min_size(
        box=(0.0, 0.0, 20.0, 20.0),
        image_size=100,
        min_side=50.0,
    )

    assert expanded == (0, 0, 50.0, 50.0)


def test_evidence_scores_to_crop_box_returns_valid_image_region() -> None:
    scores = torch.zeros(49, dtype=torch.float32)
    scores[0] = 10.0

    crop_box = evidence_scores_to_crop_box(
        token_scores=scores,
        image_size=224,
        temperature=1.0,
        threshold_ratio=0.9,
        min_crop_ratio=0.35,
        padding_ratio=0.0,
    )

    left, top, right, bottom = crop_box
    assert left == 0
    assert top == 0
    assert right > left
    assert bottom > top


def test_crop_area_ratio_uses_image_area() -> None:
    assert crop_area_ratio((0, 0, 112, 112), image_size=224) == 0.25


def test_fuse_embeddings_l2_normalizes_output() -> None:
    first = torch.tensor([[3.0, 4.0]])
    second = torch.tensor([[0.0, 2.0]])

    fused = fuse_embeddings([first, second])

    assert fused.shape == (1, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(fused, ord=2, dim=1), torch.ones(1))
