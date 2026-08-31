from __future__ import annotations

import pytest
import torch

from scripts.generate_swin_evidence_token_visualization import token_scores_to_heatmap


def test_token_scores_to_heatmap_reshapes_square_grid() -> None:
    scores = torch.arange(4, dtype=torch.float32)

    heatmap = token_scores_to_heatmap(scores, temperature=1.0)

    assert tuple(heatmap.shape) == (2, 2)
    assert float(heatmap.min()) == 0.0
    assert float(heatmap.max()) == 1.0


def test_token_scores_to_heatmap_rejects_non_square_tokens() -> None:
    scores = torch.arange(6, dtype=torch.float32)

    with pytest.raises(ValueError, match="square token grid"):
        token_scores_to_heatmap(scores, temperature=1.0)


def test_token_scores_to_heatmap_rejects_non_positive_temperature() -> None:
    scores = torch.arange(4, dtype=torch.float32)

    with pytest.raises(ValueError, match="temperature"):
        token_scores_to_heatmap(scores, temperature=0.0)
