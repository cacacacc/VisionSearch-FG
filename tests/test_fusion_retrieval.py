from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_fusion_retrieval import ensure_same_records, fuse_embeddings


def test_fuse_embeddings_concat_l2_normalizes_each_output_row() -> None:
    primary = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    secondary = np.array([[0.0, 2.0], [0.0, 5.0]], dtype=np.float32)

    fused = fuse_embeddings(primary, secondary, fusion="concat_l2")

    assert fused.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), np.ones(2), atol=1e-6)


def test_fuse_embeddings_average_l2_requires_matching_dims() -> None:
    primary = np.ones((2, 3), dtype=np.float32)
    secondary = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="same dimension"):
        fuse_embeddings(primary, secondary, fusion="average_l2")


def test_ensure_same_records_rejects_mismatched_image_ids() -> None:
    primary = [{"image_id": 1, "label": 0, "class_name": "a"}]
    secondary = [{"image_id": 2, "label": 0, "class_name": "a"}]

    with pytest.raises(ValueError, match="Record mismatch"):
        ensure_same_records(primary, secondary)
