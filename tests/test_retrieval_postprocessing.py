from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_retrieval_postprocessing import (  # noqa: E402
    average_view_embeddings,
    query_expand_embeddings,
    rank_query_to_gallery,
)


def test_average_view_embeddings_l2_normalizes_average() -> None:
    view_a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    view_b = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)

    averaged = average_view_embeddings([view_a, view_b])

    assert averaged.shape == (2, 2)
    assert np.allclose(np.linalg.norm(averaged, axis=1), 1.0)
    assert np.allclose(averaged[0], np.array([0.70710677, 0.70710677], dtype=np.float32))


def test_query_expansion_moves_query_toward_top_neighbors() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ranked_indices = np.array([[1, 2], [0, 2], [1, 0]])

    expanded = query_expand_embeddings(
        embeddings=embeddings,
        ranked_indices=ranked_indices,
        top_k=1,
        alpha=0.5,
    )

    assert expanded.shape == embeddings.shape
    assert np.allclose(np.linalg.norm(expanded, axis=1), 1.0)
    assert expanded[0, 1] > embeddings[0, 1]


def test_rank_query_to_gallery_excludes_self() -> None:
    embeddings = np.eye(3, dtype=np.float32)

    ranked = rank_query_to_gallery(
        query_embeddings=embeddings,
        gallery_embeddings=embeddings,
        exclude_self=True,
    )

    assert ranked.shape == (3, 2)
    assert all(index not in ranked[index] for index in range(3))
