from __future__ import annotations

import numpy as np
import pytest

from visionsearch_fg.retrieval import (
    evaluate_retrieval,
    l2_normalize,
    rank_gallery_for_queries,
    recall_at_k,
)


def test_l2_normalize_returns_unit_vectors() -> None:
    embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])

    normalized = l2_normalize(embeddings)

    assert np.allclose(np.linalg.norm(normalized, axis=1), np.array([1.0, 1.0]))


def test_recall_at_k_counts_same_class_neighbors() -> None:
    labels = np.array([0, 0, 1, 1])
    ranked_indices = np.array(
        [
            [1, 2, 3],
            [0, 2, 3],
            [3, 0, 1],
            [2, 0, 1],
        ]
    )

    assert recall_at_k(ranked_indices, labels, k=1) == 1.0


def test_rank_gallery_for_queries_excludes_query_itself() -> None:
    similarity = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )

    ranked_indices = rank_gallery_for_queries(similarity, exclude_self=True)

    assert list(ranked_indices[:, 0]) == [1, 0, 1]


def test_evaluate_retrieval_reports_recall_and_map() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    metrics = evaluate_retrieval(embeddings=embeddings, labels=labels, recall_ks=(1, 2))

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert metrics["mAP"] == pytest.approx(1.0)
