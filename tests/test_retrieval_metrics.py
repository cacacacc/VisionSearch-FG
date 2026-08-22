from __future__ import annotations

import numpy as np
import pytest

from visionsearch_fg.retrieval import (
    dot_product_similarity_matrix,
    euclidean_distance_matrix,
    evaluate_retrieval,
    l2_normalize,
    rank_gallery_by_distance,
    rank_gallery_for_queries,
    recall_at_k,
)


def test_l2_normalize_returns_unit_vectors() -> None:
    embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])

    normalized = l2_normalize(embeddings)

    assert np.allclose(np.linalg.norm(normalized, axis=1), np.array([1.0, 1.0]))


def test_dot_product_similarity_matrix_keeps_magnitude_effect() -> None:
    embeddings = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])

    similarity = dot_product_similarity_matrix(embeddings)

    assert similarity[0, 1] == pytest.approx(2.0)
    assert similarity[1, 1] == pytest.approx(4.0)


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


def test_rank_gallery_by_distance_excludes_query_itself() -> None:
    distance = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.8],
            [0.9, 0.8, 0.0],
        ]
    )

    ranked_indices = rank_gallery_by_distance(distance, exclude_self=True)

    assert list(ranked_indices[:, 0]) == [1, 0, 1]


def test_euclidean_distance_matrix_reports_pairwise_distances() -> None:
    embeddings = np.array([[0.0, 0.0], [3.0, 4.0]])

    distance = euclidean_distance_matrix(embeddings)

    assert distance[0, 1] == pytest.approx(5.0)
    assert distance[1, 0] == pytest.approx(5.0)


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


def test_evaluate_retrieval_supports_euclidean_metric() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    metrics = evaluate_retrieval(
        embeddings=embeddings,
        labels=labels,
        recall_ks=(1, 2),
        metric="euclidean",
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert metrics["mAP"] == pytest.approx(1.0)


def test_evaluate_retrieval_rejects_unknown_metric() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="Unsupported retrieval metric"):
        evaluate_retrieval(
            embeddings=embeddings,
            labels=labels,
            recall_ks=(1,),
            metric="manhattan",  # type: ignore[arg-type]
        )
