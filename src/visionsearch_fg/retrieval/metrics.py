from __future__ import annotations

from typing import Literal

import numpy as np

RetrievalMetric = Literal["cosine", "euclidean"]


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize every embedding vector independently to unit length."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, embedding_dim]")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return all-pairs cosine similarity after L2 normalization."""
    normalized_embeddings = l2_normalize(embeddings)
    return normalized_embeddings @ normalized_embeddings.T


def dot_product_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, embedding_dim]")
    return embeddings @ embeddings.T


def euclidean_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the all-pairs Euclidean distance matrix.

    The vectorized identity ``||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b`` avoids a
    Python loop over image pairs. Small negative values caused by floating
    point roundoff are clamped before taking the square root.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, embedding_dim]")

    squared_norms = np.sum(embeddings * embeddings, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * (embeddings @ embeddings.T)
    return np.sqrt(np.maximum(squared_distances, 0.0))


def rank_gallery_for_queries(
    similarity: np.ndarray,
    exclude_self: bool = True,
) -> np.ndarray:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")

    scores = similarity.copy()
    if exclude_self:
        np.fill_diagonal(scores, -np.inf)
        ranked_indices = np.argsort(-scores, axis=1)
        query_indices = np.arange(scores.shape[0])[:, None]
        return ranked_indices[ranked_indices != query_indices].reshape(scores.shape[0], -1)

    return np.argsort(-scores, axis=1)


def rank_gallery_by_distance(
    distance: np.ndarray,
    exclude_self: bool = True,
) -> np.ndarray:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")

    scores = distance.copy()
    if exclude_self:
        np.fill_diagonal(scores, np.inf)
        ranked_indices = np.argsort(scores, axis=1)
        query_indices = np.arange(scores.shape[0])[:, None]
        return ranked_indices[ranked_indices != query_indices].reshape(scores.shape[0], -1)

    return np.argsort(scores, axis=1)


def rank_embeddings(
    embeddings: np.ndarray,
    metric: RetrievalMetric,
    exclude_self: bool = True,
) -> np.ndarray:
    """Convert embeddings into one ranked gallery list per query image."""
    if metric == "cosine":
        similarity = cosine_similarity_matrix(embeddings)
        return rank_gallery_for_queries(similarity, exclude_self=exclude_self)
    if metric == "euclidean":
        distance = euclidean_distance_matrix(embeddings)
        return rank_gallery_by_distance(distance, exclude_self=exclude_self)
    raise ValueError(f"Unsupported retrieval metric: {metric}")


def recall_at_k(ranked_indices: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")

    top_k = ranked_indices[:, :k]
    query_labels = labels[:, None]
    hits = labels[top_k] == query_labels
    return float(hits.any(axis=1).mean())


def mean_average_precision(ranked_indices: np.ndarray, labels: np.ndarray) -> float:
    """Average per-query precision at every relevant retrieved position."""
    average_precisions = [
        average_precision(
            ranked_indices=query_ranked_indices,
            labels=labels,
            query_index=query_index,
        )
        for query_index, query_ranked_indices in enumerate(ranked_indices)
    ]
    return float(np.mean(average_precisions))


def average_precision(
    ranked_indices: np.ndarray,
    labels: np.ndarray,
    query_index: int,
) -> float:
    """Compute one query's average precision from its ranked gallery."""
    relevant = labels[ranked_indices] == labels[query_index]
    if not relevant.any():
        return 0.0

    relevant_ranks = np.flatnonzero(relevant) + 1
    precision_at_relevant = np.arange(1, relevant_ranks.size + 1) / relevant_ranks
    return float(precision_at_relevant.mean())


def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    recall_ks: tuple[int, ...],
    metric: RetrievalMetric = "cosine",
) -> dict:
    """Compute Recall@K and mAP for an image-to-image retrieval run.

    The current protocol treats every embedding as both a query and a gallery
    item, then excludes the query itself from its ranked results. Labels define
    relevance: another image is relevant when it belongs to the same class.
    """
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of samples")

    ranked_indices = rank_embeddings(embeddings=embeddings, metric=metric, exclude_self=True)

    metrics = {f"recall@{k}": recall_at_k(ranked_indices, labels, k) for k in recall_ks}
    metrics["mAP"] = mean_average_precision(ranked_indices, labels)
    return metrics
