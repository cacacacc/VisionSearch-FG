from __future__ import annotations

import numpy as np


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, embedding_dim]")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized_embeddings = l2_normalize(embeddings)
    return normalized_embeddings @ normalized_embeddings.T


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


def recall_at_k(ranked_indices: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")

    top_k = ranked_indices[:, :k]
    query_labels = labels[:, None]
    hits = labels[top_k] == query_labels
    return float(hits.any(axis=1).mean())


def mean_average_precision(ranked_indices: np.ndarray, labels: np.ndarray) -> float:
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
) -> dict:
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of samples")

    similarity = cosine_similarity_matrix(embeddings)
    ranked_indices = rank_gallery_for_queries(similarity, exclude_self=True)

    metrics = {f"recall@{k}": recall_at_k(ranked_indices, labels, k) for k in recall_ks}
    metrics["mAP"] = mean_average_precision(ranked_indices, labels)
    return metrics
