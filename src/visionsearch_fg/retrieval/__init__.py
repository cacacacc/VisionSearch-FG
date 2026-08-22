"""Embedding indexing and image retrieval metrics."""

from visionsearch_fg.retrieval.metrics import (
    RetrievalMetric,
    average_precision,
    cosine_similarity_matrix,
    dot_product_similarity_matrix,
    euclidean_distance_matrix,
    evaluate_retrieval,
    l2_normalize,
    mean_average_precision,
    rank_embeddings,
    rank_gallery_by_distance,
    rank_gallery_for_queries,
    recall_at_k,
)

__all__ = [
    "RetrievalMetric",
    "average_precision",
    "cosine_similarity_matrix",
    "dot_product_similarity_matrix",
    "euclidean_distance_matrix",
    "evaluate_retrieval",
    "l2_normalize",
    "mean_average_precision",
    "rank_embeddings",
    "rank_gallery_by_distance",
    "rank_gallery_for_queries",
    "recall_at_k",
]
