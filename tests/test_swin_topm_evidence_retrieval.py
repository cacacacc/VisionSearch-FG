from __future__ import annotations

import pytest
import torch

from scripts.evaluate_swin_topm_evidence_retrieval import (
    compute_topm_token_weights,
    fuse_embeddings,
    parse_topm_variant,
    pool_topm_class_evidence_tokens,
)


def test_compute_topm_token_weights_sum_to_one_per_sample() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.1, 0.2, 4.0]])
    classifier_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    weights = compute_topm_token_weights(
        tokens=tokens,
        logits=logits,
        classifier_weight=classifier_weight,
        top_m=2,
        temperature=1.0,
    )

    assert tuple(weights.shape) == (2, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))


def test_topm_one_matches_single_predicted_class_evidence() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    logits = torch.tensor([[0.1, 4.0]])
    classifier_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    weights = compute_topm_token_weights(
        tokens=tokens,
        logits=logits,
        classifier_weight=classifier_weight,
        top_m=1,
        temperature=1.0,
    )
    expected = torch.softmax(torch.tensor([[0.0, 2.0]]), dim=1)

    torch.testing.assert_close(weights, expected)


def test_pool_topm_class_evidence_tokens_returns_weighted_average() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [3.0, 2.0]]])
    logits = torch.tensor([[2.0, 0.0]])
    classifier_weight = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    pooled = pool_topm_class_evidence_tokens(
        tokens=tokens,
        logits=logits,
        classifier_weight=classifier_weight,
        top_m=1,
        temperature=1.0,
    )
    weights = torch.softmax(torch.tensor([[0.0, 2.0]]), dim=1)
    expected = torch.einsum("bn,bnc->bc", weights, tokens)

    torch.testing.assert_close(pooled, expected)


def test_compute_topm_token_weights_rejects_invalid_top_m() -> None:
    tokens = torch.zeros(1, 2, 3)
    logits = torch.zeros(1, 4)
    classifier_weight = torch.zeros(4, 3)

    with pytest.raises(ValueError, match="top_m"):
        compute_topm_token_weights(
            tokens=tokens,
            logits=logits,
            classifier_weight=classifier_weight,
            top_m=5,
            temperature=1.0,
        )


def test_fuse_embeddings_l2_normalizes_output() -> None:
    first = torch.tensor([[3.0, 4.0]])
    second = torch.tensor([[0.0, 2.0]])

    fused = fuse_embeddings([first, second])

    assert fused.shape == (1, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(fused, ord=2, dim=1), torch.ones(1))


def test_parse_topm_variant_reads_variant_metadata() -> None:
    assert parse_topm_variant("global") == (None, None)
    assert parse_topm_variant("topm3_evidence_tau0_5") == (3, 0.5)
    assert parse_topm_variant("global_topm10_evidence_tau1_concat_l2") == (10, 1.0)
