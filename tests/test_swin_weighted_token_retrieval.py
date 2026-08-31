from __future__ import annotations

import torch

from scripts.evaluate_swin_weighted_token_retrieval import (
    compute_token_class_evidence,
    format_temperature,
    parse_variant_temperature,
    pool_weighted_tokens,
)


def test_compute_token_class_evidence_uses_target_class_weight() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 1.0], [1.0, 3.0]],
        ]
    )
    classifier_weight = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    target_labels = torch.tensor([1, 2])

    scores = compute_token_class_evidence(tokens, classifier_weight, target_labels)

    torch.testing.assert_close(scores, torch.tensor([[0.0, 4.0], [3.0, 4.0]]))


def test_pool_weighted_tokens_matches_softmax_weighted_average() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [3.0, 2.0]]])
    scores = torch.tensor([[0.0, 2.0]])

    pooled = pool_weighted_tokens(tokens=tokens, scores=scores, temperature=1.0)
    weights = torch.softmax(scores, dim=1)
    expected = torch.einsum("bn,bnc->bc", weights, tokens)

    torch.testing.assert_close(pooled, expected)


def test_lower_temperature_makes_pooling_closer_to_top_token() -> None:
    tokens = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])
    scores = torch.tensor([[0.0, 5.0]])

    cold = pool_weighted_tokens(tokens=tokens, scores=scores, temperature=0.1)
    warm = pool_weighted_tokens(tokens=tokens, scores=scores, temperature=5.0)

    assert cold[0, 0] > warm[0, 0]


def test_format_temperature_is_path_friendly() -> None:
    assert format_temperature(0.5) == "0_5"
    assert format_temperature(1.0) == "1"


def test_parse_variant_temperature_reads_path_friendly_float() -> None:
    assert parse_variant_temperature("evidence_weighted_tau0_5") == 0.5
    assert parse_variant_temperature("global_evidence_weighted_tau2_concat_l2") == 2.0
    assert parse_variant_temperature("global") is None
