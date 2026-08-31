from __future__ import annotations

import torch

from scripts.evaluate_swin_local_token_retrieval import (
    flatten_swin_spatial_features,
    fuse_global_local,
    parse_variant_top_k,
    pool_topk_tokens,
)


def test_flatten_swin_spatial_features_accepts_channel_last() -> None:
    features = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)

    tokens = flatten_swin_spatial_features(features)

    assert tuple(tokens.shape) == (2, 6, 4)
    torch.testing.assert_close(tokens[0, 0], features[0, 0, 0])


def test_flatten_swin_spatial_features_accepts_channel_first() -> None:
    features = torch.arange(2 * 4 * 2 * 3).reshape(2, 4, 2, 3)

    tokens = flatten_swin_spatial_features(features)

    assert tuple(tokens.shape) == (2, 6, 4)
    torch.testing.assert_close(tokens[0, 0], features[0, :, 0, 0])


def test_pool_topk_tokens_uses_l2_response() -> None:
    tokens = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 3.0],
                [2.0, 0.0],
            ]
        ]
    )

    pooled = pool_topk_tokens(tokens, k=2)

    torch.testing.assert_close(pooled, torch.tensor([[1.0, 1.5]]))


def test_fuse_global_local_returns_l2_normalized_concat() -> None:
    global_embedding = torch.tensor([[3.0, 4.0]])
    local_embedding = torch.tensor([[0.0, 2.0]])

    fused = fuse_global_local(global_embedding, local_embedding)

    assert tuple(fused.shape) == (1, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(fused, dim=1), torch.ones(1))


def test_parse_variant_top_k() -> None:
    assert parse_variant_top_k("global_local_top5_concat_l2") == 5
    assert parse_variant_top_k("global") is None
