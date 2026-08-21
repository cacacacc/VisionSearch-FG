from __future__ import annotations

import pytest
import torch

from visionsearch_fg.analysis import build_prediction_records, summarize_prediction_records


def test_build_prediction_records_exports_top_k_predictions() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.8, 0.2],
            [0.7, 0.2, 0.6],
        ]
    )
    labels = torch.tensor([1, 2])
    image_ids = torch.tensor([11, 12])

    records = build_prediction_records(
        logits=logits,
        labels=labels,
        image_ids=image_ids,
        paths=["a.jpg", "b.jpg"],
        true_class_names=["class_b", "class_c"],
        class_names=["class_a", "class_b", "class_c"],
        top_k=2,
    )

    assert records[0]["correct"] is True
    assert records[0]["top_k_labels"] == [1, 2]
    assert records[1]["correct"] is False
    assert records[1]["top_k_correct"] is True
    assert records[1]["pred_class"] == "class_a"


def test_summarize_prediction_records_computes_accuracy() -> None:
    records = [
        {"correct": True, "top_k_correct": True, "true_probability": 0.8, "pred_probability": 0.8},
        {"correct": False, "top_k_correct": True, "true_probability": 0.3, "pred_probability": 0.4},
    ]

    summary = summarize_prediction_records(records)

    assert summary["total_samples"] == 2
    assert summary["num_errors"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["top_k_accuracy"] == 1.0
    assert summary["mean_true_probability"] == pytest.approx(0.55)
