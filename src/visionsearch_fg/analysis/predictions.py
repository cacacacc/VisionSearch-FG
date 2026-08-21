from __future__ import annotations

from typing import Any

import torch


def build_prediction_records(
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_ids: torch.Tensor,
    paths: list[str],
    true_class_names: list[str],
    class_names: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Convert model outputs into JSON-serializable prediction records."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch_size]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same batch size")
    if len(paths) != labels.shape[0] or len(true_class_names) != labels.shape[0]:
        raise ValueError("paths and true_class_names must match batch size")

    k = min(top_k, logits.shape[1])
    probabilities = logits.softmax(dim=1)
    top_probabilities, top_indices = probabilities.topk(k=k, dim=1)

    records: list[dict[str, Any]] = []
    for row_index in range(labels.shape[0]):
        true_label = int(labels[row_index].item())
        top_labels = [int(label) for label in top_indices[row_index].tolist()]
        top_probs = [float(probability) for probability in top_probabilities[row_index].tolist()]
        pred_label = top_labels[0]

        records.append(
            {
                "image_id": int(image_ids[row_index].item()),
                "path": paths[row_index],
                "true_label": true_label,
                "true_class": true_class_names[row_index],
                "pred_label": pred_label,
                "pred_class": _class_name(class_names, pred_label),
                "correct": pred_label == true_label,
                "top_k_correct": true_label in top_labels,
                "true_probability": float(probabilities[row_index, true_label].item()),
                "pred_probability": top_probs[0],
                "top_k_labels": top_labels,
                "top_k_classes": [_class_name(class_names, label) for label in top_labels],
                "top_k_probabilities": top_probs,
            }
        )

    return records


def summarize_prediction_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize per-image prediction records into aggregate metrics."""
    if not records:
        return {
            "total_samples": 0,
            "num_correct": 0,
            "num_top_k_correct": 0,
            "num_errors": 0,
            "accuracy": 0.0,
            "top_k_accuracy": 0.0,
            "mean_true_probability": 0.0,
            "mean_pred_probability": 0.0,
        }

    total_samples = len(records)
    num_correct = sum(1 for record in records if record["correct"])
    num_top_k_correct = sum(1 for record in records if record["top_k_correct"])
    mean_true_probability = (
        sum(float(record["true_probability"]) for record in records) / total_samples
    )
    mean_pred_probability = (
        sum(float(record["pred_probability"]) for record in records) / total_samples
    )

    return {
        "total_samples": total_samples,
        "num_correct": num_correct,
        "num_top_k_correct": num_top_k_correct,
        "num_errors": total_samples - num_correct,
        "accuracy": num_correct / total_samples,
        "top_k_accuracy": num_top_k_correct / total_samples,
        "mean_true_probability": mean_true_probability,
        "mean_pred_probability": mean_pred_probability,
    }


def _class_name(class_names: list[str], label: int) -> str:
    if 0 <= label < len(class_names):
        return class_names[label]
    return f"class_{label}"
