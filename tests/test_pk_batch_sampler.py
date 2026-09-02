from __future__ import annotations

import pytest

from visionsearch_fg.data import PKBatchSampler


def test_pk_batch_sampler_builds_balanced_batches() -> None:
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    sampler = PKBatchSampler(labels=labels, classes_per_batch=2, samples_per_class=2, seed=7)

    batch = next(iter(sampler))
    batch_labels = [labels[index] for index in batch]

    assert len(batch) == 4
    assert len(set(batch_labels)) == 2
    assert all(batch_labels.count(label) == 2 for label in set(batch_labels))


def test_pk_batch_sampler_rejects_too_many_classes_per_batch() -> None:
    labels = [0, 0, 1, 1]

    with pytest.raises(ValueError):
        PKBatchSampler(labels=labels, classes_per_batch=3, samples_per_class=2)


def test_pk_batch_sampler_replaces_when_class_has_too_few_samples() -> None:
    labels = [0, 1]
    sampler = PKBatchSampler(labels=labels, classes_per_batch=2, samples_per_class=2, seed=7)

    batch = next(iter(sampler))
    batch_labels = [labels[index] for index in batch]

    assert len(batch) == 4
    assert batch_labels.count(0) == 2
    assert batch_labels.count(1) == 2
