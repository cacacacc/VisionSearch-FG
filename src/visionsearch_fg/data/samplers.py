from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """Build batches with P classes and K samples per class."""

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        samples_per_class: int,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if classes_per_batch < 1:
            raise ValueError("classes_per_batch must be greater than or equal to 1")
        if samples_per_class < 1:
            raise ValueError("samples_per_class must be greater than or equal to 1")
        if not labels:
            raise ValueError("PKBatchSampler requires at least one label")

        self.labels = [int(label) for label in labels]
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.drop_last = drop_last
        self.batch_size = classes_per_batch * samples_per_class
        self.class_to_indices = self._group_indices_by_class(self.labels)
        self.epoch = 0

        if len(self.class_to_indices) < classes_per_batch:
            raise ValueError(
                "classes_per_batch cannot exceed the number of classes available "
                f"({classes_per_batch} > {len(self.class_to_indices)})"
            )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        classes = sorted(self.class_to_indices)
        class_cursors = {label: 0 for label in classes}
        class_queues = {
            label: self._shuffled_indices(indices=indices, rng=rng)
            for label, indices in self.class_to_indices.items()
        }

        for _ in range(len(self)):
            selected_classes = rng.sample(classes, self.classes_per_batch)
            batch: list[int] = []
            for label in selected_classes:
                batch.extend(
                    self._take_k(
                        label=label,
                        class_queues=class_queues,
                        class_cursors=class_cursors,
                        rng=rng,
                    )
                )
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return max(1, len(self.labels) // self.batch_size)
        return math.ceil(len(self.labels) / self.batch_size)

    def _take_k(
        self,
        label: int,
        class_queues: dict[int, list[int]],
        class_cursors: dict[int, int],
        rng: random.Random,
    ) -> list[int]:
        indices = class_queues[label]
        cursor = class_cursors[label]

        if len(indices) < self.samples_per_class:
            return [rng.choice(indices) for _ in range(self.samples_per_class)]

        if cursor + self.samples_per_class > len(indices):
            indices = self._shuffled_indices(indices=self.class_to_indices[label], rng=rng)
            class_queues[label] = indices
            cursor = 0

        selected = indices[cursor : cursor + self.samples_per_class]
        class_cursors[label] = cursor + self.samples_per_class
        return selected

    @staticmethod
    def _group_indices_by_class(labels: Sequence[int]) -> dict[int, list[int]]:
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            class_to_indices[int(label)].append(index)
        return dict(class_to_indices)

    @staticmethod
    def _shuffled_indices(indices: Sequence[int], rng: random.Random) -> list[int]:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        return shuffled
