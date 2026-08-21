from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from visionsearch_fg.engine.metrics import accuracy, top_k_accuracy


@dataclass(frozen=True)
class TrainStats:
    loss: float
    accuracy: float
    num_samples: int


@dataclass(frozen=True)
class EvalStats:
    loss: float
    accuracy: float
    top5_accuracy: float
    num_samples: int


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> TrainStats:
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress = tqdm(_limit_batches(dataloader, max_batches), desc="train", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        batch_accuracy = accuracy(output.logits.detach(), labels)

        total_loss += loss.item() * batch_size
        total_correct += batch_accuracy * batch_size
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return TrainStats(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        num_samples=total_samples,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: Iterable[dict],
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EvalStats:
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_top5_correct = 0.0
    total_samples = 0

    progress = tqdm(_limit_batches(dataloader, max_batches), desc="valid", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)
        loss = criterion(output.logits, labels)

        batch_size = labels.shape[0]
        batch_accuracy = accuracy(output.logits, labels)
        batch_top5_accuracy = top_k_accuracy(output.logits, labels, k=5)

        total_loss += loss.item() * batch_size
        total_correct += batch_accuracy * batch_size
        total_top5_correct += batch_top5_accuracy * batch_size
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
            top5=f"{total_top5_correct / total_samples:.4f}",
        )

    return EvalStats(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        top5_accuracy=total_top5_correct / total_samples,
        num_samples=total_samples,
    )


def _limit_batches(dataloader: Iterable[dict], max_batches: int | None) -> Iterable[dict]:
    if max_batches is None:
        yield from dataloader
        return

    if max_batches < 1:
        raise ValueError("max_batches must be greater than or equal to 1")

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        yield batch
