from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from visionsearch_fg.engine.metrics import accuracy, macro_f1_score, top_k_accuracy


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
    macro_f1: float
    num_samples: int


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> TrainStats:
    """执行一个 epoch 的优化，并返回按样本数加权的统计量。

    loss 和 accuracy 会先乘以 batch size 再累计，因此即使最后一个 batch 较小，
    统计结果也仍然正确。``max_batches`` 只用于 CPU smoke test 和快速调试，
    不应被用于悄悄改变正式实验协议。
    """
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress = tqdm(_limit_batches(dataloader, max_batches), desc="train", leave=False)
    for batch in progress:
        # 数据集返回字典以保留元数据，但优化步骤只使用图像张量和标签。
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
    """在无梯度模式下评估模型，并计算分类诊断指标。

    prediction 和 label 会收集到 CPU 上，从而在完整验证集上计算 macro-F1，
    而不是对每个 batch 的 F1 做平均。
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_top5_correct = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress = tqdm(_limit_batches(dataloader, max_batches), desc="valid", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)
        loss = criterion(output.logits, labels)

        batch_size = labels.shape[0]
        batch_accuracy = accuracy(output.logits, labels)
        batch_top5_accuracy = top_k_accuracy(output.logits, labels, k=5)
        # 保留类别预测，用于后续完整验证集级别的 macro-F1 计算。
        predictions = output.logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += batch_accuracy * batch_size
        total_top5_correct += batch_top5_accuracy * batch_size
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
            top5=f"{total_top5_correct / total_samples:.4f}",
        )

    return EvalStats(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        top5_accuracy=total_top5_correct / total_samples,
        macro_f1=macro_f1_score(
            predictions=torch.cat(all_predictions),
            targets=torch.cat(all_labels),
        ),
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
