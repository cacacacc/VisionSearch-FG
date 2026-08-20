# Phase 1 Day 2：训练循环

## 目标

实现 ResNet-18 baseline 的最小训练闭环：

```text
batch -> forward -> loss -> backward -> optimizer.step -> validate -> metric
```

这一步的目标不是追求高准确率，而是确认训练机制正确。

## What：是什么

训练循环负责让模型参数根据 loss 更新。验证循环负责在不更新参数的情况下评估模型。

核心函数：

| 函数 | 作用 |
| --- | --- |
| `accuracy` | 计算 top-1 accuracy |
| `top_k_accuracy` | 计算 top-k accuracy，例如 top-5 |
| `train_one_epoch` | 跑一个训练 epoch |
| `validate` | 跑一次验证 |

## Why：为什么需要

如果训练循环不可靠，后续所有实验结果都不可信。例如 loss 没有反向传播、验证时还在更新梯度、accuracy 计算错了，都会让实验分析失效。

## Where：在 pipeline 的位置

```text
CUB Dataset
-> DataLoader
-> ResNet-18
-> CrossEntropyLoss
-> optimizer
-> validation metrics
```

## Tensor Shape

```text
images: [B, 3, 224, 224]
labels: [B]
embedding: [B, 512]
logits: [B, 200]
loss: scalar
```

## How：如何运行

CPU 冒烟训练：

```powershell
python scripts\train_baseline.py --epochs 1 --batch-size 2 --pretrained false --freeze-backbone --max-train-batches 2 --max-val-batches 2 --device cpu
```

这个命令只跑极少量 batch，用来验证完整链路。

## Trade-off：取舍

`max_train_batches` 和 `max_val_batches` 不是正式实验设置，只用于轻量验证。正式实验应该跑完整 train/test split，并记录完整配置。

`pretrained false` 可以避免首次运行时下载权重，但真实 baseline 建议使用 ImageNet pretrained weights，因为 CUB 数据集规模较小，迁移学习更合理。

## Research Value：科研价值

训练循环是后续所有方法比较的共同实验平台。只有 baseline 训练、验证和指标稳定，后续加入 Swin-Tiny、contrastive loss、retrieval metric 时，比较才有意义。

## 本次验证结果

已在真实 CUB 数据上完成 CPU 小批量闭环：

```text
train_samples: 4
val_samples: 4
train_loss: 5.1700
val_loss: 5.2569
```

由于只跑了 4 个训练样本和随机初始化模型，accuracy 为 0 是正常现象。
