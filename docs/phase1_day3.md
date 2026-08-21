# Phase 1 Day 3：Baseline 实验配置与输出规范

## 目标

把训练脚本从“能跑”升级为“能做实验”。核心要求是：每次实验都有独立配置、独立日志、独立 checkpoint，避免结果互相覆盖。

## What：是什么

一次实验运行现在会生成一个独立 run：

```text
outputs/
├── checkpoints/<experiment_name>/<timestamp>_<experiment_name>/
│   ├── best.pt
│   └── last.pt
└── logs/<experiment_name>/<timestamp>_<experiment_name>/
    ├── config.yaml
    ├── metadata.json
    └── history.json
```

其中：

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | 保存本次运行使用的原始配置 |
| `metadata.json` | 保存运行时覆盖参数、设备、batch 数限制等信息 |
| `history.json` | 保存每个 epoch 的 loss、accuracy、top-5 accuracy |
| `best.pt` | 验证集 top-1 accuracy 最好的模型 |
| `last.pt` | 最后一个 epoch 的模型 |

## Why：为什么需要

如果所有训练都写到同一个文件：

```text
baseline_resnet18_best.pt
baseline_resnet18_history.json
```

后一次运行会覆盖前一次运行。这样无法回答基本科研问题：

```text
这个结果是由哪个配置产生的？
这个 checkpoint 对应哪个 epoch？
是否使用了 pretrained？
是否冻结了 backbone？
```

因此，从 Day 3 开始，实验输出必须带有 experiment name 和 timestamp。

## Where：在 pipeline 的位置

这一步位于训练系统的外围：

```text
config
-> train_baseline.py
-> training loop
-> run directory
-> logs / checkpoints
-> later analysis
```

它不改变模型结构，但决定实验是否可追踪、可比较、可复现。

## How：如何运行

CPU smoke 配置：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_cpu_smoke.yaml --device cpu
```

冻结 backbone 的 baseline 配置：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_frozen.yaml --device cpu
```

如果只想临时限制 batch 数，可以使用命令行覆盖：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_frozen.yaml --device cpu --max-train-batches 20 --max-val-batches 20
```

## Trade-off：取舍

冻结 backbone 的优点是训练快、显存/内存压力低，适合轻薄本；缺点是模型只能学习最后的分类头，无法让底层视觉特征适配 CUB。

不冻结 backbone 的优点是上限更高；缺点是 CPU 训练慢，且更容易在小数据集上过拟合。

## Research Value：科研价值

Day 3 的实验问题是：

```text
只训练 ImageNet 预训练 ResNet-18 的分类头，CUB 上是否有基本可学习信号？
```

这会成为后续 fine-tuning、Swin-Tiny、contrastive learning 和 retrieval 的参照线。

## 本次真实 baseline 结果

运行命令：

```powershell
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_frozen.yaml --device cpu
```

运行信息：

| 项目 | 值 |
| --- | --- |
| experiment | `baseline_resnet18_frozen` |
| run id | `20260821_200139_baseline_resnet18_frozen` |
| device | `cpu` |
| epochs | `3` |
| batch size | `8` |
| pretrained | `true` |
| freeze backbone | `true` |
| train samples | `5994` |
| val samples | `5794` |

指标：

| epoch | train loss | train acc | val loss | val acc | val top-5 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.1927 | 2.17% | 4.7612 | 7.01% | 21.94% |
| 2 | 4.6071 | 11.08% | 4.2478 | 17.88% | 42.70% |
| 3 | 4.1740 | 21.99% | 3.8609 | 24.70% | 53.76% |

checkpoint：

```text
outputs/checkpoints/baseline_resnet18_frozen/20260821_200139_baseline_resnet18_frozen/best.pt
outputs/checkpoints/baseline_resnet18_frozen/20260821_200139_baseline_resnet18_frozen/last.pt
```

结论：冻结 ImageNet 预训练 ResNet-18 backbone 后，仅训练分类头已经能在 CUB 上取得明显高于随机猜测的结果。CUB 有 200 类，随机 top-1 约为 0.5%，本次 epoch 3 的验证 top-1 为 24.70%，说明数据读取、标签映射、训练 loop、验证 loop 和 checkpoint 保存链路都已经可用。
