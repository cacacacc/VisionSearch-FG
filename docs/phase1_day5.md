# Phase 1 Day 5：非冻结 backbone 的小规模 fine-tuning Probe

## 目标

Day 5 的目标不是追求正式精度，而是验证“解冻 ResNet-18 backbone 后，完整训练链路是否仍然可靠”。

冻结 backbone 时，训练只更新最后的分类头；非冻结 backbone 时，反向传播会穿过整个 ResNet-18。它能让 ImageNet 预训练特征适配 CUB，但代价是 CPU 训练更慢、显存或内存压力更高，也更容易在小数据集上过拟合。

## What：是什么

新增配置：

```text
configs/baseline_resnet18_unfrozen_cpu_probe.yaml
```

关键设置：

| 项目 | 值 |
| --- | --- |
| backbone | `resnet18` |
| pretrained | `true` |
| freeze backbone | `false` |
| epochs | `1` |
| batch size | `4` |
| learning rate | `0.00001` |
| max train batches | `20` |
| max val batches | `40` |

这是一份 CPU probe 配置，只用于确认链路和观察初始信号，不用于写正式实验结论。

## Why：为什么学习这个

迁移学习通常有两个阶段：

```text
阶段 1：冻结 backbone，只训练分类头
阶段 2：解冻部分或全部 backbone，用更小学习率 fine-tune
```

原因是 backbone 已经从 ImageNet 学到通用视觉特征。如果一开始就用大学习率训练全部参数，容易破坏预训练特征；如果永远冻结 backbone，模型又无法适配 CUB 的细粒度鸟类差异。

## Where：在 pipeline 的位置

```text
CUB images
-> ResNet-18 backbone
-> embedding [B, 512]
-> classifier [B, 200]
-> loss
-> backward
```

冻结时：

```text
backbone: requires_grad = false
classifier: requires_grad = true
```

非冻结时：

```text
backbone: requires_grad = true
classifier: requires_grad = true
```

## 如何运行

```powershell
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_unfrozen_cpu_probe.yaml --device cpu
```

输出目录仍然遵守 Day 3 的规范：

```text
outputs/checkpoints/baseline_resnet18_unfrozen_cpu_probe/<run_id>/
outputs/logs/baseline_resnet18_unfrozen_cpu_probe/<run_id>/
```

## 本阶段判断标准

完成 Day 5 后，至少要确认：

- 训练脚本能正常加载 ImageNet 预训练 ResNet-18。
- `freeze_backbone=false` 时，可训练参数数量显著增加。
- CPU 上能完成一轮小规模训练和验证。
- `history.json`、`metadata.json`、`best.pt`、`last.pt` 正常保存。

## 科研注意

这个 probe 的验证集只取前 40 个 batch，因此指标可能受类别顺序影响，不能和完整验证集 baseline 直接比较。它的价值是工程验证。正式对照需要使用完整验证集，并最好在 GPU 上进行多 epoch 训练。

## 本次 CPU Probe 结果

运行命令：

```powershell
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_unfrozen_cpu_probe.yaml --device cpu
```

运行信息：

| 项目 | 值 |
| --- | --- |
| experiment | `baseline_resnet18_unfrozen_cpu_probe` |
| run id | `20260821_205638_baseline_resnet18_unfrozen_cpu_probe` |
| device | `cpu` |
| epochs | `1` |
| batch size | `4` |
| pretrained | `true` |
| freeze backbone | `false` |
| total parameters | `11279112` |
| trainable parameters | `11279112` |
| train samples | `80` |
| val samples | `160` |

参数量对比：

| 设置 | total parameters | trainable parameters |
| --- | ---: | ---: |
| frozen backbone | 11,279,112 | 102,600 |
| unfrozen backbone | 11,279,112 | 11,279,112 |

指标：

| epoch | train loss | train acc | val loss | val acc | val top-5 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.4298 | 0.00% | 5.4369 | 5.00% | 10.62% |

checkpoint：

```text
outputs/checkpoints/baseline_resnet18_unfrozen_cpu_probe/20260821_205638_baseline_resnet18_unfrozen_cpu_probe/best.pt
outputs/checkpoints/baseline_resnet18_unfrozen_cpu_probe/20260821_205638_baseline_resnet18_unfrozen_cpu_probe/last.pt
```

结论：非冻结 backbone 的训练链路已经跑通，且 metadata 能正确记录可训练参数数量。因为本次只训练 80 张图、验证 160 张图，所以这个分数只能证明工程链路可用，不能和 Day 3 的完整验证集结果做正式精度对比。
