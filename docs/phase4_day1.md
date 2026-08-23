# Phase 4 Day 1：ResNet-18 vs Swin-Tiny Classification

## 目标

本阶段进入 Transformer-based backbone 对照实验，比较 ResNet-18 与 Swin-Tiny 在 CUB-200-2011 细粒度鸟类分类上的表现。这里不是要求两个模型使用完全相同 learning rate，而是要求数据、split、输入分辨率、augmentation、训练预算和 evaluation protocol 可比；每个模型应使用适合自身结构的合理超参数，并完整报告配置。

## Research Question

```text
Transformer-based backbone 是否能够更有效地学习细粒度鸟类视觉特征？
```

## Controlled Variables

固定项包括 CUB-200-2011、固定 train/validation/test split、224 x 224 输入、HFlip augmentation、validation evaluation protocol、best validation checkpoint 选择方式。ResNet-18 使用 Phase 1 已完成的 Full Fine-tuning + HFlip 作为当前 CNN baseline；Swin-Tiny 使用 ImageNet pretrained backbone、full fine-tuning、AdamW、backbone LR 5e-5、classifier LR 5e-4。

## Metrics

本实验记录 Accuracy、Macro-F1、Top-5 Accuracy、Parameter Count 和 Training Time。Accuracy 衡量整体分类正确率；Macro-F1 对类别更公平，适合 200 类细粒度分类；Parameter Count 和 Training Time 用于判断性能提升是否伴随明显成本增加。

## Swin-Tiny 运行指令

GPU 正式训练指令：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --device cuda
```

如果显存不足，先把配置里的 `batch_size` 从 8 改成 4，或者临时运行：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --batch-size 4 `
  --device cuda
```

如果没有 GPU，但仍然要用 CPU 跑正式训练，使用 CPU formal 配置。该配置仍使用固定 split、224 输入、HFlip、30 epoch 和 early stopping，但 batch size 调整为 2，`num_workers` 调整为 0，所有结果必须明确报告 `device=cpu` 和 `batch_size=2`：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_cpu_formal.yaml `
  --device cpu
```

CPU 正式训练会很慢，这是计算成本的一部分，不影响它作为项目内正式实验；但后续与 GPU 论文结果比较时，不能把 training time 直接混在一起解释。

## 当前对照表

| Model | Pretrained | Max Epoch | Best Epoch | Val Acc | Val Macro-F1 | Val Top-5 | Params | Train Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 Full FT + HFlip | ImageNet | 20 | 11 | 69.67% | 未记录 | 89.33% | 11.28M | 未记录 |
| Swin-Tiny Full FT + HFlip | ImageNet | 30 | 4 | 75.50% | 75.28% | 93.42% | 27.67M | 18.50 min |

Swin-Tiny 结果来自 `outputs/logs/baseline_swin_tiny_protocol/20260823_012929_baseline_swin_tiny_protocol/history.json` 和对应 `best.pt`。该 run 由 RTX 4070 执行，用户确认使用 `configs/baseline_swin_tiny_protocol.yaml` 原配置：batch size 8、ImageNet pretrained、full fine-tuning、backbone LR 5e-5、classifier LR 5e-4、weight decay 1e-4、early stopping patience 5。由于本地 `metadata.json` 曾保留 smoke test 信息，本实验以真实 `history.json`、`best.pt` 内部 checkpoint metadata 和用户确认的训练配置为准。

Swin-Tiny 在 epoch 4 达到最高 validation accuracy 75.50%，之后连续 5 个 epoch 没有超过该指标，因此 early stopping 在 epoch 9 结束。最高 Top-5 accuracy 出现在 epoch 7，为 93.75%；上表为了和主指标 checkpoint 对齐，记录 best accuracy checkpoint 对应的 Top-5 accuracy 93.42%。

## 判断标准

当前结果支持 Swin-Tiny 在分类任务上优于 ResNet-18：Val Acc 提升 5.83 percentage points，Val Top-5 提升 4.09 percentage points，并且 Macro-F1 达到 75.28%。代价是参数量从 11.28M 增加到 27.67M，约为 ResNet-18 的 2.45 倍。下一步不能直接假设 Swin 的 retrieval embedding 也一定更好，必须执行 Phase 4.2 的 backbone retrieval comparison。
