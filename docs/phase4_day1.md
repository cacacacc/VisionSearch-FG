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
| ResNet-18 Full FT + HFlip | ImageNet | 20 | 11 | 69.67% | 待补 | 89.33% | 11.28M | 待补 |
| Swin-Tiny Full FT + HFlip | ImageNet | 30 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

## 判断标准

如果 Swin-Tiny 在 Accuracy 和 Macro-F1 上明显超过 ResNet-18，并且训练成本可接受，说明 Transformer-based backbone 对细粒度鸟类特征有更强表达能力。如果 Swin-Tiny 只提升很小但参数和训练时间显著增加，则后续应谨慎选择它作为默认 backbone。如果 Swin-Tiny 不如 ResNet-18，应先检查 LR、batch size、augmentation 和早停，而不是直接否定 Transformer backbone。
