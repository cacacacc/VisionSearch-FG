# ResNet-18 vs Swin-Tiny Classification

## Motivation

本实验比较 CNN backbone 与 Transformer-based backbone 在 CUB-200-2011 细粒度鸟类分类上的表现。核心问题是 Swin-Tiny 是否能比 ResNet-18 更有效地建模局部纹理、部件关系和长程视觉上下文。

## Protocol

两个模型都使用固定 train/validation split、224 x 224 输入、HFlip augmentation、ImageNet pretrained initialization、CrossEntropy loss 和 best validation checkpoint。ResNet-18 对照使用 Phase 1 已完成的 Full Fine-tuning + HFlip 结果；Swin-Tiny 使用 `configs/baseline_swin_tiny_protocol.yaml`，max epoch 为 30，early stopping patience 为 5。学习率不强行相同：ResNet 与 Swin 使用各自合理的 AdamW 配置。

## Swin-Tiny Command

GPU command：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --device cuda
```

CPU-only formal command：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_cpu_formal.yaml `
  --device cpu
```

CPU-only formal run 必须在结果表中记录 `device=cpu`、`batch_size=2` 和完整 training time。该结果可以作为本项目正式实验，但它的 training time 不能与 GPU run 直接比较。

## Result

| Model | Pretrained | Max Epoch | Best Epoch | Val Acc | Val Macro-F1 | Val Top-5 | Params | Train Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 Full FT + HFlip | ImageNet | 20 | 11 | 69.67% | 待补 | 89.33% | 11.28M | 待补 |
| Swin-Tiny Full FT + HFlip | ImageNet | 30 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

## Decision

结果待 Swin-Tiny 训练完成后填写。判断时不能只看 Accuracy，还要同时看 Macro-F1、Parameter Count 和 Training Time。若 Swin-Tiny 提升明显，后续 retrieval 和 representation learning 可以优先迁移到 Swin backbone；若提升有限，则 ResNet-18 仍是更轻量、可解释、训练成本更低的默认 baseline。
