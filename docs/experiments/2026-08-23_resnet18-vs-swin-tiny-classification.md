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
| ResNet-18 Full FT + HFlip | ImageNet | 20 | 11 | 69.67% | 未记录 | 89.33% | 11.28M | 未记录 |
| Swin-Tiny Full FT + HFlip | ImageNet | 30 | 4 | 75.50% | 75.28% | 93.42% | 27.67M | 18.50 min |

Swin-Tiny 使用 `configs/baseline_swin_tiny_protocol.yaml` 原配置在 RTX 4070 上运行。该配置使用 batch size 8、ImageNet pretrained Swin-Tiny、full fine-tuning、AdamW、backbone LR 5e-5、classifier LR 5e-4、weight decay 1e-4 和 early stopping patience 5。训练日志共记录 9 个 epoch，best checkpoint 位于 epoch 4；由于 epoch 5 到 epoch 9 的 validation accuracy 均未超过 75.50%，early stopping 在 epoch 9 后触发。最高 Top-5 accuracy 是 epoch 7 的 93.75%，表格中记录的是 best accuracy checkpoint 对应的 Top-5 accuracy 93.42%。

## Decision

Swin-Tiny 在 classification validation 上明显优于 ResNet-18：Val Acc 从 69.67% 提升到 75.50%，提升 5.83 percentage points；Val Top-5 从 89.33% 提升到 93.42%，提升 4.09 percentage points。这个结果说明 Transformer-based backbone 在当前 CUB 细粒度分类设置下更强，但它也带来更高参数成本，27.67M 参数约为 ResNet-18 的 2.45 倍。下一步需要执行 Backbone Retrieval Comparison，验证更高分类性能是否同步转化为更好的 embedding retrieval quality。
