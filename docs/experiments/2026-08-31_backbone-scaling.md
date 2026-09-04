# 实验 10.1：更强 Backbone 优化

## 实验动机

前面的实验已经说明，`BBox crop + 448 input` 能同时提升分类和 retrieval 表征。现在输入信息已经更加集中、局部细节也更充分，下一个问题是：

```text
当前 encoder 是否有足够能力表达这些细粒度差异？
```

ResNet18 容量较小，Swin-Tiny 已经明显优于早期 ResNet baseline，说明 representation capacity 很可能是项目瓶颈之一。因此本实验比较更强 visual backbone，包括 ConvNeXt V2 Nano / Tiny，以及算力允许时的 Swin-Small。

## 研究问题

```text
在相同数据划分、输入分辨率、crop 策略和训练协议下，
更强 backbone 是否能提升 CUB 细粒度分类性能与 retrieval representation？
```

同时需要回答：

```text
额外参数量、训练时间和检索成本是否值得换取性能提升？
```

## 对照组

主实验固定：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Input：`BBox crop + 448`
- Fine-tuning：full fine-tuning
- Augmentation：Horizontal Flip
- Optimizer：AdamW
- LR：backbone `5e-5`，classifier `5e-4`
- Epoch：30
- Early stopping：5
- Checkpoint selection：最高 validation accuracy

| 组别 | Config | Backbone | Pretraining | 目的 |
| --- | --- | --- | --- | --- |
| A | `configs/backbone_resnet18_bbox448.yaml` | ResNet18 | ImageNet-1K | 低容量 CNN 对照 |
| B | `configs/foreground_swin_tiny_bbox448.yaml` | Swin-Tiny | ImageNet-1K | 当前最强 baseline |
| C | `configs/backbone_convnextv2_nano_bbox448.yaml` | ConvNeXt V2 Nano | FCMAE + ImageNet-22K -> 1K | 小型 ConvNeXt V2 |
| D | `configs/backbone_convnextv2_tiny_bbox448.yaml` | ConvNeXt V2 Tiny | FCMAE + ImageNet-22K -> 1K | 更强 ConvNeXt V2 |
| E | `configs/backbone_swin_small_bbox448.yaml` | Swin-Small | ImageNet-22K -> 1K | 算力允许时的 Transformer 扩展 |

其中 Swin-Small 使用 timm backbone，并在 YAML 中显式设置：

```yaml
timm_kwargs:
  img_size: 448
```

原因是 timm 的 Swin-Small pretrained variant 名称中带有 `224`，默认构造时会检查输入尺寸；这里需要让模型结构按 448 输入建立。

注意：ConvNeXt V2 的 pretrained source 与 Swin-Tiny 不完全一致，因此如果 ConvNeXt V2 更强，结论应表述为：

```text
更强 architecture + 更强 pretraining 的组合带来收益。
```

而不是简单说：

```text
architecture 一定更好。
```

## 训练指令

### A 组：ResNet18 + BBox 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_resnet18_bbox448.yaml `
  --device cuda
```

### C 组：ConvNeXt V2 Nano + BBox 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_convnextv2_nano_bbox448.yaml `
  --device cuda
```

### D 组：ConvNeXt V2 Tiny + BBox 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --device cuda
```

### E 组：Swin-Small + BBox 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_swin_small_bbox448.yaml `
  --device cuda
```

如果 CUDA 显存不足，优先将 batch size 降到 4：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --device cuda `
  --batch-size 4
```

Swin-Small 如果仍然 OOM，可以继续降到 2。

## 检索评估指令

训练完成后，把 `$run` 替换成对应 checkpoint 目录名。

### ResNet18 retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_resnet18_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_resnet18_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

### ConvNeXt V2 Nano retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_convnextv2_nano_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_convnextv2_nano_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

### ConvNeXt V2 Tiny retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_convnextv2_tiny_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

### Swin-Small retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_swin_small_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_swin_small_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

## 本轮待补检索评估

### ConvNeXt V2 Nano

```powershell
$run = "20260831_110608_backbone_convnextv2_nano_bbox448"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_convnextv2_nano_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_convnextv2_nano_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

### ConvNeXt V2 Tiny

```powershell
$run = "20260831_113000_backbone_convnextv2_tiny_bbox448"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_convnextv2_tiny_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

### Swin-Small

```powershell
$run = "20260831_115625_backbone_swin_small_bbox448"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_swin_small_bbox448.yaml `
  --checkpoint "outputs\checkpoints\backbone_swin_small_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\backbone_scaling_bbox448
```

## 最终实验表

| Backbone | Config | Params | Resolution | Crop | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Search Latency | Train Time |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | `backbone_resnet18_bbox448.yaml` | 待运行 | 448 | bbox | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| Swin-Tiny | `foreground_swin_tiny_bbox448.yaml` | 27,673,154 | 448 | bbox | 81.33% | 70.58% | 89.75% | 93.67% | 58.97% | 0.054 ms/query | 22.18 min |
| ConvNeXt V2 Nano | `backbone_convnextv2_nano_bbox448.yaml` | 15,111,000 | 448 | bbox | 87.00% | 78.42% | 93.00% | 96.33% | 68.55% | 0.092 ms/query | 23.28 min |
| ConvNeXt V2 Tiny | `backbone_convnextv2_tiny_bbox448.yaml` | 28,020,296 | 448 | bbox | 86.42% | 82.75% | 93.75% | 96.92% | 75.06% | 0.070 ms/query | 24.21 min |
| Swin-Small | `backbone_swin_small_bbox448.yaml` | 48,991,058 | 448 | bbox | 86.50% | 81.92% | 94.75% | 96.75% | 72.82% | 0.064 ms/query | 26.86 min |

## 训练阶段详细结果

| Backbone | Run ID | Params | Best Epoch | Accuracy | Macro-F1 | Top-5 Acc | Epochs Ran | Train Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny | `20260831_090913_foreground_swin_tiny_bbox448` | 27,673,154 | 7 | 81.33% | 81.05% | 95.42% | 12 | 22.18 min |
| ConvNeXt V2 Nano | `20260831_110608_backbone_convnextv2_nano_bbox448` | 15,111,000 | 7 | 87.00% | 86.72% | 97.08% | 12 | 23.28 min |
| ConvNeXt V2 Tiny | `20260831_113000_backbone_convnextv2_tiny_bbox448` | 28,020,296 | 3 | 86.42% | 85.74% | 97.33% | 8 | 24.21 min |
| Swin-Small | `20260831_115625_backbone_swin_small_bbox448` | 48,991,058 | 5 | 86.50% | 86.07% | 97.25% | 10 | 26.86 min |

当前已找到三组新 backbone 的 retrieval evaluation 输出：

```text
outputs/embeddings/backbone_scaling_bbox448/<run_id>/cosine/summary.json
```

## 当前结果分析

从分类结果看，三组更强 backbone 都显著优于当前最强的 `Swin-Tiny + BBox 448` baseline：

```text
Swin-Tiny:          81.33% Accuracy / 81.05% Macro-F1
ConvNeXt V2 Nano:  87.00% Accuracy / 86.72% Macro-F1
ConvNeXt V2 Tiny:  86.42% Accuracy / 85.74% Macro-F1
Swin-Small:        86.50% Accuracy / 86.07% Macro-F1
```

其中 ConvNeXt V2 Nano 的结果最值得注意。它只有 15.11M 参数，明显小于 Swin-Tiny 的 27.67M 参数，但 Accuracy 提升了 5.67 个百分点，Macro-F1 提升了 5.67 个百分点，训练时间只从 22.18 min 增加到 23.28 min。这说明在当前 `BBox crop + 448` 条件下，ConvNeXt V2 Nano 是更高性价比的 backbone。

从 retrieval 结果看，ConvNeXt V2 Tiny 最强：

```text
Swin-Tiny:          Recall@1 = 70.58% / mAP = 58.97%
ConvNeXt V2 Nano:  Recall@1 = 78.42% / mAP = 68.55%
ConvNeXt V2 Tiny:  Recall@1 = 82.75% / mAP = 75.06%
Swin-Small:        Recall@1 = 81.92% / mAP = 72.82%
```

这说明分类 Accuracy 最强的 backbone 不一定给出最好的 retrieval ranking structure。Nano 的分类最好，但 Tiny 的 embedding 排序质量更强，mAP 比 Swin-Tiny 提升 16.09 个百分点。

ConvNeXt V2 Tiny 和 Swin-Tiny 参数量接近：

```text
Swin-Tiny:          27.67M params
ConvNeXt V2 Tiny:  28.02M params
```

但 ConvNeXt V2 Tiny 达到更高的 Accuracy、Recall@1 和 mAP。因此在同等参数规模下，ConvNeXt V2 Tiny 是更强的 retrieval backbone。

Swin-Small 的参数量最大，Recall@5 最高，但 Accuracy 和 mAP 没有超过 ConvNeXt V2 Tiny。这提示当前任务不是简单的“参数越多越好”，更强预训练、小数据 fine-tuning 稳定性和模型 inductive bias 都会影响最终结果。

当前训练阶段的最强设置是：

```text
ConvNeXt V2 Nano + BBox crop + 448 input
```

当前检索阶段的最强设置是：

```text
ConvNeXt V2 Tiny + BBox crop + 448 input
```

因此最终应保留两个结论：

```text
分类优先：ConvNeXt V2 Nano。
检索优先：ConvNeXt V2 Tiny。
综合推荐：ConvNeXt V2 Tiny，适合作为下一阶段 retrieval representation 的主 backbone。
```

## 效率分析

当前 retrieval latency 是在 1200 张 validation 图像的 embedding 上做 cosine brute-force search 得到的 CPU 侧结果。所有 backbone 的单 query 检索时间都低于 0.1 ms，因此在当前数据规模下，检索阶段不是主要瓶颈。

需要注意的是，这里的 Search Latency 只衡量 embedding 已经提取完成后的向量检索时间，不包含模型前向推理时间。对于真实系统，ConvNeXt V2 Tiny 和 Swin-Small 的主要额外成本会体现在 feature extraction，而不是向量检索本身。

## 最终结论

本实验说明，更强 backbone 是当前项目非常有效的优化方向。

分类性能方面，ConvNeXt V2 Nano 最强：

```text
Accuracy = 87.00%
Macro-F1 = 86.72%
Top-5 Acc = 97.08%
```

检索性能方面，ConvNeXt V2 Tiny 最强：

```text
Recall@1 = 82.75%
Recall@5 = 93.75%
Recall@10 = 96.92%
mAP = 75.06%
```

相对 `Swin-Tiny + BBox 448`，ConvNeXt V2 Tiny 在几乎相同参数量下获得了非常明显的 retrieval 提升：

```text
Accuracy: 81.33% -> 86.42% (+5.09 percentage points)
Recall@1: 70.58% -> 82.75% (+12.17 percentage points)
Recall@5: 89.75% -> 93.75% (+4.00 percentage points)
Recall@10: 93.67% -> 96.92% (+3.25 percentage points)
mAP: 58.97% -> 75.06% (+16.09 percentage points)
```

因此，后续如果目标是分类 leaderboard，可以优先使用：

```text
ConvNeXt V2 Nano + BBox crop + 448 input
```

如果目标是 image retrieval / visual search 系统，应优先使用：

```text
ConvNeXt V2 Tiny + BBox crop + 448 input
```

这个结果也说明，当前项目的性能瓶颈不只是输入分辨率和 foreground crop，encoder 的 representation capacity 与 pretrained representation quality 同样关键。下一步可以在 ConvNeXt V2 Tiny 上继续研究 SupCon、DINOv2 或更强 retrieval-specific representation learning。
