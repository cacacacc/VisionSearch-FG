# 实验 12：Strong Backbone 上的 CE + SupCon

## 实验动机

前面 Phase 10 和 Phase 11 已经得到两个强基线：

```text
ConvNeXt V2 Tiny + BBox crop + 448 input
DINOv2 Small partial last 2 blocks + BBox crop + 448 input
```

其中 ConvNeXt V2 Tiny 是当前整体检索最强模型，DINOv2 Small partial last 2 blocks 是当前 DINOv2 内部检索最强模型。

Phase 5 已经证明，在 ResNet-18 上加入 SupCon 和 projection head 可以改变 embedding geometry。现在要回答更重要的问题：

```text
当 backbone 已经足够强时，CE + SupCon 是否还能继续提升 retrieval representation？
```

## 研究问题

```text
在强 backbone、BBox crop、448 input 的设置下，
CE + SupCon 是否能在不明显牺牲分类性能的前提下提升 Recall@K 和 mAP？
```

同时比较两种 retrieval feature：

```text
Backbone embedding h
Projection feature z
```

原因是 SupCon loss 作用在 projection space `z` 上，但最终部署时未必一定要使用 `z`。如果 `h` 更强，说明 projection head 主要承担训练约束；如果 `z` 更强，说明 projection space 本身更适合检索。

## 对照组

固定变量：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Input：`BBox crop + 448`
- Augmentation：`hflip`
- Optimizer：AdamW
- Epoch：30
- Early stopping：5
- Loss：`L = L_CE + 0.1 L_SupCon`
- SupCon temperature：`0.2`
- SupCon feature：projection feature `z`

| 组别 | Config | Backbone | Fine-tuning | 目的 |
| --- | --- | --- | --- | --- |
| A | `configs/backbone_convnextv2_tiny_bbox448.yaml` | ConvNeXt V2 Tiny | full CE | 当前整体 retrieval 最强 CE baseline |
| B | `configs/phase12_convnextv2_tiny_bbox448_ce_supcon.yaml` | ConvNeXt V2 Tiny | full CE + SupCon | 测试强 CNN backbone 加 SupCon 是否继续提升 |
| C | `configs/backbone_dinov2_small_bbox448_partial_last2blocks.yaml` | DINOv2 Small | partial CE | 当前 DINOv2 retrieval 最强 baseline |
| D | `configs/phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon.yaml` | DINOv2 Small | partial CE + SupCon | 测试 self-supervised backbone 加 SupCon 是否继续提升 |

## 已知基线

| Model | Training | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt V2 Tiny | CE | 86.42% | 82.75% | 93.75% | 96.92% | 75.06% |
| DINOv2 Small | Partial last 2 blocks CE | 87.25% | 79.75% | 94.67% | 97.67% | 70.48% |

## 训练指令

### 1. ConvNeXt V2 Tiny CE + SupCon

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase12_convnextv2_tiny_bbox448_ce_supcon.yaml `
  --device cuda
```

显存不足时：

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase12_convnextv2_tiny_bbox448_ce_supcon.yaml `
  --device cuda `
  --batch-size 2
```

### 2. DINOv2 Small Partial Last 2 Blocks CE + SupCon

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon.yaml `
  --device cuda
```

显存不足时：

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon.yaml `
  --device cuda `
  --batch-size 2
```

## 检索评估指令

每个模型训练完成后都评估两次：

```text
1. --feature embedding
2. --feature projection
```

### 1. ConvNeXt V2 Tiny：Embedding Retrieval

```powershell
$runConv = Get-ChildItem outputs\checkpoints\phase12_convnextv2_tiny_bbox448_ce_supcon |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase12_convnextv2_tiny_bbox448_ce_supcon.yaml `
  --checkpoint "outputs\checkpoints\phase12_convnextv2_tiny_bbox448_ce_supcon\$runConv\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\phase12_strong_backbone_ce_supcon_embedding
```

### 2. ConvNeXt V2 Tiny：Projection Retrieval

```powershell
$runConv = Get-ChildItem outputs\checkpoints\phase12_convnextv2_tiny_bbox448_ce_supcon |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase12_convnextv2_tiny_bbox448_ce_supcon.yaml `
  --checkpoint "outputs\checkpoints\phase12_convnextv2_tiny_bbox448_ce_supcon\$runConv\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cuda `
  --output-dir outputs\embeddings\phase12_strong_backbone_ce_supcon_projection
```

### 3. DINOv2 Small Partial：Embedding Retrieval

```powershell
$runDino = Get-ChildItem outputs\checkpoints\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon.yaml `
  --checkpoint "outputs\checkpoints\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon\$runDino\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\phase12_strong_backbone_ce_supcon_embedding
```

### 4. DINOv2 Small Partial：Projection Retrieval

```powershell
$runDino = Get-ChildItem outputs\checkpoints\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon.yaml `
  --checkpoint "outputs\checkpoints\phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon\$runDino\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cuda `
  --output-dir outputs\embeddings\phase12_strong_backbone_ce_supcon_projection
```

如果 CUDA 评估报错，将评估指令里的：

```powershell
--device cuda
```

改成：

```powershell
--device cpu
```

## 最终实验表

| Model | Training | Feature | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Storage | Search Latency | Run ID |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt V2 Tiny | CE baseline | embedding | 86.42% | 82.75% | 93.75% | 96.92% | 75.06% | 3.52 MiB | 0.070 ms/query | `20260831_113000_backbone_convnextv2_tiny_bbox448` |
| ConvNeXt V2 Tiny | CE + SupCon | embedding | 84.25% | 79.50% | 93.25% | 96.50% | 72.02% | 3.52 MiB | 0.056 ms/query | `20260902_085648_phase12_convnextv2_tiny_bbox448_ce_supcon` |
| ConvNeXt V2 Tiny | CE + SupCon | projection | 84.25% | 77.00% | 91.58% | 95.08% | 68.24% | 2.34 MiB | 0.055 ms/query | `20260902_085648_phase12_convnextv2_tiny_bbox448_ce_supcon` |
| DINOv2 Small Partial Last 2 Blocks | CE baseline | embedding | 87.25% | 79.75% | 94.67% | 97.67% | 70.48% | 1.76 MiB | 0.054 ms/query | `20260901_123245_backbone_dinov2_small_bbox448_partial_last2blocks` |
| DINOv2 Small Partial Last 2 Blocks | CE + SupCon | embedding | 86.25% | 79.83% | 94.42% | 97.42% | 70.54% | 1.76 MiB | 0.056 ms/query | `20260902_105600_phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon` |
| DINOv2 Small Partial Last 2 Blocks | CE + SupCon | projection | 86.25% | 76.92% | 91.50% | 94.92% | 65.15% | 1.76 MiB | 0.055 ms/query | `20260902_105600_phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon` |

## 当前训练结果

| Model | Run ID | Best Epoch | Accuracy | Macro-F1 | Top-5 Acc | Train CE Loss | Train SupCon Loss | Train Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt V2 Tiny CE + SupCon | `20260902_085648_phase12_convnextv2_tiny_bbox448_ce_supcon` | 6 | 84.25% | 83.96% | 96.17% | 0.0947 | 0.0952 | 62.62 min |
| DINOv2 Small Partial Last 2 Blocks CE + SupCon | `20260902_105600_phase12_dinov2_small_bbox448_partial_last2blocks_ce_supcon` | 6 | 86.25% | 85.91% | 96.58% | 0.0770 | 0.0892 | 20.34 min |

## 当前检索结果

| Model | Feature | Feature Dim | Recall@1 | Recall@5 | Recall@10 | mAP | Search Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt V2 Tiny CE + SupCon | embedding | 768 | 79.50% | 93.25% | 96.50% | 72.02% | 0.056 ms/query |
| ConvNeXt V2 Tiny CE + SupCon | projection | 512 | 77.00% | 91.58% | 95.08% | 68.24% | 0.055 ms/query |
| DINOv2 Small Partial Last 2 Blocks CE + SupCon | embedding | 384 | 79.83% | 94.42% | 97.42% | 70.54% | 0.056 ms/query |
| DINOv2 Small Partial Last 2 Blocks CE + SupCon | projection | 384 | 76.92% | 91.50% | 94.92% | 65.15% | 0.055 ms/query |

## 当前分析

ConvNeXt V2 Tiny 上的 CE + SupCon 没有超过 CE baseline：

```text
CE baseline embedding:      Accuracy = 86.42% / Recall@1 = 82.75% / mAP = 75.06%
CE + SupCon embedding:      Accuracy = 84.25% / Recall@1 = 79.50% / mAP = 72.02%
CE + SupCon projection:     Accuracy = 84.25% / Recall@1 = 77.00% / mAP = 68.24%
```

相比 CE baseline，CE + SupCon 的 backbone embedding 下降为：

```text
Accuracy: 86.42% -> 84.25% (-2.17 percentage points)
Recall@1: 82.75% -> 79.50% (-3.25 percentage points)
Recall@5: 93.75% -> 93.25% (-0.50 percentage points)
Recall@10: 96.92% -> 96.50% (-0.42 percentage points)
mAP: 75.06% -> 72.02% (-3.04 percentage points)
```

projection feature 进一步低于 backbone embedding：

```text
embedding mAP:  72.02%
projection mAP: 68.24%
```

这说明在当前 `BBox crop + 448 input + ConvNeXt V2 Tiny` 设置下，CE fine-tuning 已经学到较强的 retrieval geometry。直接加入 SupCon 并没有继续改善类内邻域结构，反而削弱了分类性能和检索排序。

一个合理解释是：当前 batch size 为 4，two-view 后每个 step 的有效类别和正样本结构仍然有限，SupCon 对强 backbone 的约束质量不足；同时强 backbone 在 CE 目标下已经能形成较好的 fine-grained embedding，额外的 SupCon loss 可能引入了不稳定的几何约束。

因此对 ConvNeXt V2 Tiny，不建议继续简单增大 SupCon 权重。后续若要继续优化 contrastive learning，应优先考虑：

```text
class-balanced sampler
larger effective batch size / gradient accumulation
hard negative mining
part-aware or local-feature retrieval objective
```

DINOv2 Small partial last 2 blocks 上的 CE + SupCon 也没有带来实质提升：

```text
DINOv2 CE baseline embedding:  Accuracy = 87.25% / Recall@1 = 79.75% / mAP = 70.48%
DINOv2 CE + SupCon embedding:  Accuracy = 86.25% / Recall@1 = 79.83% / mAP = 70.54%
DINOv2 CE + SupCon projection: Accuracy = 86.25% / Recall@1 = 76.92% / mAP = 65.15%
```

相比 CE baseline，CE + SupCon 的 backbone embedding 变化为：

```text
Accuracy: 87.25% -> 86.25% (-1.00 percentage points)
Recall@1: 79.75% -> 79.83% (+0.08 percentage points)
Recall@5: 94.67% -> 94.42% (-0.25 percentage points)
Recall@10: 97.67% -> 97.42% (-0.25 percentage points)
mAP: 70.48% -> 70.54% (+0.06 percentage points)
```

这个 `+0.06` mAP 提升过小，不能视为有效提升。更重要的是，projection feature 的 mAP 只有 `65.15%`，明显低于 backbone embedding。这说明在当前设置下，SupCon projection space 并没有形成更好的检索空间。

## 阶段结论

在强 backbone 上，当前 CE + SupCon 设置没有超过对应 CE baseline：

```text
ConvNeXt V2 Tiny:
CE baseline mAP = 75.06%
CE + SupCon embedding mAP = 72.02%
CE + SupCon projection mAP = 68.24%

DINOv2 Small Partial Last 2 Blocks:
CE baseline mAP = 70.48%
CE + SupCon embedding mAP = 70.54%
CE + SupCon projection mAP = 65.15%
```

因此，本实验的结论不是“SupCon 无效”，而是更精确地说：

```text
在当前 batch size、two-view augmentation、lambda=0.1、temperature=0.2 的设置下，
直接把 SupCon 加到强 backbone 上，并不能稳定提升 fine-grained retrieval。
```

Phase 12 的结果支持后续把优化重点转向：

```text
1. class-balanced sampler / PK sampler
2. larger effective batch size 或 memory bank
3. hard negative mining
4. part-aware local retrieval / foreground-aware local feature
```

当前最终模型选择保持不变：

```text
分类最强：DINOv2 Base Frozen + BBox crop + 448 input
DINOv2 内部检索最强：DINOv2 Small Partial Last 2 Blocks + BBox crop + 448 input
整体检索最强：ConvNeXt V2 Tiny + BBox crop + 448 input
```

## 结果解释

CE + SupCon 的 projection feature 没有优于 embedding feature：

```text
ConvNeXt V2 Tiny: embedding mAP = 72.02% / projection mAP = 68.24%
DINOv2 Small:     embedding mAP = 70.54% / projection mAP = 65.15%
```

这说明当前 projection head 更像是训练时承接 SupCon objective 的辅助空间，而不是更适合最终 retrieval deployment 的表示空间。

CE + SupCon 的 embedding feature 也没有稳定超过 CE baseline：

```text
ConvNeXt V2 Tiny: 75.06% -> 72.02%
DINOv2 Small:     70.48% -> 70.54%
```

DINOv2 的 `+0.06` mAP 过小，不足以构成可靠收益。因此不能声称 SupCon 改善了强 backbone 的 backbone representation。

两个强 backbone 的 Accuracy 都出现下降：

```text
ConvNeXt V2 Tiny: 86.42% -> 84.25%
DINOv2 Small:     87.25% -> 86.25%
```

因此当前 CE + SupCon 版本不应作为主模型。ConvNeXt V2 Tiny CE baseline 仍然是整体检索最强模型。
