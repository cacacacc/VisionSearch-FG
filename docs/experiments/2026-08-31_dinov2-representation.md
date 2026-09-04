# 实验 11.1：DINOv2 Self-Supervised Representation

## 实验动机

前面的 backbone scaling 实验表明，`ConvNeXt V2 Tiny + BBox crop + 448 input` 已经把 retrieval mAP 提升到 75.06%。这说明更强 backbone 和更强 pretraining 对 retrieval representation 非常关键。

下一步不应该只是继续增大普通 supervised backbone，而是测试更强的 self-supervised pretrained visual backbone：

```text
DINOv2
```

DINOv2 的核心价值在于，它不是只为 ImageNet 分类训练，而是学习更通用的视觉表征。对于 image retrieval，通用表征质量可能比分类头本身更重要。

## 研究问题

```text
相比 ImageNet supervised / weakly supervised backbone，
DINOv2 的 self-supervised representation 是否能进一步提升 CUB 细粒度检索性能？
```

同时关注：

```text
Frozen pretrained representation 是否已经足够强？
小数据集上 full fine-tuning 是否会继续提升，还是破坏 pretrained representation？
```

## 对照组

固定变量：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Input：`BBox crop + 448`
- Augmentation：Horizontal Flip
- Optimizer：AdamW
- Epoch：30
- Early stopping：5
- Checkpoint selection：最高 validation accuracy

| 组别 | Config | Backbone | Fine-tuning | 目的 |
| --- | --- | --- | --- | --- |
| A | `configs/backbone_convnextv2_tiny_bbox448.yaml` | ConvNeXt V2 Tiny | full | 当前 retrieval 最强 baseline |
| B | `configs/backbone_dinov2_small_bbox448_frozen.yaml` | DINOv2 ViT-S/14 | frozen linear probe | 测试 frozen self-supervised representation |
| C | `configs/backbone_dinov2_base_bbox448_frozen.yaml` | DINOv2 ViT-B/14 | frozen linear probe | 测试更大 DINOv2 frozen representation |
| D | `configs/backbone_dinov2_small_bbox448_fullft.yaml` | DINOv2 ViT-S/14 | full fine-tuning | 测试小数据 full fine-tuning 是否有效 |
| E | `configs/backbone_dinov2_small_bbox448_partial_lastblock.yaml` | DINOv2 ViT-S/14 | partial last block | 测试保守解冻是否优于 full fine-tuning |
| F | `configs/backbone_dinov2_small_bbox448_partial_last2blocks.yaml` | DINOv2 ViT-S/14 | partial last 2 blocks | 测试更大范围保守解冻是否继续提升 |

DINOv2 配置中显式使用：

```yaml
timm_kwargs:
  img_size: 448
  global_pool: token
```

原因是 DINOv2 的 pretrained checkpoint 对应 class-token representation。如果使用 `global_pool: avg`，timm 会创建 `fc_norm`，导致 checkpoint 中的 `norm.weight` / `norm.bias` 和模型结构不匹配。

## 训练指令

### B 组：DINOv2 Small Frozen

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_small_bbox448_frozen.yaml `
  --device cuda
```

### C 组：DINOv2 Base Frozen

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_base_bbox448_frozen.yaml `
  --device cuda
```

### D 组：DINOv2 Small Full Fine-tuning

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_small_bbox448_fullft.yaml `
  --device cuda
```

如果 CUDA 显存不足，优先降低 batch size：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_base_bbox448_frozen.yaml `
  --device cuda `
  --batch-size 4
```

DINOv2 Base 如果仍然 OOM，可以降到 2。第一次运行可能需要下载 pretrained weights。

## 检索评估指令

训练完成后，把 `$run` 替换成对应 checkpoint 目录名。

### DINOv2 Small Frozen Retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_dinov2_small_bbox448_frozen.yaml `
  --checkpoint "outputs\checkpoints\backbone_dinov2_small_bbox448_frozen\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\dinov2_bbox448
```

### DINOv2 Base Frozen Retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_dinov2_base_bbox448_frozen.yaml `
  --checkpoint "outputs\checkpoints\backbone_dinov2_base_bbox448_frozen\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\dinov2_bbox448
```

### DINOv2 Small Full Fine-tuning Retrieval

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_dinov2_small_bbox448_fullft.yaml `
  --checkpoint "outputs\checkpoints\backbone_dinov2_small_bbox448_fullft\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\dinov2_bbox448
```

## 最终实验表

| Backbone | Config | Fine-tuning | Params | Resolution | Crop | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Search Latency | Train Time |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt V2 Tiny | `backbone_convnextv2_tiny_bbox448.yaml` | full | 28,020,296 | 448 | bbox | 86.42% | 82.75% | 93.75% | 96.92% | 75.06% | 0.070 ms/query | 24.21 min |
| DINOv2 Small | `backbone_dinov2_small_bbox448_frozen.yaml` | frozen | 22,000,712 | 448 | bbox | 87.42% | 75.67% | 93.67% | 96.58% | 63.33% | 0.052 ms/query | 17.67 min |
| DINOv2 Base | `backbone_dinov2_base_bbox448_frozen.yaml` | frozen | 86,468,552 | 448 | bbox | 89.25% | 78.00% | 93.33% | 96.75% | 68.02% | 0.053 ms/query | 41.66 min |
| DINOv2 Small | `backbone_dinov2_small_bbox448_fullft.yaml` | full | 22,000,712 | 448 | bbox | 77.25% | 70.08% | 89.08% | 94.17% | 56.38% | 0.053 ms/query | 25.23 min |
| DINOv2 Small | `backbone_dinov2_small_bbox448_partial_lastblock.yaml` | partial last block | 22,000,712 | 448 | bbox | 87.17% | 78.58% | 93.92% | 97.08% | 67.84% | 0.054 ms/query | 13.56 min |
| DINOv2 Small | `backbone_dinov2_small_bbox448_partial_last2blocks.yaml` | partial last 2 blocks | 22,000,712 | 448 | bbox | 87.25% | 79.75% | 94.67% | 97.67% | 70.48% | 0.054 ms/query | 15.35 min |

## 当前训练结果

| Backbone | Run ID | Fine-tuning | Params | Trainable Params | Best Epoch | Accuracy | Macro-F1 | Top-5 Acc | Epochs Ran | Train Time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 Small | `20260831_130335_backbone_dinov2_small_bbox448_frozen` | frozen | 22,000,712 | 77,000 | 14 | 87.42% | 87.39% | 97.75% | 18 | 17.67 min |
| DINOv2 Base | `20260901_084631_backbone_dinov2_base_bbox448_frozen` | frozen | 86,468,552 | 153,800 | 14 | 89.25% | 89.06% | 98.17% | 19 | 41.66 min |
| DINOv2 Small | `20260901_094749_backbone_dinov2_small_bbox448_fullft` | full | 22,000,712 | 22,000,712 | 6 | 77.25% | 76.84% | 94.67% | 11 | 25.23 min |
| DINOv2 Small | `20260901_120505_backbone_dinov2_small_bbox448_partial_lastblock` | partial last block | 22,000,712 | 1,853,000 | 6 | 87.17% | 87.00% | 98.25% | 11 | 13.56 min |
| DINOv2 Small | `20260901_123245_backbone_dinov2_small_bbox448_partial_last2blocks` | partial last 2 blocks | 22,000,712 | 3,628,232 | 8 | 87.25% | 86.99% | 97.42% | 13 | 15.35 min |

## 当前检索结果

| Backbone | Run ID | Feature Dim | Storage | Recall@1 | Recall@5 | Recall@10 | mAP | Search Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 Small | `20260831_130335_backbone_dinov2_small_bbox448_frozen` | 384 | 1.76 MiB | 75.67% | 93.67% | 96.58% | 63.33% | 0.052 ms/query |
| DINOv2 Base | `20260901_084631_backbone_dinov2_base_bbox448_frozen` | 768 | 3.52 MiB | 78.00% | 93.33% | 96.75% | 68.02% | 0.053 ms/query |
| DINOv2 Small | `20260901_094749_backbone_dinov2_small_bbox448_fullft` | 384 | 1.76 MiB | 70.08% | 89.08% | 94.17% | 56.38% | 0.053 ms/query |
| DINOv2 Small | `20260901_120505_backbone_dinov2_small_bbox448_partial_lastblock` | 384 | 1.76 MiB | 78.58% | 93.92% | 97.08% | 67.84% | 0.054 ms/query |
| DINOv2 Small | `20260901_123245_backbone_dinov2_small_bbox448_partial_last2blocks` | 384 | 1.76 MiB | 79.75% | 94.67% | 97.67% | 70.48% | 0.054 ms/query |

当前已找到 DINOv2 Small Frozen、DINOv2 Base Frozen、DINOv2 Small Full Fine-tuning、DINOv2 Small Partial Last Block 和 DINOv2 Small Partial Last 2 Blocks 的 retrieval evaluation 输出：

```text
outputs/embeddings/dinov2_bbox448/20260831_130335_backbone_dinov2_small_bbox448_frozen/cosine/summary.json
outputs/embeddings/dinov2_bbox448/20260901_084631_backbone_dinov2_base_bbox448_frozen/cosine/summary.json
outputs/embeddings/dinov2_bbox448/20260901_094749_backbone_dinov2_small_bbox448_fullft/cosine/summary.json
outputs/embeddings/dinov2_bbox448/20260901_120505_backbone_dinov2_small_bbox448_partial_lastblock/cosine/summary.json
outputs/embeddings/dinov2_bbox448/20260901_123245_backbone_dinov2_small_bbox448_partial_last2blocks/cosine/summary.json
```

## 当前分析

DINOv2 Small Frozen 的结果非常关键。它只训练 77,000 个分类头参数，backbone 完全冻结，但分类结果已经超过所有上一阶段 full fine-tuning backbone：

```text
ConvNeXt V2 Nano full fine-tuning: 87.00% Accuracy / 86.72% Macro-F1
ConvNeXt V2 Tiny full fine-tuning: 86.42% Accuracy / 85.74% Macro-F1
DINOv2 Small frozen:              87.42% Accuracy / 87.39% Macro-F1
DINOv2 Base frozen:               89.25% Accuracy / 89.06% Macro-F1
```

这说明 DINOv2 的 self-supervised pretrained representation 本身已经具备很强的 CUB 细粒度可分性。Base Frozen 只训练 153,800 个分类头参数，就把 Accuracy 提升到 89.25%，是目前所有 validation classification 实验中的最强结果。

DINOv2 Base 相比 DINOv2 Small 的分类收益为：

```text
Accuracy: 87.42% -> 89.25% (+1.83 percentage points)
Macro-F1: 87.39% -> 89.06% (+1.67 percentage points)
Top-5 Acc: 97.75% -> 98.17% (+0.42 percentage points)
```

这说明扩大 DINOv2 backbone 对 CUB 细粒度线性分类仍然有效，但训练时间也从 17.67 min 增加到 41.66 min，成本明显上升。

DINOv2 Small full fine-tuning 明显低于 frozen linear probe：

```text
DINOv2 Small frozen:  87.42% Accuracy / 87.39% Macro-F1 / 63.33% mAP
DINOv2 Small full:    77.25% Accuracy / 76.84% Macro-F1 / 56.38% mAP
```

这说明在当前训练协议下，小数据集上的全量 CE fine-tuning 同时破坏了 DINOv2 原本很强的分类可分性和 embedding ranking structure。虽然 backbone learning rate 已经设置为 `1e-5`，模型仍然快速过拟合：训练 accuracy 接近 96% 后，validation accuracy 停在 77% 左右。这个结果支持一个重要判断：

```text
DINOv2 不适合直接用当前 CE full fine-tuning 协议训练。
```

如果后续还想 fine-tune DINOv2，应采用更保守的策略，例如只解冻最后 1-2 个 transformer block、更小 backbone learning rate、layer-wise learning rate decay，或者使用 retrieval-oriented loss。

DINOv2 Small full fine-tuning 相比 frozen 的 retrieval 下降为：

```text
Recall@1: 75.67% -> 70.08% (-5.59 percentage points)
Recall@5: 93.67% -> 89.08% (-4.59 percentage points)
Recall@10: 96.58% -> 94.17% (-2.41 percentage points)
mAP: 63.33% -> 56.38% (-6.95 percentage points)
```

DINOv2 Small partial last block 的分类结果明显优于 full fine-tuning，并且基本接近 frozen linear probe：

```text
DINOv2 Small frozen:             87.42% Accuracy / 87.39% Macro-F1 / 17.67 min
DINOv2 Small full fine-tuning:   77.25% Accuracy / 76.84% Macro-F1 / 25.23 min
DINOv2 Small partial last block: 87.17% Accuracy / 87.00% Macro-F1 / 13.56 min
DINOv2 Small partial last 2 blocks: 87.25% Accuracy / 86.99% Macro-F1 / 15.35 min
```

这说明保守解冻最后 1-2 个 transformer block 都没有破坏 DINOv2 pretrained representation，明显比 full fine-tuning 更稳定。`partial last 2 blocks` 的分类 Accuracy 略高于 `partial last block`，Macro-F1 和 Top-5 Acc 没有同步提升，但 retrieval 指标进一步提升，因此它是当前 DINOv2 Small 系列中最好的 fine-tuning 方案。

```text
DINOv2 Small frozen:                Recall@1 = 75.67% / Recall@5 = 93.67% / Recall@10 = 96.58% / mAP = 63.33%
DINOv2 Small partial last block:    Recall@1 = 78.58% / Recall@5 = 93.92% / Recall@10 = 97.08% / mAP = 67.84%
DINOv2 Small partial last 2 blocks: Recall@1 = 79.75% / Recall@5 = 94.67% / Recall@10 = 97.67% / mAP = 70.48%
DINOv2 Base frozen:                 Recall@1 = 78.00% / Recall@5 = 93.33% / Recall@10 = 96.75% / mAP = 68.02%
```

相比 DINOv2 Small frozen，partial last 2 blocks 的 retrieval 提升为：

```text
Recall@1: 75.67% -> 79.75% (+4.08 percentage points)
Recall@5: 93.67% -> 94.67% (+1.00 percentage points)
Recall@10: 96.58% -> 97.67% (+1.09 percentage points)
mAP: 63.33% -> 70.48% (+7.15 percentage points)
```

相比 DINOv2 Small partial last block，partial last 2 blocks 也有稳定提升：

```text
Recall@1: 78.58% -> 79.75% (+1.17 percentage points)
Recall@5: 93.92% -> 94.67% (+0.75 percentage points)
Recall@10: 97.08% -> 97.67% (+0.59 percentage points)
mAP: 67.84% -> 70.48% (+2.64 percentage points)
```

这个结果说明，partial fine-tuning 对 DINOv2 是有效路线：它基本保留了 pretrained representation，同时显著改善 CUB retrieval 的邻域排序结构。更重要的是，DINOv2 Small partial last 2 blocks 用 384 维 embedding 和 1.76 MiB 存储，已经超过 DINOv2 Base frozen 的 retrieval mAP。

不过它仍然没有超过 ConvNeXt V2 Tiny：

```text
ConvNeXt V2 Tiny full fine-tuning:   Recall@1 = 82.75% / mAP = 75.06%
DINOv2 Small partial last 2 blocks:  Recall@1 = 79.75% / mAP = 70.48%
```

因此当前主检索模型仍然是 `ConvNeXt V2 Tiny + BBox crop + 448 input`。DINOv2 partial fine-tuning 作为一个很有价值的对照，证明了 frozen self-supervised representation 可以通过小范围适配获得明显 retrieval 收益。

retrieval 方面，DINOv2 Base 相比 DINOv2 Small 有明显提升：

```text
Recall@1: 75.67% -> 78.00% (+2.33 percentage points)
Recall@5: 93.67% -> 93.33% (-0.34 percentage points)
Recall@10: 96.58% -> 96.75% (+0.17 percentage points)
mAP: 63.33% -> 68.02% (+4.69 percentage points)
```

这说明扩大 DINOv2 backbone 确实改善了 embedding ranking structure，尤其体现在 mAP 上。

但是 DINOv2 Base Frozen 的 retrieval 仍然没有超过 ConvNeXt V2 Tiny：

```text
ConvNeXt V2 Tiny full fine-tuning: Recall@1 = 82.75% / mAP = 75.06%
DINOv2 Small frozen:              Recall@1 = 75.67% / mAP = 63.33%
DINOv2 Base frozen:               Recall@1 = 78.00% / mAP = 68.02%
DINOv2 Small partial last 2 blocks: Recall@1 = 79.75% / mAP = 70.48%
```

这说明 DINOv2 frozen 的线性分类可分性很强，但 embedding 的邻域排序结构还不是当前最优。换句话说，高 Accuracy 不必然意味着高 mAP。这个结果很好地支持了本项目的核心观点：

```text
Classification performance 和 retrieval representation quality 相关，但不是同一个目标。
```

DINOv2 Base 的 embedding 维度为 768，存储约 3.52 MiB，查询延迟约 0.053 ms/query，与 ConvNeXt V2 Tiny 和 Swin-Tiny 的向量检索开销接近。因此 DINOv2 Base 的主要成本不是向量检索，而是训练和特征提取阶段。

从当前结果看，最强模型需要按任务目标区分：

```text
分类最强：DINOv2 Base Frozen + BBox crop + 448 input
检索最强：ConvNeXt V2 Tiny + BBox crop + 448 input
```

当前 visual search 主 backbone 仍然是：

```text
ConvNeXt V2 Tiny + BBox crop + 448 input
```

## 下一步实验

下一步不建议继续做 DINOv2 full fine-tuning。更合理的方向是：

```text
DINOv2 frozen / partial fine-tuning
Vs
ConvNeXt V2 Tiny full fine-tuning
```

其中 DINOv2 partial fine-tuning 可以只解冻最后 1-2 个 transformer block，并使用更小的 backbone learning rate。这个实验能回答：是否可以保留 DINOv2 pretrained representation，同时让它适配 CUB 细粒度检索。

## DINOv2 Partial Fine-tuning 指令

### 运行顺序

优先运行：

```text
1. DINOv2 Small partial last block
2. DINOv2 Small partial last 2 blocks
```

当前 last 2 blocks 已经完成，并且 retrieval mAP 从 frozen 的 `63.33%` 提升到 `70.48%`。该结果证明，更大范围但仍然保守的 partial fine-tuning 可以继续改善 embedding ranking。

### 训练：Last Block

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_small_bbox448_partial_lastblock.yaml `
  --device cuda
```

显存不足时：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_small_bbox448_partial_lastblock.yaml `
  --device cuda `
  --batch-size 4
```

### 评估：Last Block Retrieval

训练完成后，将 `$run` 替换为 `outputs\checkpoints\backbone_dinov2_small_bbox448_partial_lastblock` 下生成的 run id。

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_dinov2_small_bbox448_partial_lastblock.yaml `
  --checkpoint "outputs\checkpoints\backbone_dinov2_small_bbox448_partial_lastblock\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\dinov2_bbox448
```

### 训练：Last 2 Blocks

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_small_bbox448_partial_last2blocks.yaml `
  --device cuda
```

显存不足时：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_dinov2_small_bbox448_partial_last2blocks.yaml `
  --device cuda `
  --batch-size 4
```

### 评估：Last 2 Blocks Retrieval

训练完成后，将 `$run` 替换为 `outputs\checkpoints\backbone_dinov2_small_bbox448_partial_last2blocks` 下生成的 run id。

```powershell
$run = "<run_id>"

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\backbone_dinov2_small_bbox448_partial_last2blocks.yaml `
  --checkpoint "outputs\checkpoints\backbone_dinov2_small_bbox448_partial_last2blocks\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\dinov2_bbox448
```

### 结果整理位置

结果继续同步到本文档的最终实验表和当前检索结果表：

```text
docs/experiments/2026-08-31_dinov2-representation.md
```

## 阶段结论与后续假设

当前 DINOv2 frozen 结果没有超过 ConvNeXt V2 Tiny 的 retrieval mAP。这说明 DINOv2 的 self-supervised representation 具有很强的线性分类能力，但直接使用 frozen embedding 做 CUB retrieval 时，类内邻域结构仍不如经过 CUB full fine-tuning 的 ConvNeXt V2 Tiny。

DINOv2 partial fine-tuning 的结果明显优于 frozen 和 full fine-tuning：

```text
DINOv2 Small frozen:                Accuracy = 87.42% / mAP = 63.33%
DINOv2 Small full fine-tuning:      Accuracy = 77.25% / mAP = 56.38%
DINOv2 Small partial last block:    Accuracy = 87.17% / mAP = 67.84%
DINOv2 Small partial last 2 blocks: Accuracy = 87.25% / mAP = 70.48%
```

这说明 DINOv2 在小规模细粒度数据集上不适合直接全量 CE fine-tuning，但适合小范围、低学习率的 partial fine-tuning。当前 DINOv2 最优检索方案是 `DINOv2 Small partial last 2 blocks`。

DINOv2 Base Frozen 的 Accuracy 是目前最高，但 mAP 仍低于 ConvNeXt V2 Tiny，这强化了项目中的重要论点：

```text
classification head 的线性可分性
不等于
embedding space 的 retrieval ranking quality
```

当前 DINOv2 Small full fine-tuning 已经明显降低分类 Accuracy 和 retrieval mAP：

```text
Accuracy: 87.42% -> 77.25%
mAP: 63.33% -> 56.38%
```

因此 full fine-tuning 路线暂时不适合作为主模型。DINOv2 后续应优先采用 partial fine-tuning / adapter / LoRA / metric learning，而不是全量 CE fine-tuning。

当前最终选择为：

```text
分类最强：DINOv2 Base Frozen + BBox crop + 448 input
DINOv2 内部检索最强：DINOv2 Small Partial Last 2 Blocks + BBox crop + 448 input
整体检索最强：ConvNeXt V2 Tiny + BBox crop + 448 input
```
