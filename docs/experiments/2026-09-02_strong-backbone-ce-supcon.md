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
| ConvNeXt V2 Tiny | CE + SupCon | embedding |  |  |  |  |  |  |  |  |
| ConvNeXt V2 Tiny | CE + SupCon | projection |  |  |  |  |  |  |  |  |
| DINOv2 Small Partial Last 2 Blocks | CE baseline | embedding | 87.25% | 79.75% | 94.67% | 97.67% | 70.48% | 1.76 MiB | 0.054 ms/query | `20260901_123245_backbone_dinov2_small_bbox448_partial_last2blocks` |
| DINOv2 Small Partial Last 2 Blocks | CE + SupCon | embedding |  |  |  |  |  |  |  |  |
| DINOv2 Small Partial Last 2 Blocks | CE + SupCon | projection |  |  |  |  |  |  |  |  |

## 预期结论

如果 CE + SupCon 的 projection feature 明显优于 embedding feature，说明 contrastive objective 学到的检索空间主要存在于 projection head 中。

如果 CE + SupCon 的 embedding feature 也超过 CE baseline，说明 SupCon 不只是改善 projection space，也真正改善了 backbone representation。

如果 Accuracy 下降但 mAP 上升，需要把该模型定位为 retrieval-oriented model，而不是 classification-oriented model。

如果 ConvNeXt V2 Tiny CE + SupCon 不能超过 CE baseline，则说明当前强 CNN backbone 的 CE fine-tuning 已经形成较好的 retrieval geometry，后续应优先做 sampler、hard negative mining 或 part-aware retrieval，而不是继续增加 SupCon 权重。
