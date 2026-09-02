# 实验 9.1：输入分辨率 448

## 实验动机

BBox crop 解决的是无关背景像素过多的问题；提高输入分辨率解决的是鸟主体局部细节像素不足的问题。细粒度鸟类识别依赖喙、眼睛、头部颜色边界、翼部纹理和羽毛边缘等局部线索。如果这些区域在 224 输入中只占很少像素，模型即使结构更强，也很难恢复已经丢失的细节。

因此本阶段把输入从 224 提高到 448。448 相对 224 的像素数量增加 4 倍，训练和推理成本也会明显增加。这个实验不只是为了追求更高数值，而是为了判断：

```text
更高分辨率是否能与 foreground-aware crop 形成叠加收益？
```

## 研究问题

```text
在相同 backbone、split、optimizer、augmentation 和 evaluation protocol 下，
输入分辨率从 224 提高到 448 是否能提升 CUB 细粒度分类和检索性能？
```

## 对照组

| 组别 | Config | 输入 | Resolution | 目的 |
| --- | --- | --- | ---: | --- |
| A | `configs/baseline_swin_tiny_protocol.yaml` | 原图 | 224 | 已有 Swin 原图 baseline |
| B | `configs/foreground_swin_tiny_bbox224.yaml` | BBox crop | 224 | 已有 foreground-aware baseline |
| C | `configs/baseline_swin_tiny_448.yaml` | 原图 | 448 | 验证纯 resolution 提升 |
| D | `configs/foreground_swin_tiny_bbox448.yaml` | BBox crop | 448 | 验证 foreground + detail 是否叠加 |

主要固定变量：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Backbone：ImageNet pretrained Swin-Tiny
- Fine-tuning：full fine-tuning
- Augmentation：Horizontal Flip
- Optimizer：AdamW
- LR：backbone `5e-5`，classifier `5e-4`
- Epoch：30
- Early stopping：5
- Checkpoint selection：最高 validation accuracy

主要变化变量：

```text
image_size: 224 -> 448
crop_mode: none / bbox
```

## 工程注意

448 输入的像素数量是 224 的 4 倍。Swin-Tiny 虽然使用 window attention，但 token 数量增加后，训练时间和内存都会明显上升。

为了保持实验可比性，配置文件默认保留 `batch_size: 8`。如果本机 CPU 或内存跑不动，可以在命令行用 `--batch-size 4` 或 `--batch-size 2` 覆盖。这样会引入一个额外变量，因此最终表格需要记录实际 batch size。

## 训练指令

### C 组：Original 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_448.yaml `
  --device cuda
```

如果 CUDA 显存不足，使用较小 batch：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_448.yaml `
  --device cuda `
  --batch-size 4
```

### D 组：BBox 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\foreground_swin_tiny_bbox448.yaml `
  --device cuda
```

如果 CUDA 显存不足，使用较小 batch：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\foreground_swin_tiny_bbox448.yaml `
  --device cuda `
  --batch-size 4
```

## 检索评估指令

### Original 448 retrieval

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\baseline_swin_tiny_448 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\baseline_swin_tiny_448.yaml `
  --checkpoint "outputs\checkpoints\baseline_swin_tiny_448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\resolution_448
```

### BBox 448 retrieval

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\foreground_swin_tiny_bbox448 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\foreground_swin_tiny_bbox448.yaml `
  --checkpoint "outputs\checkpoints\foreground_swin_tiny_bbox448\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\resolution_448
```

## 结果记录表

| 方法 | Run ID | Resolution | Crop | Batch | Best Epoch | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin Original 224 | `20260823_012929_baseline_swin_tiny_protocol` | 224 | none | 8 | 4 | 75.50% | 75.28% | 93.42% | 61.25% | 84.33% | 90.67% | 49.04% |
| Swin BBox 224 | `20260830_141148_foreground_swin_tiny_bbox224` | 224 | bbox | 8 | 6 | 80.00% | 79.52% | 94.92% | 68.42% | 86.50% | 93.08% | 55.51% |
| Swin Original 448 | `20260831_084639_baseline_swin_tiny_448` | 448 | none | 8 | 8 | 78.83% | 78.37% | 95.08% | 67.08% | 88.17% | 93.58% | 53.92% |
| Swin BBox 448 | `20260831_090913_foreground_swin_tiny_bbox448` | 448 | bbox | 8 | 7 | 81.33% | 81.05% | 95.42% | 70.58% | 89.75% | 93.67% | 58.97% |

当前本地已找到两个 448 设置的 retrieval evaluation 输出：

```text
outputs/embeddings/resolution_448/20260831_084639_baseline_swin_tiny_448/cosine/summary.json
outputs/embeddings/resolution_448/20260831_090913_foreground_swin_tiny_bbox448/cosine/summary.json
```

## 训练阶段详细结果

| 方法 | Run ID | Best Epoch | Val Acc | Macro-F1 | Top-5 Acc | Epochs Ran | Train Time | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin Original 448 | `20260831_084639_baseline_swin_tiny_448` | 8 | 78.83% | 78.37% | 95.08% | 12 | 22.26 min | 27,673,154 |
| Swin BBox 448 | `20260831_090913_foreground_swin_tiny_bbox448` | 7 | 81.33% | 81.05% | 95.42% | 12 | 22.18 min | 27,673,154 |

## 结果分析

从分类结果看，`Original 448` 明显优于 `Original 224`：

```text
Val Acc: 75.50% -> 78.83%  (+3.33 percentage points)
Macro-F1: 75.28% -> 78.37% (+3.09 percentage points)
Top-5 Acc: 93.42% -> 95.08% (+1.66 percentage points)
Recall@1: 61.25% -> 67.08% (+5.83 percentage points)
Recall@5: 84.33% -> 88.17% (+3.84 percentage points)
Recall@10: 90.67% -> 93.58% (+2.91 percentage points)
mAP: 49.04% -> 53.92% (+4.88 percentage points)
```

这说明单纯提高输入分辨率确实能帮助 Swin-Tiny 捕捉更多细粒度视觉信息，而且收益不仅体现在分类 Accuracy，也体现在 retrieval ranking structure。

`BBox 448` 进一步优于 `BBox 224`：

```text
Val Acc: 80.00% -> 81.33%  (+1.33 percentage points)
Macro-F1: 79.52% -> 81.05% (+1.53 percentage points)
Top-5 Acc: 94.92% -> 95.42% (+0.50 percentage points)
Recall@1: 68.42% -> 70.58% (+2.16 percentage points)
Recall@5: 86.50% -> 89.75% (+3.25 percentage points)
Recall@10: 93.08% -> 93.67% (+0.59 percentage points)
mAP: 55.51% -> 58.97% (+3.46 percentage points)
```

这说明在 foreground-aware crop 已经减少背景干扰后，提高 resolution 仍然能带来额外收益。换句话说，BBox crop 和 high resolution 存在正向叠加效果。

`BBox 448` 的 retrieval embedding 维度为 768，在 1200 张验证样本上的 float32 存储约为 3.52 MiB，cosine brute-force 查询延迟约为 0.054 ms/query。这个规模下检索开销仍然很低，因此 448 分辨率带来的主要成本集中在训练和特征提取阶段，而不是后续向量检索阶段。

`BBox 448` 相比 `Original 448` 也有明显分类优势：

```text
Val Acc: 78.83% -> 81.33% (+2.50 percentage points)
Macro-F1: 78.37% -> 81.05% (+2.68 percentage points)
Recall@1: 67.08% -> 70.58% (+3.50 percentage points)
Recall@5: 88.17% -> 89.75% (+1.58 percentage points)
Recall@10: 93.58% -> 93.67% (+0.09 percentage points)
mAP: 53.92% -> 58.97% (+5.05 percentage points)
```

这说明即使在 448 高分辨率下，去除背景并把像素预算集中到鸟主体上仍然重要。

```text
Resolution 提升分类性能。
Resolution 同时提升 retrieval 表征质量。
BBox crop + 448 继续提升分类和 retrieval。
Foreground + Detail 的组合优于单独使用 foreground 224。
```

## 当前结论

当前最强设置是：

```text
Swin-Tiny + BBox crop + 448 input
```

它达到：

```text
Accuracy = 81.33%
Recall@1 = 70.58%
Recall@5 = 89.75%
Recall@10 = 93.67%
mAP = 58.97%
```

这是目前项目中非常强的 foreground + high-resolution baseline。完整结果说明：提高输入分辨率可以增强细粒度分类和检索表征；在此基础上加入 BBox crop 后，模型进一步减少背景干扰，把更多像素预算用于鸟类主体和局部判别区域。
