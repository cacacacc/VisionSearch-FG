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
  --device cpu
```

如果 CPU 太慢或内存不足，使用较小 batch：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_448.yaml `
  --device cpu `
  --batch-size 4
```

### D 组：BBox 448

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\foreground_swin_tiny_bbox448.yaml `
  --device cpu
```

如果 CPU 太慢或内存不足，使用较小 batch：

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\foreground_swin_tiny_bbox448.yaml `
  --device cpu `
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
| Swin Original 448 | 待运行 | 448 | none | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| Swin BBox 448 | 待运行 | 448 | bbox | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |

## 预期解释

如果 `Original 448` 明显优于 `Original 224`，说明单纯提高分辨率能帮助模型看到更多细节。

如果 `BBox 448` 明显优于 `BBox 224`，说明在去除大量背景后，额外像素确实被用于鸟主体细节。

如果 `Original 448` 提升有限，而 `BBox 448` 提升明显，说明像素预算需要先集中到鸟主体上，高分辨率才有稳定收益。

如果 `BBox 448` 分类提升但 retrieval 不提升，说明更高分辨率改善了分类判别，但 embedding 排序结构仍需要 SupCon、ArcFace 或 local-token metric learning 进一步约束。
