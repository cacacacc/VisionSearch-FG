# 实验 11.1：Angular Margin Classification

## 实验动机

当前 retrieval 使用 cosine similarity 和 L2-normalized embedding。普通 CE classifier 的训练目标和最终检索度量并不完全一致；ArcFace / CosFace 则直接在 normalized angular space 中做分类，使训练目标更接近检索时使用的 cosine geometry。

ArcFace 的核心变化是对真实类别加入 angular margin：

```text
cos(theta_y) -> cos(theta_y + m)
```

CosFace 的核心变化是在真实类别 cosine score 上减去 margin：

```text
cos(theta_y) -> cos(theta_y) - m
```

这两者都要求同类样本更靠近自己的 class direction，同时让不同类别之间形成更清晰的 angular separation。

## 研究问题

```text
在相同 Swin-Tiny + BBox crop 输入下，
Angular margin classification 是否比普通 CE 学到更适合 cosine retrieval 的 embedding？
```

## 对照设计

| 组别 | Config | Objective | 输入 | Feature | 目的 |
| --- | --- | --- | --- | --- | --- |
| A | `configs/foreground_swin_tiny_bbox224.yaml` | CE | BBox 224 | embedding | 已有 foreground-aware baseline |
| B | `configs/foreground_swin_tiny_bbox224_arcface.yaml` | ArcFace | BBox 224 | embedding | 验证 angular margin 是否提升 retrieval |
| C | `configs/foreground_swin_tiny_bbox224_arcface_supcon.yaml` | ArcFace + SupCon | BBox 224 | embedding | 验证 class-margin 与 pair-based loss 是否互补 |
| D | `configs/foreground_swin_tiny_bbox224_cosface.yaml` | CosFace | BBox 224 | embedding | 可选，比较 additive cosine margin |

固定变量：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Backbone：ImageNet pretrained Swin-Tiny
- Input：BBox crop 224，margin = 0.15
- Augmentation：Horizontal Flip
- Optimizer：AdamW
- Backbone LR：`5e-5`
- Head LR：`5e-4`
- Epoch：30
- Early stopping：5
- Retrieval metric：cosine

主要变化：

```text
CE classifier -> ArcFace / CosFace classifier
```

## 实现说明

本实验新增 `AngularMarginHead` 和 `train_margin_classifier.py`。

训练阶段：

```text
embedding -> L2 normalize
class weight -> L2 normalize
cosine logits
真实类别加入 margin
scale 后计算 CrossEntropy
```

Validation 阶段不加入 margin，只使用 scaled cosine logits 做分类评估。Retrieval 阶段直接抽取 backbone embedding，并使用 cosine similarity。

## 训练指令

### B 组：ArcFace

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_margin_classifier.py `
  --config configs\foreground_swin_tiny_bbox224_arcface.yaml `
  --device cuda
```

### C 组：ArcFace + SupCon

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_margin_classifier.py `
  --config configs\foreground_swin_tiny_bbox224_arcface_supcon.yaml `
  --device cuda
```

### D 组：CosFace，可选

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_margin_classifier.py `
  --config configs\foreground_swin_tiny_bbox224_cosface.yaml `
  --device cuda
```

## 检索评估指令

### ArcFace retrieval

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\foreground_swin_tiny_bbox224_arcface | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\foreground_swin_tiny_bbox224_arcface.yaml `
  --checkpoint "outputs\checkpoints\foreground_swin_tiny_bbox224_arcface\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\angular_margin
```

### ArcFace + SupCon retrieval

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\foreground_swin_tiny_bbox224_arcface_supcon | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\foreground_swin_tiny_bbox224_arcface_supcon.yaml `
  --checkpoint "outputs\checkpoints\foreground_swin_tiny_bbox224_arcface_supcon\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\angular_margin
```

### CosFace retrieval，可选

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\foreground_swin_tiny_bbox224_cosface | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\foreground_swin_tiny_bbox224_cosface.yaml `
  --checkpoint "outputs\checkpoints\foreground_swin_tiny_bbox224_cosface\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\angular_margin
```

如果 CUDA 评估不稳定，把 retrieval 指令中的 `--device cuda` 改成 `--device cpu`。

## 结果记录表

| 方法 | Run ID | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE BBox 224 | `20260830_141148_foreground_swin_tiny_bbox224` | 80.00% | 79.52% | 94.92% | 68.42% | 86.50% | 93.08% | 55.51% |
| ArcFace BBox 224 | `20260904_105853_foreground_swin_tiny_bbox224_arcface` | 84.92% | 84.50% | 95.17% | 82.33% | 91.58% | 95.17% | 77.30% |
| ArcFace + SupCon BBox 224 | `20260905_094114_foreground_swin_tiny_bbox224_arcface_supcon` | 84.50% | 84.29% | 94.67% | 79.58% | 91.33% | 95.92% | 73.20% |
| CosFace BBox 224 | `20260905_100822_foreground_swin_tiny_bbox224_cosface` | 84.33% | 83.98% | 93.92% | 80.25% | 91.08% | 94.67% | 74.46% |

## 训练记录

| 方法 | Best Epoch | 实际训练 Epoch | 训练时间 | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| ArcFace BBox 224 | 13 | 18 | 15.40 min | 27.67M |
| ArcFace + SupCon BBox 224 | 7 | 9 | 10.39 min | 27.67M |
| CosFace BBox 224 | 14 | 19 | 16.17 min | 27.67M |

## 结果分析

ArcFace 是本实验最强方法。与普通 CE BBox 224 相比，ArcFace 将 Val Acc 从 `80.00%` 提升到 `84.92%`，Recall@1 从 `68.42%` 提升到 `82.33%`，mAP 从 `55.51%` 提升到 `77.30%`。这说明让训练目标直接匹配 cosine angular geometry 对当前 retrieval representation 非常有效。

CosFace 也明显优于普通 CE，但低于 ArcFace。CosFace 的 mAP 为 `74.46%`，说明 additive cosine margin 能改善 embedding space，但在当前配置下不如 ArcFace 的 angular margin。

ArcFace + SupCon 没有超过单独 ArcFace。它的分类指标接近 ArcFace，但 Recall@1 和 mAP 明显下降，说明当前 SupCon 权重、temperature、two-view augmentation 或 batch composition 与 angular margin 存在冲突。这个结果不能解释为 SupCon 无效，而应解释为：class-margin-based 约束和 pair-based 约束不能直接无调参叠加。

## 结论

本实验建立了一个非常重要的结果：在 Swin-Tiny + BBox 224 条件下，Angular Margin Classification 比普通 CE 更适合学习 cosine retrieval embedding。ArcFace 甚至在 224 输入分辨率下超过了此前 ConvNeXt V2 Tiny BBox 448 + TTA + Query Expansion 的 mAP，因此它应成为后续 metric learning 方向的核心 baseline。

下一步最值得做的是：

1. 对 ArcFace checkpoint 做 Flip TTA 和 Query Expansion，验证后处理是否还能继续提升 mAP。
2. 将 ArcFace 从 Swin-Tiny BBox 224 迁移到 ConvNeXt V2 Tiny BBox 448，验证强 backbone + angular margin 是否叠加。
3. 若继续加入 SupCon，需要先使用 P×K sampler 或 hard negative mining，而不是沿用当前 random batch。
