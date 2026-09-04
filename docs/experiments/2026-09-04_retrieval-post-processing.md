# 实验 13：Retrieval Post-processing

## 实验动机

本阶段不再改变 encoder，也不重新训练 backbone，而是在已有 embedding 的基础上优化 retrieval ranking。

这类方法必须和 representation improvement 分开报告，因为它回答的是另一个问题：

```text
已经有 embedding 以后，怎样通过后处理把排序排得更好？
```

当前主模型固定为：

```text
ConvNeXt V2 Tiny + BBox crop + 448 input + CE
Run ID: 20260831_113000_backbone_convnextv2_tiny_bbox448
```

当前 baseline：

```text
Accuracy: 86.42%
Recall@1: 82.75%
Recall@5: 93.75%
Recall@10: 96.92%
mAP: 75.06%
```

## 研究问题

```text
不改变模型参数时，
Multi-view feature averaging 和 Query Expansion 能否提升 fine-grained retrieval ranking？
```

## 实验设计

### 13.1 Multi-view / TTA Feature Averaging

比较：

```text
BBox only
BBox + horizontal flip
Original + BBox
Original + Original flip + BBox + BBox flip
```

每个 view 先提取 embedding 并 L2 normalize，再做平均：

```text
f = Normalize(mean(Normalize(f_i)))
```

### 13.2 Query Expansion

第一次检索得到 top-k neighbors 后，对 query 进行扩展：

```text
q' = Normalize(q + alpha * mean(top-k neighbor features))
```

测试：

```text
top-k = 3, 5, 10
alpha = 0.1, 0.2, 0.5
```

风险：

```text
如果第一次检索结果错误，query expansion 会造成 query drift。
```

### 13.3 Re-ranking

本阶段先不实现 local descriptor re-ranking。原因是 list-wise local re-ranking 需要可靠的 local descriptors 或 part features。当前项目已有的证据表明，local feature 有价值，但 hard crop 容易失败，因此 re-ranking 应该作为下一阶段的高级模块，而不是和 TTA/QE 混在一起。

## 评估指令

固定 checkpoint：

```powershell
cd D:\code\VisionSearch-FG

$runBest = "20260831_113000_backbone_convnextv2_tiny_bbox448"
$checkpointBest = "outputs\checkpoints\backbone_convnextv2_tiny_bbox448\$runBest\best.pt"
```

### 1. BBox Only Baseline Sanity Check

该结果应该接近已知 baseline `mAP 75.06%`。

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint $checkpointBest `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views bbox `
  --qe-top-k 0 `
  --qe-alpha 0 `
  --device cuda `
  --output-dir outputs\embeddings\phase13_retrieval_postprocessing
```

### 2. BBox + Flip TTA

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint $checkpointBest `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views bbox bbox_flip `
  --qe-top-k 0 `
  --qe-alpha 0 `
  --device cuda `
  --output-dir outputs\embeddings\phase13_retrieval_postprocessing
```

### 3. Original + BBox Feature Averaging

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint $checkpointBest `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views original bbox `
  --qe-top-k 0 `
  --qe-alpha 0 `
  --device cuda `
  --output-dir outputs\embeddings\phase13_retrieval_postprocessing
```

### 4. Original + BBox + Flip Feature Averaging

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint $checkpointBest `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views original original_flip bbox bbox_flip `
  --qe-top-k 0 `
  --qe-alpha 0 `
  --device cuda `
  --output-dir outputs\embeddings\phase13_retrieval_postprocessing
```

### 5. Query Expansion Grid on BBox Only

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint $checkpointBest `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views bbox `
  --qe-top-k 3 5 10 `
  --qe-alpha 0.1 0.2 0.5 `
  --device cuda `
  --output-dir outputs\embeddings\phase13_retrieval_postprocessing
```

### 6. Query Expansion Grid on Best Multi-view

如果第 2-4 组中某个 multi-view 的 mAP 高于 BBox only，则继续在最佳 view 上跑 Query Expansion。下面以 `original original_flip bbox bbox_flip` 为例：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --checkpoint $checkpointBest `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views original original_flip bbox bbox_flip `
  --qe-top-k 3 5 10 `
  --qe-alpha 0.1 0.2 0.5 `
  --device cuda `
  --output-dir outputs\embeddings\phase13_retrieval_postprocessing
```

如果 CUDA 评估报错，把所有指令里的：

```powershell
--device cuda
```

改成：

```powershell
--device cpu
```

## 最终实验表

| Method | Views | QE top-k | QE alpha | Recall@1 | Recall@5 | Recall@10 | mAP | Storage | Search Latency | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | bbox | 0 | 0 | 82.75% | 93.75% | 96.92% | 75.06% | 3.52 MiB | 0.070 ms/query | 已知 ConvNeXt baseline |
| TTA | bbox + bbox_flip | 0 | 0 | 82.67% | 94.42% | 97.25% | 75.74% | 3.52 MiB | 0.056 ms/query | TTA 有效，接近 QE 最优 |
| Feature averaging | original + bbox | 0 | 0 | 82.00% | 93.92% | 97.33% | 74.73% | 3.52 MiB | 0.057 ms/query | Recall@10 提升，但 mAP 下降 |
| Feature averaging | original + original_flip + bbox + bbox_flip | 0 | 0 | 82.42% | 94.58% | 97.08% | 75.38% | 3.52 MiB | 0.056 ms/query | 高于 baseline，但不如 BBox TTA |
| Query Expansion | bbox | 3 | 0.1 | 82.58% | 93.92% | 96.67% | 75.39% | 3.52 MiB | 0.060 ms/query | 小幅提升 mAP |
| Query Expansion | bbox | 3 | 0.2 | 82.92% | 93.67% | 96.42% | 75.60% | 3.52 MiB | 0.059 ms/query | Recall@1 提升 |
| Query Expansion | bbox | 3 | 0.5 | 82.67% | 93.25% | 96.17% | 75.82% | 3.52 MiB | 0.056 ms/query | 单独 QE 最佳 mAP |
| Query Expansion | bbox | 5 | 0.1 | 82.58% | 93.75% | 96.83% | 75.26% | 3.52 MiB | 0.057 ms/query | 小幅提升 |
| Query Expansion | bbox | 5 | 0.2 | 82.58% | 93.67% | 96.67% | 75.33% | 3.52 MiB | 0.056 ms/query | 小幅提升 |
| Query Expansion | bbox | 5 | 0.5 | 82.25% | 93.67% | 96.25% | 75.37% | 3.52 MiB | 0.057 ms/query | 小幅提升 |
| Query Expansion | bbox | 10 | 0.1 | 82.42% | 93.67% | 96.92% | 74.95% | 3.52 MiB | 0.055 ms/query | 基本持平 |
| Query Expansion | bbox | 10 | 0.2 | 82.08% | 93.58% | 96.75% | 74.74% | 3.52 MiB | 0.056 ms/query | 低于 baseline |
| Query Expansion | bbox | 10 | 0.5 | 80.92% | 93.75% | 96.75% | 73.90% | 3.52 MiB | 0.056 ms/query | query drift 明显 |
| TTA + Query Expansion | bbox + bbox_flip | 3 | 0.1 | 82.83% | 94.08% | 97.00% | 76.19% | 3.52 MiB | 0.057 ms/query | 明显优于单独 TTA |
| TTA + Query Expansion | bbox + bbox_flip | 3 | 0.2 | 83.17% | 94.00% | 96.75% | 76.45% | 3.52 MiB | 0.056 ms/query | 当前最佳 Recall@1 |
| TTA + Query Expansion | bbox + bbox_flip | 3 | 0.5 | 82.83% | 93.83% | 96.58% | 76.82% | 3.52 MiB | 0.057 ms/query | 当前最佳 mAP |
| TTA + Query Expansion | bbox + bbox_flip | 5 | 0.1 | 82.92% | 94.42% | 97.00% | 75.97% | 3.52 MiB | 0.057 ms/query | 稳定提升 |
| TTA + Query Expansion | bbox + bbox_flip | 5 | 0.2 | 82.58% | 94.42% | 96.83% | 76.10% | 3.52 MiB | 0.057 ms/query | 稳定提升 |
| TTA + Query Expansion | bbox + bbox_flip | 5 | 0.5 | 82.92% | 93.75% | 96.67% | 76.21% | 3.52 MiB | 0.055 ms/query | 稳定提升 |
| TTA + Query Expansion | bbox + bbox_flip | 10 | 0.1 | 82.67% | 94.33% | 97.08% | 75.60% | 3.52 MiB | 0.056 ms/query | 高于 baseline |
| TTA + Query Expansion | bbox + bbox_flip | 10 | 0.2 | 82.08% | 94.33% | 97.00% | 75.43% | 3.52 MiB | 0.056 ms/query | 高于 baseline |
| TTA + Query Expansion | bbox + bbox_flip | 10 | 0.5 | 81.67% | 93.75% | 96.83% | 74.68% | 3.52 MiB | 0.056 ms/query | query drift 明显 |

## 当前输出

已生成：

```text
outputs/embeddings/phase13_retrieval_postprocessing/20260831_113000_backbone_convnextv2_tiny_bbox448/bbox/embedding/summary.json
outputs/embeddings/phase13_retrieval_postprocessing/20260831_113000_backbone_convnextv2_tiny_bbox448/original_bbox/embedding/summary.json
outputs/embeddings/phase13_retrieval_postprocessing/20260831_113000_backbone_convnextv2_tiny_bbox448/bbox_bbox_flip/embedding/summary.json
outputs/embeddings/phase13_retrieval_postprocessing/20260831_113000_backbone_convnextv2_tiny_bbox448/original_original_flip_bbox_bbox_flip/embedding/summary.json
```

## 当前分析

当前最好的 post-processing 结果来自 Query Expansion：

```text
Baseline:                  Recall@1 = 82.75% / Recall@5 = 93.75% / Recall@10 = 96.92% / mAP = 75.06%
BBox + Flip TTA:           Recall@1 = 82.67% / Recall@5 = 94.42% / Recall@10 = 97.25% / mAP = 75.74%
QE top3 alpha0.5:          Recall@1 = 82.67% / Recall@5 = 93.25% / Recall@10 = 96.17% / mAP = 75.82%
BBox + Flip TTA + QE top3 alpha0.5: Recall@1 = 82.83% / Recall@5 = 93.83% / Recall@10 = 96.58% / mAP = 76.82%
```

TTA + Query Expansion 将 mAP 从 `75.06%` 提升到 `76.82%`，提升 `+1.76 percentage points`。这说明 ConvNeXt V2 Tiny 的 top-3 retrieval neighbors 已经有一定可信度，并且 BBox + Flip TTA 先降低了单视角 embedding 噪声，使 Query Expansion 更可靠。

单独 Query Expansion 的提升主要体现在 mAP，而不是 Recall@K：

```text
Recall@1: 82.75% -> 82.67% (-0.08 percentage points)
Recall@5: 93.75% -> 93.25% (-0.50 percentage points)
Recall@10: 96.92% -> 96.17% (-0.75 percentage points)
mAP: 75.06% -> 75.82% (+0.76 percentage points)
```

这意味着 Query Expansion 改善了完整排序质量，但没有提升 Top-K 是否命中同类的概率。因此它适合作为 mAP-oriented post-processing，而不是 Recall@K-oriented post-processing。

TTA + Query Expansion 则同时提升了 Recall@1 和 mAP：

```text
Recall@1: 82.75% -> 82.83% (+0.08 percentage points)
Recall@5: 93.75% -> 93.83% (+0.08 percentage points)
Recall@10: 96.92% -> 96.58% (-0.34 percentage points)
mAP: 75.06% -> 76.82% (+1.76 percentage points)
```

如果只看 Recall@1，最佳组合是 `BBox + Flip TTA + QE top3 alpha0.2`：

```text
Recall@1 = 83.17%
mAP = 76.45%
```

如果以 mAP 作为主指标，最佳组合是 `BBox + Flip TTA + QE top3 alpha0.5`：

```text
Recall@1 = 82.83%
mAP = 76.82%
```

BBox + Flip TTA 也带来了稳定收益：

```text
Recall@1: 82.75% -> 82.67% (-0.08 percentage points)
Recall@5: 93.75% -> 94.42% (+0.67 percentage points)
Recall@10: 96.92% -> 97.25% (+0.33 percentage points)
mAP: 75.06% -> 75.74% (+0.68 percentage points)
```

这说明水平翻转 averaging 可以降低单一视角带来的 embedding 偶然性。和 Query Expansion 不同，BBox + Flip TTA 同时提升了 Recall@5、Recall@10 和 mAP，因此它更适合作为稳定的默认 post-processing。

Query Expansion 的风险也已经出现：当 top-k 过大、alpha 过大时，结果明显下降。

```text
BBox QE top3 alpha0.5:             mAP = 75.82%
BBox QE top10 alpha0.5:            mAP = 73.90%
BBox + Flip QE top3 alpha0.5:      mAP = 76.82%
BBox + Flip QE top10 alpha0.5:     mAP = 74.68%
```

这就是典型 query drift：引入太多邻居后，错误或弱相关样本会把 query embedding 拉向不稳定区域。

Original + BBox feature averaging 没有提升 mAP：

```text
Baseline bbox:                         mAP = 75.06%
Original + BBox average:               mAP = 74.73%
Original + Original flip + BBox + Flip: mAP = 75.38%
```

但 Original view 相关方法提升了 Recall@10 或 Recall@5：

```text
Original + BBox Recall@10: 96.92% -> 97.33%
Four-view Recall@5:        93.75% -> 94.58%
Four-view Recall@10:       96.92% -> 97.08%
```

这说明 original view 确实提供了一些全局上下文，但也引入背景噪声。当前 ConvNeXt V2 Tiny 的 BBox embedding 已经足够强，简单 averaging original view 不如 foreground-focused BBox TTA 稳定。

## 阶段结论

当前 Phase 13 的最优结果是：

```text
ConvNeXt V2 Tiny BBox 448 + Flip TTA + Query Expansion top3 alpha0.5
mAP = 76.82%
```

相比不做 post-processing 的 baseline：

```text
mAP: 75.06% -> 76.82% (+1.76 percentage points)
```

这是一个有效但温和的 post-processing 提升。它的科研意义是：后处理确实能改善 embedding ranking，但提升幅度小于更换 backbone、提高分辨率和 foreground-aware input。因此后处理应该作为最终系统优化，而不是主要 representation improvement。

当前建议：

```text
默认 retrieval backbone：ConvNeXt V2 Tiny BBox 448
默认稳定 post-processing：BBox + Flip TTA
默认 mAP-oriented post-processing：BBox + Flip TTA + Query Expansion top3 alpha0.5
默认 Recall@1-oriented post-processing：BBox + Flip TTA + Query Expansion top3 alpha0.2
暂不采用：Original + BBox average
谨慎采用：四视图 averaging
```
