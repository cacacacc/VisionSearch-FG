# 实验 7.2：Original + BBox Feature Fusion Retrieval

## 实验动机

Experiment 7.1 证明，使用 CUB bounding box 裁剪鸟主体后，Swin-Tiny 的 classification 和 retrieval 都明显提升。但单独使用 BBox crop 可能丢失原图中的整体姿态、轮廓和少量有用上下文。因此本实验进一步验证：原图 feature 和 BBox crop feature 是否互补，融合后能否继续提升 retrieval 排序质量。

## 研究问题

```text
Original image embedding 与 BBox crop embedding 是否包含互补信息？
```

## 方法

本实验不重新训练模型，只使用两个已经训练好的 checkpoint 提取 validation split embedding：

| 输入分支 | Config | Checkpoint | Feature |
| --- | --- | --- | --- |
| Original branch | `configs/baseline_swin_tiny_protocol.yaml` | `outputs/checkpoints/baseline_swin_tiny_protocol/20260823_012929_baseline_swin_tiny_protocol/best.pt` | Swin backbone embedding, 768-D |
| BBox branch | `configs/foreground_swin_tiny_bbox224.yaml` | `outputs/checkpoints/foreground_swin_tiny_bbox224/20260830_141148_foreground_swin_tiny_bbox224/best.pt` | Swin backbone embedding, 768-D |

融合方式：

```text
h_original = L2Normalize(original_embedding)
h_bbox     = L2Normalize(bbox_embedding)
h_fusion   = L2Normalize(concat(h_original, h_bbox))
```

最终 fusion embedding 为 1536-D。Retrieval protocol 与前面实验一致：validation split 同时作为 query 和 gallery，排除 query 自身，使用 cosine similarity，报告 Recall@1 / Recall@5 / Recall@10 / mAP。

## 运行指令

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\evaluate_fusion_retrieval.py `
  --primary-config configs\baseline_swin_tiny_protocol.yaml `
  --primary-checkpoint outputs\checkpoints\baseline_swin_tiny_protocol\20260823_012929_baseline_swin_tiny_protocol\best.pt `
  --secondary-config configs\foreground_swin_tiny_bbox224.yaml `
  --secondary-checkpoint outputs\checkpoints\foreground_swin_tiny_bbox224\20260830_141148_foreground_swin_tiny_bbox224\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --fusion concat_l2 `
  --device cpu `
  --output-dir outputs\embeddings\fusion_retrieval
```

## 最终结果

| Method | Input | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP | Query Time ms/query | Storage MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny Original | 原图 224 | 768 | 61.25% | 84.33% | 90.67% | 49.04% | 0.0533 | 3.516 |
| Swin-Tiny BBox Crop | BBox crop 224 | 768 | 68.42% | 86.50% | 93.08% | 55.51% | 0.0879 | 3.516 |
| Original + BBox concat | 原图 224 + BBox crop 224 | 1536 | 69.75% | 89.42% | 94.33% | 57.52% | 0.0774 | 7.031 |
| Original + BBox concat PCA | 原图 224 + BBox crop 224 | 512 | 69.83% | 89.50% | 94.33% | 57.72% | 0.0518 | 2.344 |

相对 BBox crop 单分支，feature fusion 的提升为：

```text
Recall@1  +1.33 percentage points
Recall@5  +2.92 percentage points
Recall@10 +1.25 percentage points
mAP       +2.01 percentage points
```

相对原图 Swin-Tiny baseline，feature fusion 的提升为：

```text
Recall@1  +8.50 percentage points
Recall@5  +5.09 percentage points
Recall@10 +3.66 percentage points
mAP       +8.48 percentage points
```

## 结果解释

Original + BBox feature fusion 在所有 retrieval 指标上都优于单独的原图分支和单独的 BBox crop 分支，说明原图全局上下文与 bbox 前景细节确实存在互补。BBox crop 是主要提升来源，因为它已经把 mAP 从 49.04% 提升到 55.51%；fusion 进一步把 mAP 提升到 57.52%，说明原图中的姿态、轮廓或少量上下文仍然能帮助排序。

原始 fusion 的代价是存储上升：fusion embedding 从 768-D 增加到 1536-D，float32 storage 从 3.516 MiB 增加到 7.031 MiB。Experiment 7.3 进一步证明，将 fusion embedding PCA 压缩到 512-D 后，mAP 从 57.52% 小幅提升到 57.72%，storage 降到 2.344 MiB。因此当前默认 retrieval 表示应优先采用 `Original + BBox concat PCA 512-D`，原始 1536-D fusion 保留为未压缩上限对照。

## 当前结论

当前 retrieval 最强配置是：

```text
Swin-Tiny Original + BBox Crop concat_l2
PCA 512-D
Recall@1 = 69.83%
Recall@5 = 89.50%
Recall@10 = 94.33%
mAP = 57.72%
```

下一步优先级建议是：继续做 `BBox crop 384` 或 `BBox crop + ArcFace/SupCon`，分别验证更高分辨率和更强 metric learning 是否还能继续提高数值。
