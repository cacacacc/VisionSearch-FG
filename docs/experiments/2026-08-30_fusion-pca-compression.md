# 实验 7.3：Fusion PCA 512-D Compression

## 实验动机

Experiment 7.2 中，Original + BBox feature fusion 得到了当前最强 retrieval 结果，但 embedding dimension 从 768-D 增加到 1536-D，存储成本也从 3.516 MiB 增加到 7.031 MiB。该实验进一步验证：能否把 1536-D fusion embedding 压缩到 512-D，同时保留大部分检索性能。

## 研究问题

```text
Fusion embedding 的冗余信息是否足够多，使 PCA 512-D 能保留 retrieval 表现？
```

## 方法

本实验不重新训练模型，也不重新提取 feature。输入固定为 Experiment 7.2 保存的 validation fusion embedding：

```text
outputs/embeddings/fusion_retrieval/20260823_012929_baseline_swin_tiny_protocol__20260830_141148_foreground_swin_tiny_bbox224/concat_l2/cosine/embeddings.npy
```

固定项包括 validation query/gallery、排除 query 自身、cosine similarity、Recall@1 / Recall@5 / Recall@10 / mAP 计算方式。唯一改变项是 embedding dimension：

```text
1536-D original fusion
512-D PCA-compressed fusion
```

PCA 在当前 validation embedding 上拟合，用于开发阶段的压缩对照。正式 test protocol 中应明确 PCA 的拟合集合，优先使用 train/gallery embedding 拟合后再应用到 test query/gallery，避免把 test distribution 信息混入模型选择过程。

## 运行指令

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\compare_embedding_dimensions.py `
  --embeddings outputs\embeddings\fusion_retrieval\20260823_012929_baseline_swin_tiny_protocol__20260830_141148_foreground_swin_tiny_bbox224\concat_l2\cosine\embeddings.npy `
  --records outputs\embeddings\fusion_retrieval\20260823_012929_baseline_swin_tiny_protocol__20260830_141148_foreground_swin_tiny_bbox224\concat_l2\cosine\records.csv `
  --dimensions 1536 512 `
  --output-dir outputs\experiments\fusion_pca_512
```

## 实验结果

| Method | Projection | Dim | Explained Variance | Storage MiB | Query Time ms/query | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fusion original | original | 1536 | 100.00% | 7.031 | 0.0589 | 69.75% | 89.42% | 94.33% | 57.52% |
| Fusion PCA 512-D | PCA | 512 | 96.54% | 2.344 | 0.0518 | 69.83% | 89.50% | 94.33% | 57.72% |

相对原始 1536-D fusion，PCA 512-D 的变化为：

```text
Recall@1  +0.08 percentage points
Recall@5  +0.08 percentage points
Recall@10 +0.00 percentage points
mAP       +0.20 percentage points
Storage   -66.67%
Latency   -12.11%
```

## 结果解释

Fusion PCA 512-D 保留了 96.54% 的 PCA explained variance，并且没有造成 retrieval 指标下降。mAP 从 57.52% 小幅提升到 57.72%，说明原始 1536-D fusion embedding 中存在一定冗余或噪声，PCA 压缩起到了轻微去噪作用。

因此，当前更适合作为默认 retrieval 表示的是 `Fusion PCA 512-D`：它保留了当前最强 retrieval 表现，同时把 float32 embedding storage 从 7.031 MiB 降到 2.344 MiB，成本明显更低。原始 1536-D fusion 可以继续作为上限对照，但不再是最优的效率配置。
