# Similarity Metric Comparison

## Motivation

本实验比较 cosine similarity 和 euclidean distance 对同一批 CE-trained ResNet-18 embedding 的 retrieval 结果影响。实验目的不是提升性能，而是确认 representation 和 retrieval metric 是两个独立变量：同一个视觉特征空间，在不同距离度量下可能得到不同排序。

## Protocol

输入 embedding 固定为 Phase 2 Day 1 生成的 `embeddings.npy`，对应记录固定为同目录下的 `records.csv`。query/gallery 都来自 validation split，共 1,200 张图片；每张图片作为 query 时排除自身，再计算 Recall@1、Recall@5、Recall@10 和 mAP。唯一变化是 metric：cosine similarity 按相似度降序排序，euclidean distance 按距离升序排序。

## Command

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_similarity_metrics.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --output-dir outputs\experiments\similarity_metric_comparison\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

## Result

| Metric | Samples | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosine Similarity | 1,200 | 512 | 58.25% | 80.75% | 87.42% | 47.76% |
| Euclidean Distance | 1,200 | 512 | 54.33% | 77.67% | 85.42% | 43.69% |

## Decision

Cosine similarity 在所有 retrieval 指标上优于 euclidean distance：Recall@1 高 3.92 个百分点，Recall@5 高 3.08 个百分点，Recall@10 高 2.00 个百分点，mAP 高 4.07 个百分点。后续 retrieval 实验默认使用 cosine similarity，并在所有方法之间固定该 metric；euclidean distance 只作为 metric ablation 或补充分析使用。这个结果说明当前 CE feature 的方向信息比原始欧氏距离更稳定，直接用 euclidean distance 会引入 feature norm 的影响。
