# Embedding Normalization

## Motivation

本实验比较 Raw Embedding 和 L2-normalized Embedding 对 CE-trained ResNet-18 retrieval 的影响。实验目标是理解 embedding geometry：检索排序到底主要依赖 feature direction，还是也依赖 feature magnitude。

## Protocol

输入 embedding 固定为 Phase 2 Day 1 生成的 `embeddings.npy`，记录文件固定为同目录下的 `records.csv`。query/gallery 都来自 validation split，共 1,200 张图片；每张图片作为 query 时排除自身，再计算 Recall@1、Recall@5、Recall@10 和 mAP。Raw Embedding 使用 dot-product similarity，保留 magnitude；L2-normalized Embedding 使用 normalized dot-product similarity，等价于 cosine similarity。

## Command

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_embedding_normalization.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --output-dir outputs\experiments\embedding_normalization\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

## Result

| Variant | Samples | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Dot Product | 1,200 | 512 | 46.17% | 73.00% | 81.58% | 33.61% |
| L2-normalized Cosine | 1,200 | 512 | 58.25% | 80.75% | 87.42% | 47.76% |

## Decision

L2-normalized Cosine 在所有指标上显著优于 Raw Dot Product：Recall@1 高 12.08 个百分点，Recall@5 高 7.75 个百分点，Recall@10 高 5.83 个百分点，mAP 高 14.15 个百分点。后续 retrieval 实验固定使用 L2-normalized embedding 与 cosine similarity。Raw feature magnitude 在当前 CE-trained ResNet-18 中会干扰检索排序，不应作为默认相似度的一部分。
