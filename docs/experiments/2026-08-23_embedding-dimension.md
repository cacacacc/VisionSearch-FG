# Embedding Dimension

## Motivation

本实验评估 embedding dimension 对 retrieval 性能、存储成本和搜索延迟的影响。目标是理解信息表达能力与计算成本之间的 trade-off：更高维度通常保留更多视觉信息，但也带来更高存储和检索成本；更低维度更轻量，但可能损失细粒度鸟类识别所需的信息。

## Protocol

输入 embedding 固定为 Phase 2 Day 1 生成的 `embeddings.npy`，记录文件固定为同目录下的 `records.csv`。512 维使用原始 CE feature；256、128、64 维使用 PCA 从同一批 512-D feature 压缩得到。所有维度统一使用 L2-normalized cosine retrieval，query/gallery 都来自 validation split，共 1,200 张图片，并在检索时排除 query 自身。

## Command

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_embedding_dimensions.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --dimensions 512 256 128 64 `
  --latency-repeats 5 `
  --output-dir outputs\experiments\embedding_dimension\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

## Result

| Dim | Projection | Explained Var | Storage MiB | Latency / Query ms | Recall@1 | Recall@5 | Recall@10 | mAP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | original | 100.00% | 2.344 | 0.0569 | 58.25% | 80.75% | 87.42% | 47.76% |
| 256 | PCA | 97.19% | 1.172 | 0.0504 | 57.50% | 80.25% | 88.00% | 47.30% |
| 128 | PCA | 91.25% | 0.586 | 0.0506 | 57.50% | 80.67% | 87.83% | 47.59% |
| 64 | PCA | 80.97% | 0.293 | 0.0504 | 56.67% | 80.42% | 88.58% | 46.78% |

## Decision

512 维仍作为科研主结果的默认报告维度，因为它是原始 CE feature，不引入额外 PCA 后处理。128 维作为当前最合理的轻量候选：相对 512 维，storage 从 2.344 MiB 降到 0.586 MiB，下降 75.00%；mAP 仅下降 0.17 个百分点，Recall@1 下降 0.75 个百分点。64 维不作为默认配置，因为 mAP 下降 0.98 个百分点且 Recall@1 下降 1.58 个百分点，说明前排排序质量已经开始受损。当前 validation gallery 只有 1,200 张，latency 数值主要反映矩阵计算和排序的综合开销，不能直接等价于大规模部署延迟。
