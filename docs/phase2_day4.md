# Phase 2 Day 4：Embedding Dimension

## 目标

本阶段比较 512、256、128、64 维 embedding 的 retrieval 性能、存储成本和搜索延迟。这里不重新训练模型，而是复用 Phase 2 Day 1 保存的 CE-trained ResNet-18 512-D embedding；512 维作为原始基线，256/128/64 维通过 PCA 做无监督压缩，然后统一使用 L2-normalized cosine retrieval。

## Research Question

```text
Embedding dimension 与信息表达能力、检索性能和计算成本之间有什么 trade-off？
```

## Controlled Variables

固定项包括 checkpoint、原始 embedding 文件、validation split、query/gallery 集合、排除 query 自身的协议、L2 normalization、cosine similarity、Recall@K 和 mAP 的计算方式。唯一主要变化是 embedding dimension。降维方法固定为 PCA，避免“直接截断前 N 维”这种没有特征重要性依据的做法。

## Method

输入文件沿用 CE Retrieval Baseline：

```text
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/embeddings.npy
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/records.csv
```

运行正式对照实验：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_embedding_dimensions.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --dimensions 512 256 128 64 `
  --latency-repeats 5 `
  --output-dir outputs\experiments\embedding_dimension\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

输出文件：

```text
outputs/experiments/embedding_dimension/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.json
outputs/experiments/embedding_dimension/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.csv
```

## Metrics

本实验记录 Recall@1、Recall@5、Recall@10、mAP、float32 embedding storage、平均总搜索延迟和平均单 query 搜索延迟。搜索延迟在当前脚本中包含全量 query-gallery similarity matrix 计算和排序，不包含模型 forward、不包含图片读取，也不包含指标统计。

## 本次结果

| Dim | Projection | Explained Var | Storage MiB | Latency / Query ms | Recall@1 | Recall@5 | Recall@10 | mAP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | original | 100.00% | 2.344 | 0.0569 | 58.25% | 80.75% | 87.42% | 47.76% |
| 256 | PCA | 97.19% | 1.172 | 0.0504 | 57.50% | 80.25% | 88.00% | 47.30% |
| 128 | PCA | 91.25% | 0.586 | 0.0506 | 57.50% | 80.67% | 87.83% | 47.59% |
| 64 | PCA | 80.97% | 0.293 | 0.0504 | 56.67% | 80.42% | 88.58% | 46.78% |

相对 512 维，128 维存储量下降 75.00%，Recall@1 只下降 0.75 个百分点，mAP 只下降 0.17 个百分点，Recall@10 还略高 0.42 个百分点。64 维虽然 Recall@10 最高，但 Recall@1 和 mAP 下降更明显，说明前排排序质量开始变差。当前 validation 只有 1,200 张图片，延迟主要受全量排序开销影响，低维带来的 search latency 收益不明显；在更大 gallery 上，维度降低对 similarity 计算成本的影响会更值得关注。

## 结论

128 维是当前最合理的轻量候选：它保留了 91.25% PCA explained variance，并在几乎不损失 mAP 的情况下把 embedding storage 降到 512 维的 25%。如果目标是科研主结果和最高可比性，继续报告 512 维；如果目标是轻量检索系统或后续部署分析，可以把 128 维作为推荐压缩配置。64 维暂时不作为默认配置，因为 mAP 和 Recall@1 损失已经更明显。
