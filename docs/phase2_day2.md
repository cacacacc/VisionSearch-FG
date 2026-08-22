# Phase 2 Day 2：Similarity Metric Comparison

## 目标

本阶段比较 cosine similarity 和 euclidean distance 对同一批视觉 embedding 的检索结果影响。这里不重新训练模型，也不重新学习 representation；唯一变化是 retrieval 阶段的距离度量，因此这是一个关于 retrieval metric 的受控实验，而不是关于 backbone 或 loss 的实验。

## Research Question

```text
不同距离度量对相同视觉 embedding 的 Retrieval 结果有什么影响？
```

## Controlled Variables

固定项包括数据 split、query/gallery 集合、checkpoint、embedding 文件、Recall@K 和 mAP 的计算方式。正式实验直接复用 Phase 2 Day 1 生成的 `embeddings.npy` 和 `records.csv`，这样可以保证 cosine 与 euclidean 使用完全相同的 512-D feature，唯一变化是排序规则：cosine 按相似度从高到低排序，euclidean 按距离从低到高排序。

## Method

推荐使用已经保存好的 CE Retrieval Baseline embedding：

```text
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/embeddings.npy
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/records.csv
```

运行正式对照实验：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_similarity_metrics.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --output-dir outputs\experiments\similarity_metric_comparison\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

输出文件：

```text
outputs/experiments/similarity_metric_comparison/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.json
outputs/experiments/similarity_metric_comparison/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.csv
```

## Metrics

本实验沿用 Phase 2 Day 1 的 retrieval 指标：Recall@1、Recall@5、Recall@10 和 mAP。Recall@K 回答“同类图片是否进入前 K 个候选结果”，mAP 回答“所有同类图片在完整排序中的整体位置是否靠前”。两者结合使用，可以避免只看某一个 top-K 指标造成误判。

## Hypothesis

如果 embedding 已经做了 L2 normalize，cosine similarity 与 euclidean distance 的排序通常会非常接近，因为二者在单位球面上存在单调关系。但如果使用的是未归一化 embedding，euclidean distance 会受到 feature norm 的影响，结果可能与 cosine 明显不同。当前脚本的 cosine 分支会内部 L2 normalize；euclidean 分支默认使用保存的原始 embedding，因此这个实验也能帮助判断 feature norm 是否影响检索排序。

## 本次结果

使用 Phase 2 Day 1 保存的同一批 embedding，得到如下结果：

| Metric | Samples | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosine Similarity | 1,200 | 512 | 58.25% | 80.75% | 87.42% | 47.76% |
| Euclidean Distance | 1,200 | 512 | 54.33% | 77.67% | 85.42% | 43.69% |

差值为 cosine 相对 euclidean 提升：Recall@1 +3.92 个百分点，Recall@5 +3.08 个百分点，Recall@10 +2.00 个百分点，mAP +4.07 个百分点。

## 结论

Cosine similarity 在所有指标上都优于 euclidean distance，说明当前 CE-trained ResNet-18 embedding 更适合用方向相似性进行检索，而原始欧氏距离会受到 feature norm 的额外影响。后续 Phase 2 retrieval baseline、SupCon、Swin 和 CE + SupCon 实验默认使用 cosine similarity；如果未来专门研究 metric learning 或 norm-aware retrieval，再单独设置 euclidean 对照。
