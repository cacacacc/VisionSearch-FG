# 最终 Test Evaluation

## 实验目的

本实验只用于最终报告，不再参与调参。

前面所有 ablation 和模型选择都基于 CUB-200-2011 official train 内部划分出的 validation split。当前实验使用 official test split：

```text
data/processed/splits/cub_test_ids.txt
```

用于评估最终 pipeline 的泛化性能。

## 最终模型

```text
ConvNeXt V2 Tiny BBox448
+ ArcFace margin=0.5, scale=16
+ BBox + BBox Flip feature averaging
+ Query Expansion
```

Checkpoint：

```text
outputs/checkpoints/foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16/20260905_141401_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16/best.pt
```

Config：

```text
configs/foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml
```

## Evaluation Protocol

- Dataset：CUB-200-2011 official test split
- Number of samples：5,794
- Query/Gallery：test split 内部 query-gallery retrieval
- Self-match：排除 query 自身
- Feature：backbone embedding
- Metric：cosine similarity
- Views：`bbox bbox_flip`
- Embedding dim：768
- Storage：16.97 MiB，float32

## 最终 Test 结果

| 方法 | QE Top-K | Alpha | Recall@1 | Recall@5 | Recall@10 | mAP | Query Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flip TTA | 0 | 0.0 | 88.16% | 94.24% | 95.51% | 83.06% | 0.3057 ms/query |
| Flip TTA + QE | 3 | 0.2 | 88.25% | 93.75% | 95.15% | 83.39% | 0.3057 ms/query |
| Flip TTA + QE | 3 | 0.5 | 88.30% | 93.63% | 94.99% | 83.59% | 0.3046 ms/query |

## Validation 与 Test 对比

| Split | 方法 | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Flip TTA | 87.33% | 94.25% | 96.92% | 83.79% |
| Validation | Flip TTA + QE top3 alpha0.5 | 87.50% | 93.33% | 96.67% | 84.21% |
| Test | Flip TTA | 88.16% | 94.24% | 95.51% | 83.06% |
| Test | Flip TTA + QE top3 alpha0.5 | 88.30% | 93.63% | 94.99% | 83.59% |

## 结果分析

最终 test mAP 为 `83.59%`，略低于 validation 上的 `84.21%`，属于合理泛化差距。与此同时，test Recall@1 达到 `88.30%`，高于 validation 的 `87.50%`，说明最终模型在 Top-1 retrieval 上没有过拟合 validation split。

Query Expansion 在 test 上仍然有效。与 Flip TTA 相比，QE top3 alpha0.5 将 mAP 从 `83.06%` 提升到 `83.59%`，Recall@1 从 `88.16%` 提升到 `88.30%`。但它也降低了 Recall@5 和 Recall@10，这与 validation 阶段观察一致：QE 更擅长改善整体排序质量，但可能带来轻微 query drift。

## 最终结论

最终推荐报告结果为：

```text
ConvNeXt V2 Tiny BBox448
+ ArcFace margin=0.5, scale=16
+ BBox + BBox Flip feature averaging
+ QE top3 alpha0.5
```

Official test retrieval performance：

```text
Recall@1 = 88.30%
Recall@5 = 93.63%
Recall@10 = 94.99%
mAP = 83.59%
```

如果希望报告更保守、不引入 Query Expansion 的结果，可以同时报告 Flip TTA-only：

```text
Recall@1 = 88.16%
Recall@5 = 94.24%
Recall@10 = 95.51%
mAP = 83.06%
```
