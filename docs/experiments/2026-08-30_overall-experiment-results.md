# 项目总实验结果表

本文档汇总 VisionSearch-FG 当前已经完成的主要实验结果。除特别说明外，结果均来自 CUB-200-2011 official train 内部划分出的 validation split，共 1,200 张图；retrieval protocol 为 query/gallery 同用 validation split，并排除 query 自身。表中的 `BBox crop` 使用了 CUB 官方 bounding box，属于 annotation-assisted setting，不能与原图输入 baseline 解释为完全相同输入条件。

## 主结果总表

| 方法 | Backbone | 输入 | Objective | Feature | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 说明 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-18 Full FT + HFlip | ResNet-18 | 原图 224 | CE | backbone h | 69.67% | 未记录 | 89.33% | 58.25% | 80.75% | 87.42% | 47.76% | 主 CNN baseline |
| Swin-Tiny Full FT + HFlip | Swin-Tiny | 原图 224 | CE | backbone h | 75.50% | 75.28% | 93.42% | 61.25% | 84.33% | 90.67% | 49.04% | 主 Transformer baseline |
| Swin-Tiny 512-D PCA | Swin-Tiny | 原图 224 | CE | PCA 512-D | 75.50% | 75.28% | 93.42% | 61.67% | 84.75% | 90.50% | 49.22% | 同一个 Swin checkpoint 的 PCA 压缩检索结果 |
| Swin-Tiny BBox Crop 224 | Swin-Tiny | BBox crop 224, margin=0.15 | CE | backbone h | 80.00% | 79.52% | 94.92% | 68.42% | 86.50% | 93.08% | 55.51% | 使用 CUB bbox，当前最强结果 |
| Swin Original + BBox concat | Swin-Tiny | 原图 224 + BBox crop 224 | CE | concat L2 backbone h | 80.00% | 79.52% | 94.92% | 69.75% | 89.42% | 94.33% | 57.52% | 未压缩 fusion retrieval 对照；分类指标沿用 BBox crop 模型 |
| Swin Original + BBox concat PCA | Swin-Tiny | 原图 224 + BBox crop 224 | CE | PCA 512-D | 80.00% | 79.52% | 94.92% | 69.83% | 89.50% | 94.33% | 57.72% | 当前最强 retrieval，且存储低于 1536-D fusion |

当前最高分类结果是 `Swin-Tiny BBox Crop 224`，Val Acc 为 80.00%。当前最高检索结果是 `Swin Original + BBox concat PCA`，mAP 为 57.72%，Recall@1 为 69.83%。相对原图 Swin-Tiny baseline，BBox crop 的 Val Acc 从 75.50% 提升到 80.00%，提升 4.50 个百分点；feature fusion PCA 的 mAP 从 49.04% 提升到 57.72%，提升 8.68 个百分点，Recall@1 从 61.25% 提升到 69.83%，提升 8.58 个百分点。这个结果和 Phase 6 的解释性分析一致：背景参与确实限制了原图输入模型的细粒度识别和检索排序，而 foreground crop 与原图全局信息存在互补。

## 主要 Baseline 与 Ablation

| 实验 | 方法 | Val Acc | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Transfer Strategy | Frozen Backbone | 50.75% | 77.92% | 未评估 | 未评估 | 未评估 | 未评估 | 训练稳定但上限低 |
| Transfer Strategy | Partial FT | 61.25% | 86.08% | 未评估 | 未评估 | 未评估 | 未评估 | 明显优于 frozen，但仍低于 full FT |
| Transfer Strategy | Full FT | 68.42% | 89.25% | 未评估 | 未评估 | 未评估 | 未评估 | ResNet 需要 full fine-tuning 适配 CUB |
| Augmentation | Basic | 64.83% | 87.58% | 未评估 | 未评估 | 未评估 | 未评估 | 原始 baseline |
| Augmentation | HFlip | 69.67% | 89.33% | 58.25% | 80.75% | 87.42% | 47.76% | ResNet 最优增强，也是 CNN retrieval baseline |
| Augmentation | RandomResizedCrop | 67.25% | 89.42% | 未评估 | 未评估 | 未评估 | 未评估 | 随机裁剪可能破坏局部细节 |
| Augmentation | RRC + HFlip + ColorJitter | 68.08% | 89.50% | 未评估 | 未评估 | 未评估 | 未评估 | Top-5 高，但 Top-1 不如 HFlip |
| Backbone | ResNet-18 HFlip | 69.67% | 89.33% | 58.25% | 80.75% | 87.42% | 47.76% | CNN baseline |
| Backbone | Swin-Tiny HFlip | 75.50% | 93.42% | 61.25% | 84.33% | 90.67% | 49.04% | 更强 backbone 同时提升分类和检索 |
| Foreground-aware | Swin-Tiny BBox Crop | 80.00% | 94.92% | 68.42% | 86.50% | 93.08% | 55.51% | 当前最有效优化方向 |
| Feature Fusion | Swin Original + BBox concat | 80.00% | 94.92% | 69.75% | 89.42% | 94.33% | 57.52% | 未压缩 fusion retrieval 对照 |
| Feature Fusion | Swin Original + BBox concat PCA | 80.00% | 94.92% | 69.83% | 89.50% | 94.33% | 57.72% | 当前最强 retrieval 结果 |

## Phase 5 Representation Learning

| 实验 | 方法 | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Loss Weight | lambda=0 | 66.33% | 65.63% | 88.83% | 52.33% | 78.67% | 85.75% | 39.24% | CE-only 对照 |
| Loss Weight | lambda=0.1 | 67.33% | 66.60% | 88.92% | 50.67% | 78.42% | 85.08% | 39.43% | 分类最优，mAP 仅小幅提升 |
| Loss Weight | lambda=0.25 | 66.83% | 65.96% | 87.58% | 50.50% | 77.33% | 85.25% | 38.27% | 无稳定收益 |
| Loss Weight | lambda=0.5 | 66.17% | 66.06% | 87.75% | 51.33% | 77.83% | 85.17% | 39.27% | mAP 略高但 Recall 不占优 |
| Loss Weight | lambda=1.0 | 64.92% | 64.36% | 87.50% | 48.67% | 74.75% | 84.08% | 37.41% | SupCon 权重过大有害 |
| Projection Head | No projection | 64.50% | 63.37% | 88.75% | 49.50% | 75.58% | 84.50% | 38.23% | 直接约束 h 效果较弱 |
| Projection Head | With MLP projection | 67.25% | 66.97% | 88.75% | 51.83% | 79.00% | 86.33% | 39.98% | projection head 有明确收益 |
| Temperature | tau=0.05 | 66.67% | 66.51% | 88.58% | 40.67% | 70.25% | 80.25% | 31.91% | retrieval 较弱 |
| Temperature | tau=0.07 | 67.00% | 66.94% | 87.92% | 40.83% | 68.92% | 79.75% | 32.07% | 默认 tau 不是当前最优 |
| Temperature | tau=0.1 | 68.00% | 67.78% | 88.33% | 43.42% | 72.42% | 82.67% | 34.89% | Phase 5 分类最优 |
| Temperature | tau=0.2 | 66.75% | 66.80% | 88.50% | 49.08% | 74.67% | 83.33% | 38.85% | Phase 5 projection retrieval 最优 |

Phase 5 的关键结论是：单纯增大 SupCon 权重不能稳定提升数值；projection head 是有效结构；temperature 对 retrieval geometry 影响明显。若继续优化 SupCon，应优先改 batch 采样策略，例如 class-balanced sampler，而不是继续盲目增大 `lambda`。

## 当前结论

当前项目数值提升路径已经比较清楚。原图输入下，Swin-Tiny 明显强于 ResNet-18；但 Phase 6 显示 Swin 和 ResNet 都仍有背景参与。BBox crop 直接把 Swin-Tiny 的 Val Acc 提升到 80.00%，mAP 提升到 55.51%，说明 foreground-aware input 是目前最有效的单模型优化方向。Original + BBox feature fusion 进一步把 mAP 提升到 57.52%，说明原图全局上下文和 bbox 前景细节存在互补。Fusion PCA 512-D 在 mAP 小幅提升到 57.72% 的同时把 storage 从 7.031 MiB 降到 2.344 MiB，因此当前默认 retrieval 表示应采用 `Swin Original + BBox concat PCA 512-D`。

下一阶段不应只盲目换更大 backbone。Phase 6 的错误模式已经指向喙、眼睛、头部、翼部纹理等局部线索不稳定，因此更有价值的方向是 `Part-aware / Local Feature Learning`：先用 CUB part annotations 量化 Grad-CAM / Swin attention 与关键部位的对齐程度，再做 local token pooling 和 local-global fusion。该方向已记录在 `docs/experiments/2026-08-30_part-aware-local-feature-learning.md`。
