# 项目总实验结果表

本文档汇总 VisionSearch-FG 当前已经完成的主要实验结果。除特别说明外，结果均来自 CUB-200-2011 official train 内部划分出的 validation split，共 1,200 张图；retrieval protocol 为 query/gallery 同用 validation split，并排除 query 自身。表中的 `BBox crop` 使用了 CUB 官方 bounding box，属于 annotation-assisted setting，不能与原图输入 baseline 解释为完全相同输入条件。

## 主结果总表

| 方法 | Backbone | 输入 | Objective | Feature | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 说明 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet-18 Full FT + HFlip | ResNet-18 | 原图 224 | CE | backbone h | 69.67% | 未记录 | 89.33% | 58.25% | 80.75% | 87.42% | 47.76% | 主 CNN baseline |
| Swin-Tiny Full FT + HFlip | Swin-Tiny | 原图 224 | CE | backbone h | 75.50% | 75.28% | 93.42% | 61.25% | 84.33% | 90.67% | 49.04% | 主 Transformer baseline |
| Swin-Tiny 512-D PCA | Swin-Tiny | 原图 224 | CE | PCA 512-D | 75.50% | 75.28% | 93.42% | 61.67% | 84.75% | 90.50% | 49.22% | 同一个 Swin checkpoint 的 PCA 压缩检索结果 |
| Swin-Tiny BBox Crop 224 | Swin-Tiny | BBox crop 224, margin=0.15 | CE | backbone h | 80.00% | 79.52% | 94.92% | 68.42% | 86.50% | 93.08% | 55.51% | foreground-aware baseline |
| Swin-Tiny BBox 224 + ArcFace | Swin-Tiny | BBox crop 224, margin=0.15 | ArcFace | backbone h | 84.92% | 84.50% | 95.17% | 82.33% | 91.58% | 95.17% | 77.30% | angular margin 明显改善 cosine retrieval |
| Swin-Tiny BBox 224 + ArcFace + Flip TTA | Swin-Tiny | BBox crop 224 + flip TTA | ArcFace + post-processing | backbone h | 84.92% | 84.50% | 95.17% | 82.67% | 91.33% | 95.50% | 77.71% | ArcFace embedding 进一步受益于 TTA |
| Swin-Tiny BBox 224 + ArcFace + Flip TTA + QE | Swin-Tiny | BBox crop 224 + flip TTA | ArcFace + post-processing | backbone h | 84.92% | 84.50% | 95.17% | 83.00% | 90.33% | 94.25% | 78.62% | Swin-Tiny ArcFace 后处理最强，QE top3 alpha0.5 |
| Swin-Tiny BBox 224 + ArcFace + SupCon | Swin-Tiny | BBox crop 224, margin=0.15 | ArcFace + SupCon | backbone h | 84.50% | 84.29% | 94.67% | 79.58% | 91.33% | 95.92% | 73.20% | 低于单独 ArcFace，说明直接叠加 SupCon 不稳定 |
| Swin-Tiny BBox 224 + CosFace | Swin-Tiny | BBox crop 224, margin=0.15 | CosFace | backbone h | 84.33% | 83.98% | 93.92% | 80.25% | 91.08% | 94.67% | 74.46% | 明显优于 CE，但低于 ArcFace |
| Swin Original + BBox concat | Swin-Tiny | 原图 224 + BBox crop 224 | CE | concat L2 backbone h | 80.00% | 79.52% | 94.92% | 69.75% | 89.42% | 94.33% | 57.52% | 未压缩 fusion retrieval 对照；分类指标沿用 BBox crop 模型 |
| Swin Original + BBox concat PCA | Swin-Tiny | 原图 224 + BBox crop 224 | CE | PCA 512-D | 80.00% | 79.52% | 94.92% | 69.83% | 89.50% | 94.33% | 57.72% | fusion 阶段最强 retrieval，且存储低于 1536-D fusion |
| Swin BBox Evidence-weighted Local | Swin-Tiny | BBox crop 224, margin=0.15 | CE | class-evidence weighted local | 80.00% | 79.52% | 94.92% | 71.25% | 87.83% | 93.58% | 59.62% | part-aware post-hoc 阶段最高 mAP / Recall@1；不重新训练 |
| Swin Head-aware Soft Selector | Swin-Tiny | BBox crop 224, margin=0.15 | CE + head selector | global+learned local | 82.83% | 82.55% | 95.42% | 67.67% | 86.92% | 93.83% | 55.71% | part-aware training 阶段分类最高；retrieval 未超过 evidence-weighted local |
| Swin-Tiny BBox Crop 448 | Swin-Tiny | BBox crop 448, margin=0.15 | CE | backbone h | 81.33% | 81.05% | 95.42% | 70.58% | 89.75% | 93.67% | 58.97% | 提高分辨率后分类提升，但 mAP 仍低于 evidence-weighted local |
| ConvNeXt V2 Nano BBox 448 | ConvNeXt V2 Nano | BBox crop 448, margin=0.15 | CE | backbone h | 87.00% | 86.72% | 97.08% | 78.42% | 93.00% | 96.33% | 68.55% | 更强 CNN backbone 明显提升 |
| ConvNeXt V2 Tiny BBox 448 | ConvNeXt V2 Tiny | BBox crop 448, margin=0.15 | CE | backbone h | 86.42% | 85.74% | 97.33% | 82.75% | 93.75% | 96.92% | 75.06% | 强 backbone CE retrieval baseline |
| ConvNeXt V2 Tiny BBox 448 + ArcFace | ConvNeXt V2 Tiny | BBox crop 448, margin=0.15 | ArcFace | backbone h | 87.83% | 87.69% | 95.92% | 85.75% | 93.33% | 96.67% | 81.33% | strong backbone + angular margin 明显叠加 |
| ConvNeXt V2 Tiny BBox 448 + ArcFace + Flip TTA | ConvNeXt V2 Tiny | BBox crop 448 + flip TTA | ArcFace + post-processing | backbone h | 87.83% | 87.69% | 95.92% | 86.50% | 93.25% | 96.75% | 82.34% | TTA 进一步提升 Recall@1 和 mAP |
| ConvNeXt V2 Tiny BBox 448 + ArcFace + Flip TTA + QE | ConvNeXt V2 Tiny | BBox crop 448 + flip TTA | ArcFace + post-processing | backbone h | 87.83% | 87.69% | 95.92% | 86.25% | 92.42% | 96.33% | 82.81% | 当前整体 mAP 最强，QE top3 alpha0.5 |
| DINOv2 Base Frozen BBox 448 | DINOv2 Base | BBox crop 448, margin=0.15 | frozen CE | backbone h | 89.25% | 89.06% | 98.17% | 78.00% | 93.33% | 96.75% | 68.02% | 当前分类最强 |
| DINOv2 Small Partial BBox 448 | DINOv2 Small | BBox crop 448, margin=0.15 | partial CE | backbone h | 87.25% | 86.99% | 97.42% | 79.75% | 94.67% | 97.67% | 70.48% | DINOv2 内部检索最强 |
| ConvNeXt V2 Tiny BBox 448 + SupCon | ConvNeXt V2 Tiny | BBox crop 448, margin=0.15 | CE + SupCon | backbone h | 84.25% | 83.96% | 96.17% | 79.50% | 93.25% | 96.50% | 72.02% | 强 backbone 上 SupCon 未超过 CE baseline |
| DINOv2 Small Partial BBox 448 + SupCon | DINOv2 Small | BBox crop 448, margin=0.15 | CE + SupCon | backbone h | 86.25% | 85.91% | 96.58% | 79.83% | 94.42% | 97.42% | 70.54% | mAP 与 CE baseline 基本持平 |
| ConvNeXt V2 Tiny BBox 448 + Flip TTA | ConvNeXt V2 Tiny | BBox crop 448 + flip TTA | CE + post-processing | backbone h | 86.42% | 85.74% | 97.33% | 82.67% | 94.42% | 97.25% | 75.74% | 稳定后处理，Recall@5/10 和 mAP 均提升 |
| ConvNeXt V2 Tiny BBox 448 + QE | ConvNeXt V2 Tiny | BBox crop 448, margin=0.15 | CE + post-processing | backbone h | 86.42% | 85.74% | 97.33% | 82.67% | 93.25% | 96.17% | 75.82% | 单独 Query Expansion 最强，top3 alpha0.5 |
| ConvNeXt V2 Tiny BBox 448 + Flip TTA + QE | ConvNeXt V2 Tiny | BBox crop 448 + flip TTA | CE + post-processing | backbone h | 86.42% | 85.74% | 97.33% | 82.83% | 93.83% | 96.58% | 76.82% | CE backbone 上最强后处理结果，TTA + Query Expansion top3 alpha0.5 |

当前最高分类结果是 `DINOv2 Base Frozen BBox 448`，Val Acc 为 89.25%，Macro-F1 为 89.06%。当前最高 mAP 检索结果是 `ConvNeXt V2 Tiny BBox 448 + ArcFace + Flip TTA + Query Expansion top3 alpha0.5`，mAP 为 82.81%；当前最高 Recall@1 是 `ConvNeXt V2 Tiny BBox 448 + ArcFace + Flip TTA + Query Expansion top3 alpha0.1`，Recall@1 为 86.67%。这说明强 backbone、BBox crop、高分辨率输入和 angular margin objective 可以叠加；ArcFace 在 normalized angular space 中训练，直接改善了 cosine retrieval geometry。

同时，`DINOv2 Base Frozen` 的分类最强但 mAP 只有 68.02%，低于 ConvNeXt V2 Tiny CE 和 ConvNeXt V2 Tiny ArcFace；`DINOv2 Small Partial` 的 mAP 提升到 70.48%，但仍没有超过 ConvNeXt V2 Tiny。这进一步说明分类性能和 retrieval representation quality 相关但不等价。Phase 12 中，强 backbone 上的 CE + SupCon 没有稳定超过 CE baseline；Phase 11.1/11.2 中 ArcFace 明显超过 CE，因此后续检索优化应优先围绕 angular margin、sampler、hard negative mining、memory bank 或 part-aware local retrieval 展开，而不是简单增加 SupCon 权重。

## Oracle 诊断结果

以下实验使用 CUB 人工 part coordinates，因此只作为 upper bound / diagnostic analysis，不能作为正式无标注检索 pipeline 与主结果等价比较。

| 方法 | Backbone | 输入 | Feature | Recall@1 | Recall@5 | Recall@10 | mAP | 说明 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Swin BBox Global | Swin-Tiny | BBox crop 224 | backbone h | 68.42% | 86.50% | 93.08% | 55.51% | Oracle crop 对照中的全局 baseline |
| Swin BBox + Oracle Head | Swin-Tiny | BBox crop 224 + CUB head crop | global_head_concat_l2 | 72.42% | 90.00% | 94.00% | 59.17% | 使用人工 head part crop；证明头部局部信息有明显上界收益 |
| Swin BBox + Oracle Head + Wing | Swin-Tiny | BBox crop 224 + head + wing crop | global_head_wing_concat_l2 | 69.42% | 88.92% | 93.58% | 56.43% | 简单加入 wing 后低于只加 head |
| Swin BBox + Oracle Head + Wing + Body | Swin-Tiny | BBox crop 224 + head + wing + body crop | global_head_wing_body_concat_l2 | 70.00% | 89.75% | 93.67% | 57.82% | 多局部融合有效，但维度更高且不如只加 head |
| Swin BBox Evidence-weighted Local Oracle | Swin-Tiny | BBox crop 224 | true-class weighted local, tau=0.5 | 75.08% | 92.75% | 95.92% | 63.18% | 使用真实类别引导 token weighting；只作为 selector 上界 |

该诊断结果说明：局部 part 信息确实有检索价值，但收益主要来自 `head`，不是简单堆叠越多 part 越好。`true-class weighted local` 的 oracle 结果进一步说明，当前 weighted local 的上界受类别预测质量影响；如果 selector 能减少错误类别引导，retrieval 仍有约 3.56 个百分点 mAP 的上界空间。下一步如果做真正的 part-aware training，应优先设计 head-aware 或 discriminative local selector，而不是平均处理所有部位。

## 自动局部裁剪诊断

以下实验不使用人工 part coordinates，只使用 Swin class-evidence token heatmap 自动生成局部 crop。

| 方法 | Backbone | 输入 | Feature | Recall@1 | Recall@5 | Recall@10 | mAP | 说明 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Swin BBox Global | Swin-Tiny | BBox crop 224 | backbone h | 68.42% | 86.50% | 93.08% | 55.51% | 对照 baseline |
| Swin BBox Evidence-weighted Local | Swin-Tiny | BBox crop 224 | token-level weighted local | 71.25% | 87.83% | 93.58% | 59.62% | soft token pooling，该阶段最高 mAP |
| Swin BBox Evidence Auto Crop | Swin-Tiny | evidence heatmap hard crop | auto crop embedding | 39.75% | 65.42% | 75.17% | 27.75% | 自动硬裁剪明显失败 |
| Swin BBox Global + Evidence Auto Crop | Swin-Tiny | BBox crop 224 + auto crop | concat_l2 | 64.08% | 85.92% | 91.67% | 50.61% | 拼接后仍低于 global baseline |
| Swin BBox Top-M Evidence | Swin-Tiny | BBox crop 224 | top-M class evidence, M=3, tau=1.0 | 71.08% | 88.08% | 93.50% | 59.34% | Recall@5 略高，但 mAP 低于 top-1 |

该结果说明：当前 class-evidence token heatmap 更适合做 soft weighting，而不适合直接转成 hard crop。硬裁剪会丢失上下文并放大错误局部区域。Top-M class ensemble 也没有超过 top-1 predicted evidence，说明额外类别主要引入噪声。因此后续 part-aware 方向应优先做 token-level local pooling、attention regularization 或 soft region aggregation，而不是 hard crop 或 top-M ensemble。

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
| Angular Margin | Swin-Tiny BBox 224 ArcFace | 84.92% | 95.17% | 82.33% | 91.58% | 95.17% | 77.30% | 单模型 ArcFace 最强 |
| Angular Margin + Post-processing | Swin-Tiny BBox 224 ArcFace + Flip TTA + QE | 84.92% | 95.17% | 83.00% | 90.33% | 94.25% | 78.62% | Swin-Tiny ArcFace 后处理最强 |
| Strong Backbone + Angular Margin | ConvNeXt V2 Tiny BBox 448 ArcFace | 87.83% | 95.92% | 85.75% | 93.33% | 96.67% | 81.33% | 强 backbone 与 ArcFace 收益叠加 |
| Strong Backbone + Angular Margin + Post-processing | ConvNeXt V2 Tiny BBox 448 ArcFace + Flip TTA + QE | 87.83% | 95.92% | 86.25% | 92.42% | 96.33% | 82.81% | 当前整体 mAP 最强 |
| Angular Margin | Swin-Tiny BBox 224 CosFace | 84.33% | 93.92% | 80.25% | 91.08% | 94.67% | 74.46% | 明显优于普通 CE |
| Angular Margin | Swin-Tiny BBox 224 ArcFace + SupCon | 84.50% | 94.67% | 79.58% | 91.33% | 95.92% | 73.20% | 低于单独 ArcFace |
| Feature Fusion | Swin Original + BBox concat | 80.00% | 94.92% | 69.75% | 89.42% | 94.33% | 57.52% | 未压缩 fusion retrieval 对照 |
| Feature Fusion | Swin Original + BBox concat PCA | 80.00% | 94.92% | 69.83% | 89.50% | 94.33% | 57.72% | fusion 阶段最高 Recall@5 / Recall@10 |
| Part-aware Post-hoc | BBox evidence-weighted local tau1.0 | 80.00% | 94.92% | 71.25% | 87.83% | 93.58% | 59.62% | part-aware post-hoc 阶段最高 mAP / Recall@1 |
| Part-aware Training | Head-aware soft selector | 82.83% | 95.42% | 67.67% | 86.92% | 93.83% | 55.71% | 分类最高，但检索未提升 |

## Phase 5 Representation Learning

| 实验 | 方法 | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Loss Weight | lambda=0 | 66.33% | 65.63% | 88.83% | 52.33% | 78.67% | 85.75% | 39.24% | CE-only 对照 |
| Loss Weight | lambda=0.1 | 67.33% | 66.60% | 88.92% | 50.67% | 78.42% | 85.08% | 39.43% | 分类最优，mAP 仅小幅提升 |
| P×K Sampling | P=8, K=2 | 67.83% | 67.17% | 89.75% | 42.08% | 71.67% | 81.08% | 35.49% | 分类略升，但 retrieval 明显低于 random baseline |
| P×K Sampling | P=4, K=4 | 65.83% | 64.65% | 87.58% | 39.17% | 71.83% | 80.75% | 34.03% | positive 更多但 negative 类别更少，整体退化 |
| Loss Weight | lambda=0.25 | 66.83% | 65.96% | 87.58% | 50.50% | 77.33% | 85.25% | 38.27% | 无稳定收益 |
| Loss Weight | lambda=0.5 | 66.17% | 66.06% | 87.75% | 51.33% | 77.83% | 85.17% | 39.27% | mAP 略高但 Recall 不占优 |
| Loss Weight | lambda=1.0 | 64.92% | 64.36% | 87.50% | 48.67% | 74.75% | 84.08% | 37.41% | SupCon 权重过大有害 |
| Projection Head | No projection | 64.50% | 63.37% | 88.75% | 49.50% | 75.58% | 84.50% | 38.23% | 直接约束 h 效果较弱 |
| Projection Head | With MLP projection | 67.25% | 66.97% | 88.75% | 51.83% | 79.00% | 86.33% | 39.98% | projection head 有明确收益 |
| Temperature | tau=0.05 | 66.67% | 66.51% | 88.58% | 40.67% | 70.25% | 80.25% | 31.91% | retrieval 较弱 |
| Temperature | tau=0.07 | 67.00% | 66.94% | 87.92% | 40.83% | 68.92% | 79.75% | 32.07% | 默认 tau 不是当前最优 |
| Temperature | tau=0.1 | 68.00% | 67.78% | 88.33% | 43.42% | 72.42% | 82.67% | 34.89% | Phase 5 分类最优 |
| Temperature | tau=0.2 | 66.75% | 66.80% | 88.50% | 49.08% | 74.67% | 83.33% | 38.85% | Phase 5 projection retrieval 最优 |

Phase 5 的关键结论是：单纯增大 SupCon 权重不能稳定提升数值；projection head 是有效结构；temperature 对 retrieval geometry 影响明显。后续补充的 P×K sampling 显示，当前 ResNet-18 CE + SupCon 的瓶颈不只是 batch composition；P8K2 分类略升但 retrieval 明显下降，P4K4 整体退化。因此不建议继续在 ResNet-18 上扩大 P×K 搜索，应优先转向 Swin/BBox 表示上的 metric learning 或 local-token retrieval 约束。

## Phase 10-12：高分辨率、强 Backbone 与强 Backbone SupCon

| 实验 | 方法 | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Resolution | Swin Original 448 | 78.83% | 78.37% | 95.08% | 67.08% | 88.17% | 93.58% | 53.92% | 纯提高分辨率有效，但不如 BBox 448 |
| Resolution | Swin BBox 448 | 81.33% | 81.05% | 95.42% | 70.58% | 89.75% | 93.67% | 58.97% | Foreground + detail 更有效 |
| Backbone | ConvNeXt V2 Nano BBox 448 | 87.00% | 86.72% | 97.08% | 78.42% | 93.00% | 96.33% | 68.55% | 小模型也强于 Swin |
| Backbone | ConvNeXt V2 Tiny BBox 448 | 86.42% | 85.74% | 97.33% | 82.75% | 93.75% | 96.92% | 75.06% | 强 backbone 阶段检索最强 |
| Backbone | Swin-Small BBox 448 | 86.50% | 86.07% | 97.25% | 81.92% | 94.75% | 96.75% | 72.82% | 参数更多，检索强但不如 ConvNeXt V2 Tiny |
| DINOv2 | DINOv2 Small Frozen | 87.42% | 87.39% | 97.75% | 75.67% | 93.67% | 96.58% | 63.33% | frozen representation 分类强，检索一般 |
| DINOv2 | DINOv2 Base Frozen | 89.25% | 89.06% | 98.17% | 78.00% | 93.33% | 96.75% | 68.02% | 当前分类最强 |
| DINOv2 | DINOv2 Small Full FT | 77.25% | 76.84% | 94.67% | 70.08% | 89.08% | 94.17% | 56.38% | 全量 fine-tuning 破坏 representation |
| DINOv2 | DINOv2 Small Partial Last 2 Blocks | 87.25% | 86.99% | 97.42% | 79.75% | 94.67% | 97.67% | 70.48% | DINOv2 内部检索最强 |
| Strong Backbone SupCon | ConvNeXt V2 Tiny CE + SupCon | 84.25% | 83.96% | 96.17% | 79.50% | 93.25% | 96.50% | 72.02% | 低于 ConvNeXt CE baseline |
| Strong Backbone SupCon | DINOv2 Small Partial CE + SupCon | 86.25% | 85.91% | 96.58% | 79.83% | 94.42% | 97.42% | 70.54% | 与 DINOv2 CE baseline 基本持平 |
| Angular Margin | Swin-Tiny BBox 224 ArcFace | 84.92% | 84.50% | 95.17% | 82.33% | 91.58% | 95.17% | 77.30% | 单模型 ArcFace 最强，训练目标更匹配 cosine retrieval |
| Angular Margin + Post-processing | Swin-Tiny BBox 224 ArcFace + Flip TTA | 84.92% | 84.50% | 95.17% | 82.67% | 91.33% | 95.50% | 77.71% | Flip TTA 小幅提升 mAP |
| Angular Margin + Post-processing | Swin-Tiny BBox 224 ArcFace + Flip TTA + QE top3 alpha0.5 | 84.92% | 84.50% | 95.17% | 83.00% | 90.33% | 94.25% | 78.62% | Swin-Tiny ArcFace 后处理最强 |
| Strong Backbone + Angular Margin | ConvNeXt V2 Tiny BBox 448 ArcFace | 87.83% | 87.69% | 95.92% | 85.75% | 93.33% | 96.67% | 81.33% | 强 backbone 与 ArcFace 收益叠加 |
| Strong Backbone + Angular Margin + Post-processing | ConvNeXt V2 Tiny BBox 448 ArcFace + Flip TTA | 87.83% | 87.69% | 95.92% | 86.50% | 93.25% | 96.75% | 82.34% | TTA 进一步提升 Recall@1 和 mAP |
| Strong Backbone + Angular Margin + Post-processing | ConvNeXt V2 Tiny BBox 448 ArcFace + Flip TTA + QE top3 alpha0.5 | 87.83% | 87.69% | 95.92% | 86.25% | 92.42% | 96.33% | 82.81% | 当前整体 mAP 最强 |
| Angular Margin | Swin-Tiny BBox 224 CosFace | 84.33% | 83.98% | 93.92% | 80.25% | 91.08% | 94.67% | 74.46% | 明显优于普通 CE，但低于 ArcFace |
| Angular Margin | Swin-Tiny BBox 224 ArcFace + SupCon | 84.50% | 84.29% | 94.67% | 79.58% | 91.33% | 95.92% | 73.20% | 直接叠加 SupCon 没有带来互补收益 |
| Retrieval Post-processing | ConvNeXt V2 Tiny BBox + Flip TTA | 86.42% | 85.74% | 97.33% | 82.67% | 94.42% | 97.25% | 75.74% | 稳定提升 Recall@5/10 和 mAP |
| Retrieval Post-processing | ConvNeXt V2 Tiny QE top3 alpha0.5 | 86.42% | 85.74% | 97.33% | 82.67% | 93.25% | 96.17% | 75.82% | 单独 Query Expansion 最强 |
| Retrieval Post-processing | ConvNeXt V2 Tiny BBox + Flip TTA + QE top3 alpha0.5 | 86.42% | 85.74% | 97.33% | 82.83% | 93.83% | 96.58% | 76.82% | CE backbone 后处理最强 |

Phase 10-12 的关键结论是：`BBox crop + 448 input + stronger backbone` 是非常有效的提升路径。ConvNeXt V2 Tiny 的 CE embedding 已经形成很强的 retrieval geometry，简单加入 SupCon 没有进一步提升。DINOv2 的 frozen linear classification 很强，但 retrieval ranking 不如 ConvNeXt；小范围 partial fine-tuning 可以显著提升 DINOv2 retrieval，但仍没有超过 ConvNeXt V2 Tiny。

Phase 13 的 BBox + Flip TTA 将 mAP 从 `75.06%` 提升到 `75.74%`，并同时提升 Recall@5/10，是更稳定的默认后处理。进一步叠加 Query Expansion top3 alpha0.5 后，mAP 提升到 `76.82%`，成为 CE backbone 上的最强后处理结果。

Phase 11.1 的 Angular Margin 实验进一步改写了当前结论：Swin-Tiny BBox 224 + ArcFace 在不使用 448 分辨率和检索后处理的情况下，mAP 达到 `77.30%`，超过 ConvNeXt V2 Tiny BBox 448 + TTA + QE。Phase 11.2 中，Swin ArcFace + Flip TTA + Query Expansion top3 alpha0.5 继续把 mAP 提升到 `78.62%`；迁移到 ConvNeXt V2 Tiny BBox448 后，基础 ArcFace mAP 达到 `81.33%`，Flip TTA + Query Expansion top3 alpha0.5 达到 `82.81%`。这说明 retrieval 的核心瓶颈不仅是 backbone capacity 和输入分辨率，也包括训练目标是否与 cosine nearest-neighbor ranking 对齐。

## 当前结论

当前项目数值提升路径已经比较清楚。原图输入下，Swin-Tiny 明显强于 ResNet-18；但 Phase 6 显示 Swin 和 ResNet 都仍有背景参与。BBox crop 直接把 Swin-Tiny 的 Val Acc 提升到 80.00%，mAP 提升到 55.51%，说明 foreground-aware input 是早期最有效的单模型优化方向。继续提高到 BBox crop 448 后，Swin-Tiny 的 mAP 达到 58.97%。再换成 ConvNeXt V2 Tiny 后，mAP 提升到 75.06%。通过 BBox + Flip TTA + Query Expansion，mAP 进一步提升到 76.82%。ArcFace 实验先把 Swin-Tiny BBox 224 的 mAP 提升到 77.30%，并通过 Flip TTA + Query Expansion 提升到 78.62%。最新的 ConvNeXt V2 Tiny BBox448 ArcFace 则达到 81.33% mAP，结合 Flip TTA + Query Expansion 后达到 82.81%，成为当前整体检索最强结果。

下一阶段不应继续盲目换更大 backbone，也不应简单增加 SupCon 权重。Phase 12 表明，在强 backbone 上直接加入 CE + SupCon 没有超过 CE baseline；Phase 11.1/11.2 表明，ArcFace 这类 angular margin objective 是当前最有效的新方向。更有价值的路线是在 ConvNeXt V2 Tiny BBox448 ArcFace 上继续做 margin / scale ablation、TTA view ablation、hard negative mining、memory bank 或 part-aware local re-ranking。
