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
| Swin BBox Evidence-weighted Local | Swin-Tiny | BBox crop 224, margin=0.15 | CE | class-evidence weighted local | 80.00% | 79.52% | 94.92% | 71.25% | 87.83% | 93.58% | 59.62% | 当前最高 mAP / Recall@1；不重新训练 |
| Swin Head-aware Soft Selector | Swin-Tiny | BBox crop 224, margin=0.15 | CE + head selector | global+learned local | 82.83% | 82.55% | 95.42% | 67.67% | 86.92% | 93.83% | 55.71% | 当前最高分类；retrieval 未超过 evidence-weighted local |

当前最高分类结果是 `Swin Head-aware Soft Selector`，Val Acc 为 82.83%，Macro-F1 为 82.55%。当前最高 mAP / Recall@1 检索结果仍是 `Swin BBox Evidence-weighted Local`，mAP 为 59.62%，Recall@1 为 71.25%；当前最高 Recall@5 / Recall@10 仍来自 `Swin Original + BBox concat PCA` 或未压缩 fusion，Recall@5 为 89.50%，Recall@10 为 94.33%。相对原图 Swin-Tiny baseline，BBox crop 的 Val Acc 从 75.50% 提升到 80.00%，提升 4.50 个百分点；head-aware selector 进一步把分类 Val Acc 提升到 82.83%。但 head-aware selector 的 retrieval mAP 只有 55.71%，说明分类性能提升不必然带来更好的检索 embedding。class-evidence weighted local pooling 的 mAP 从 49.04% 提升到 59.62%，Recall@1 从 61.25% 提升到 71.25%，仍是当前最强检索方案。

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
| Swin BBox Evidence-weighted Local | Swin-Tiny | BBox crop 224 | token-level weighted local | 71.25% | 87.83% | 93.58% | 59.62% | soft token pooling，当前最高 mAP |
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
| Feature Fusion | Swin Original + BBox concat | 80.00% | 94.92% | 69.75% | 89.42% | 94.33% | 57.52% | 未压缩 fusion retrieval 对照 |
| Feature Fusion | Swin Original + BBox concat PCA | 80.00% | 94.92% | 69.83% | 89.50% | 94.33% | 57.72% | 当前最高 Recall@5 / Recall@10 |
| Part-aware Post-hoc | BBox evidence-weighted local tau1.0 | 80.00% | 94.92% | 71.25% | 87.83% | 93.58% | 59.62% | 当前最高 mAP / Recall@1 |
| Part-aware Training | Head-aware soft selector | 82.83% | 95.42% | 67.67% | 86.92% | 93.83% | 55.71% | 分类最高，但检索未提升 |

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

当前项目数值提升路径已经比较清楚。原图输入下，Swin-Tiny 明显强于 ResNet-18；但 Phase 6 显示 Swin 和 ResNet 都仍有背景参与。BBox crop 直接把 Swin-Tiny 的 Val Acc 提升到 80.00%，mAP 提升到 55.51%，说明 foreground-aware input 是目前最有效的单模型优化方向。Original + BBox feature fusion 进一步把 mAP 提升到 57.52%，说明原图全局上下文和 bbox 前景细节存在互补。Fusion PCA 512-D 在 mAP 小幅提升到 57.72% 的同时把 storage 从 7.031 MiB 降到 2.344 MiB，因此如果优先考虑 Recall@5 / Recall@10 与存储成本，默认 retrieval 表示应采用 `Swin Original + BBox concat PCA 512-D`。

下一阶段不应只盲目换更大 backbone。Phase 6 的错误模式已经指向喙、眼睛、头部、翼部纹理等局部线索不稳定，因此更有价值的方向是 `Part-aware / Local Feature Learning`。8.2 证明 naive token norm Top-K 不够可靠；8.3 证明 class-evidence weighted local pooling 能显著提高 mAP 和 Recall@1；8.4a 的 oracle part crop 进一步证明 head 局部信息具有明确上界收益；8.4b 说明把 evidence heatmap 直接做 hard crop 会显著退化；8.4c 说明 `predicted tau=1.0` 是稳定的正式设置，而 `true tau=0.5` 暴露了 selector 上界；8.4d 说明 Top-M class ensemble 无法有效缩小该上界差距；8.5 说明 head-aware selector 能提升分类，但仅靠 CE + head selector BCE 不能改善 retrieval geometry。下一步如果目标是检索，应加入 SupCon / metric learning，而不是只做 head localization。
