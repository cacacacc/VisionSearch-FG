# CE Retrieval Baseline

## Motivation

本实验评估 CrossEntropy 训练得到的 ResNet-18 512-D feature 是否能直接用于图像检索。该实验不重新训练模型，只使用 Phase 1 中表现最好的 ResNet-18 Full Fine-tuning + HFlip checkpoint 提取 embedding。

## Protocol

评估 split 为 validation split：`data/processed/splits/cub_val_ids_seed42.txt`，共 1,200 张图片。每张图片都作为 query，同时 validation set 作为 gallery；检索时排除 query 自身，使用 L2 normalized embedding 的 cosine similarity 排序。指标为 Recall@1、Recall@5、Recall@10 和 mAP。

## Result

| Checkpoint | Samples | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260822_144519_ablation_resnet18_fullft_aug_hflip/best.pt` | 1,200 | 512 | 58.25% | 80.75% | 87.42% | 47.76% |

## Analysis

CE feature 已经形成可用的语义结构，Recall@5 和 Recall@10 较高，说明同类图片经常出现在候选结果前列。但 Recall@1 和 mAP 仍有明显提升空间，这符合预期：CrossEntropy 主要学习分类边界，并不直接约束同类样本在 embedding space 中紧密聚集。后续 SupCon 或 CE + SupCon 的目标，就是在不损害分类能力的前提下提升 Recall@K 和 mAP。

## Decision

将该结果作为 Phase 2 的 CE Retrieval Baseline。后续所有 retrieval 改进方法都需要与此表比较，并使用相同 validation split、相同 query/gallery 排除自身协议和相同指标定义。
