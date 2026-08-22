# Phase 2 Day 1：Classification Feature Retrieval Baseline

## 目标

本阶段不重新训练模型，而是直接使用 Phase 1 中 CrossEntropy 训练得到的最佳 ResNet-18 checkpoint，提取 classifier 前的 512-D feature，并评估这些 feature 是否自然形成可用于图像检索的 embedding space。

## Research Question

```text
仅使用 CrossEntropy 训练得到的 ResNet-18 feature，能否自然形成具有检索能力的 embedding space？
```

## Method

流程：

```text
Image
↓
Best ResNet-18 checkpoint
↓
512-D feature before classifier
↓
L2 normalize
↓
Cosine similarity
↓
Nearest-neighbor retrieval
```

对每张 query 图片，在同一个 validation split 中检索最相似图片。query 自己必须从 gallery 中排除，否则自身相似度为 1，会制造假的 Recall@1。

## Metrics

本阶段引入：

```text
Recall@1
Recall@5
Recall@10
mAP
```

Recall@K 表示同类图片是否出现在 top-K 检索结果中。mAP 衡量所有相关图片在完整排序中的平均位置，能比 Recall@K 更细致地反映排序质量。

## 输入 checkpoint

当前建议使用 Phase 1 augmentation ablation 中表现最好的 HFlip checkpoint：

```text
outputs/checkpoints/ablation_resnet18_fullft_aug_hflip/20260822_144519_ablation_resnet18_fullft_aug_hflip/best.pt
```

对应配置：

```text
configs/ablation_resnet18_fullft_aug_hflip.yaml
```

## 运行指令

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\ablation_resnet18_fullft_aug_hflip.yaml `
  --checkpoint outputs\checkpoints\ablation_resnet18_fullft_aug_hflip\20260822_144519_ablation_resnet18_fullft_aug_hflip\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu
```

输出目录：

```text
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/
```

主要输出：

```text
summary.json
embeddings.npy
records.csv
top_results.json
```

## Hypothesis

分类 feature 应具有一定语义结构，因此能够实现基础检索；但 CrossEntropy 主要优化分类边界，没有直接约束同类样本在 embedding space 中紧密聚集，因此 retrieval 性能应有提升空间。后续 SupCon、Swin、CE + SupCon 都必须和这个 CE Retrieval Baseline 比较。
