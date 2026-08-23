# Backbone Retrieval Comparison

## Motivation

本实验比较 ResNet-18 feature 与 Swin-Tiny feature 的 retrieval 能力。核心问题是：classification accuracy 更高的 backbone，是否一定产生更好的 retrieval embedding？如果 Swin Accuracy 高于 ResNet，但 Swin Recall@10 或 mAP 接近 ResNet，这说明分类决策能力和 embedding geometry 不是同一个目标。

## Protocol

两个 backbone 都使用各自 best validation classification checkpoint，提取 classifier 前的 feature。Retrieval protocol 完全一致：validation split 作为 query/gallery，排除 query 自身，使用 L2-normalized cosine similarity，报告 Recall@1、Recall@5、Recall@10 和 mAP。开发阶段不使用 official test。

## Commands

ResNet retrieval summary 可复用 Phase 2 CE Retrieval Baseline，也可以重新生成到 `outputs/embeddings/backbone_retrieval`。Swin-Tiny 使用 Phase 4.1 的真实 best checkpoint 提取 feature，该 checkpoint 来自 `configs/baseline_swin_tiny_protocol.yaml` 原配置，run id 为 `20260823_012929_baseline_swin_tiny_protocol`。

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --checkpoint outputs\checkpoints\baseline_swin_tiny_protocol\20260823_012929_baseline_swin_tiny_protocol\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --batch-size 8 `
  --num-workers 0 `
  --device cpu `
  --output-dir outputs\embeddings\backbone_retrieval
```

```powershell
.\.venv\Scripts\python.exe scripts\compare_backbone_retrieval.py `
  --resnet-summary outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\summary.json `
  --swin-summary outputs\embeddings\backbone_retrieval\20260823_012929_baseline_swin_tiny_protocol\cosine\summary.json `
  --output-dir outputs\experiments\backbone_retrieval_comparison\resnet18_vs_swin_tiny
```

## Result

| Backbone | Classification Val Acc | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 69.67% | 512 | 58.25% | 80.75% | 87.42% | 47.76% |
| Swin-Tiny | 75.50% | 768 | 61.25% | 84.33% | 90.67% | 49.04% |

对照表输出位于 `outputs/experiments/backbone_retrieval_comparison/resnet18_vs_swin_tiny/summary.json` 和 `summary.csv`。Swin-Tiny 的 embedding dim 为 768，高于 ResNet-18 的 512，因此本实验同时反映了 backbone 表达能力和默认 feature dimension 的综合差异；如果后续需要严格比较相同维度，可以再增加 PCA-to-512 或 projection-to-512 对照。

## Decision

Swin-Tiny 在 classification 和 retrieval 上都优于 ResNet-18。Retrieval 提升幅度为：Recall@1 +3.00 percentage points，Recall@5 +3.58 percentage points，Recall@10 +3.25 percentage points，mAP +1.28 percentage points。这个结果说明更强 backbone 可以改善 CE feature 的检索能力，但 mAP 的提升小于 classification accuracy 的提升，说明 CrossEntropy 学到的 embedding 排序结构仍然有限。后续应继续进入 SupCon 或 CE+SupCon，直接优化类内紧致性和类间间隔。
