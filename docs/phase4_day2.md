# Phase 4 Day 2：Backbone Retrieval Comparison

## 目标

本阶段比较 ResNet feature 和 Swin feature 在完全相同 retrieval protocol 下的表现。它回答一个关键问题：分类表现更好的 backbone，是否一定产生更好的 retrieval embedding？这个问题很重要，因为 classification accuracy 优化的是类别决策边界，而 retrieval 更关心 embedding space 中同类样本是否靠近、异类样本是否分开。

## Research Question

```text
分类表现更好的 backbone 是否一定产生更好的 retrieval embedding？
```

## Controlled Variables

固定项包括 validation split、query/gallery protocol、排除 query 自身、L2-normalized cosine similarity、Recall@1/5/10 和 mAP。唯一主要变化是 backbone checkpoint：ResNet-18 使用 Phase 1 最佳 Full FT + HFlip checkpoint；Swin-Tiny 使用 Phase 4.1 训练得到的 best validation checkpoint。开发阶段仍然只使用 validation，不使用 official test。

## Method

ResNet feature 已有 CE Retrieval Baseline。Swin-Tiny 训练完成后，使用同一个脚本提取 classifier 前的 embedding，并在 validation split 上评估 retrieval。

ResNet retrieval 如果需要重新生成到统一目录，可以运行：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\ablation_resnet18_fullft_aug_hflip.yaml `
  --checkpoint outputs\checkpoints\ablation_resnet18_fullft_aug_hflip\20260822_144519_ablation_resnet18_fullft_aug_hflip\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --device cpu `
  --output-dir outputs\embeddings\backbone_retrieval
```

Swin-Tiny retrieval 使用 Phase 4.1 的真实 best checkpoint。当前 Swin run 使用 `configs/baseline_swin_tiny_protocol.yaml` 原配置，run id 为 `20260823_012929_baseline_swin_tiny_protocol`：

```powershell
cd D:\code\VisionSearch-FG
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

生成两个 `summary.json` 后，运行对照表汇总：

```powershell
.\.venv\Scripts\python.exe scripts\compare_backbone_retrieval.py `
  --resnet-summary outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\summary.json `
  --swin-summary outputs\embeddings\backbone_retrieval\20260823_012929_baseline_swin_tiny_protocol\cosine\summary.json `
  --output-dir outputs\experiments\backbone_retrieval_comparison\resnet18_vs_swin_tiny
```

## Metrics

本实验记录 Recall@1、Recall@5、Recall@10 和 mAP。Recall@K 衡量同类图片是否进入 top-K；mAP 衡量同类图片在完整排序中的整体位置。Swin 的 embedding dim 是 768，ResNet 的 embedding dim 是 512，因此报告中必须同时写出 embedding dimension。

## 结果

| Backbone | Classification Val Acc | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 69.67% | 512 | 58.25% | 80.75% | 87.42% | 47.76% |
| Swin-Tiny | 75.50% | 768 | 61.25% | 84.33% | 90.67% | 49.04% |

输出文件位于 `outputs/experiments/backbone_retrieval_comparison/resnet18_vs_swin_tiny/summary.json` 和 `summary.csv`。两个 backbone 都使用 validation split 的 1,200 张图片作为 query/gallery，并排除 query 自身；metric 固定为 L2-normalized cosine similarity。

## 判断标准

本次结果显示 Swin-Tiny 不仅 classification accuracy 更高，retrieval 指标也整体高于 ResNet-18：Recall@1 提升 3.00 percentage points，Recall@5 提升 3.58 percentage points，Recall@10 提升 3.25 percentage points，mAP 提升 1.28 percentage points。结论应保持克制：更强 backbone 在当前 CE 训练下确实改善了 embedding retrieval quality，但 mAP 提升幅度小于 classification accuracy 提升，说明 classification objective 并没有充分优化 embedding 排序结构，后续 SupCon 或 CE+SupCon 仍然有研究必要。

## 维度控制补实验

由于 Swin-Tiny feature 是 768-D，而 ResNet-18 feature 是 512-D，本阶段补充 `Swin 768-D vs PCA-to-512-D` 控制实验。该实验不重新训练，也不重新提取 feature，只复用已保存的 Swin validation embeddings，并固定 query/gallery、排除 query 自身、L2-normalized cosine similarity 和 Recall@K/mAP 计算方式。

| Variant | Projection | Explained Var | Storage MiB | Latency / Query ms | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin 768-D | Original | 100.00% | 3.516 | 0.0533 | 61.25% | 84.33% | 90.67% | 49.04% |
| Swin 512-D | PCA | 99.06% | 2.344 | 0.0521 | 61.67% | 84.75% | 90.50% | 49.22% |

结果说明 Swin 压缩到 512-D 后没有明显损失，mAP 还从 49.04% 轻微上升到 49.22%。因此 Phase 4.2 中 Swin retrieval 优势不能简单解释为“维度更高带来的优势”；Swin feature 在同维度级别下仍然保持主要检索能力。该补实验输出位于 `outputs/experiments/swin_embedding_dimension/20260823_012929_baseline_swin_tiny_protocol/summary.json` 和 `summary.csv`。
