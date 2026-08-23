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

Swin-Tiny retrieval 在 Swin 训练完成后运行。把 `<swin_run_id>` 替换为你的实际 run id：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\baseline_swin_tiny_cpu_formal.yaml `
  --checkpoint outputs\checkpoints\baseline_swin_tiny_cpu_formal\<swin_run_id>\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --batch-size 2 `
  --num-workers 0 `
  --device cpu `
  --output-dir outputs\embeddings\backbone_retrieval
```

生成两个 `summary.json` 后，运行对照表汇总：

```powershell
.\.venv\Scripts\python.exe scripts\compare_backbone_retrieval.py `
  --resnet-summary outputs\embeddings\backbone_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\cosine\summary.json `
  --swin-summary outputs\embeddings\backbone_retrieval\<swin_run_id>\cosine\summary.json `
  --output-dir outputs\experiments\backbone_retrieval_comparison\resnet18_vs_swin_tiny
```

## Metrics

本实验记录 Recall@1、Recall@5、Recall@10 和 mAP。Recall@K 衡量同类图片是否进入 top-K；mAP 衡量同类图片在完整排序中的整体位置。Swin 的 embedding dim 是 768，ResNet 的 embedding dim 是 512，因此报告中必须同时写出 embedding dimension。

## 待记录结果

| Backbone | Classification Val Acc | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 69.67% | 512 | 58.25% | 80.75% | 87.42% | 47.76% |
| Swin-Tiny | 待跑 | 768 | 待跑 | 待跑 | 待跑 | 待跑 |

## 判断标准

如果 Swin-Tiny classification accuracy 更高，但 retrieval Recall@10 或 mAP 没有明显提升，说明分类能力和 embedding retrieval quality 并不完全等价。这个结果本身有研究价值，会支持后续引入 SupCon 或 metric learning。反过来，如果 Swin-Tiny 在 classification 和 retrieval 上都明显更好，说明更强 backbone 同时改善了类别判别和 embedding geometry。
