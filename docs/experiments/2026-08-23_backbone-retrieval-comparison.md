# Backbone Retrieval Comparison

## Motivation

本实验比较 ResNet-18 feature 与 Swin-Tiny feature 的 retrieval 能力。核心问题是：classification accuracy 更高的 backbone，是否一定产生更好的 retrieval embedding？如果 Swin Accuracy 高于 ResNet，但 Swin Recall@10 或 mAP 接近 ResNet，这说明分类决策能力和 embedding geometry 不是同一个目标。

## Protocol

两个 backbone 都使用各自 best validation classification checkpoint，提取 classifier 前的 feature。Retrieval protocol 完全一致：validation split 作为 query/gallery，排除 query 自身，使用 L2-normalized cosine similarity，报告 Recall@1、Recall@5、Recall@10 和 mAP。开发阶段不使用 official test。

## Commands

ResNet retrieval summary 可复用 Phase 2 CE Retrieval Baseline，也可以重新生成到 `outputs/embeddings/backbone_retrieval`。Swin-Tiny 需要在 Phase 4.1 训练完成后，用 `baseline_swin_tiny_cpu_formal` 的 `best.pt` 提取 feature。

```powershell
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

```powershell
.\.venv\Scripts\python.exe scripts\compare_backbone_retrieval.py `
  --resnet-summary outputs\embeddings\backbone_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\cosine\summary.json `
  --swin-summary outputs\embeddings\backbone_retrieval\<swin_run_id>\cosine\summary.json `
  --output-dir outputs\experiments\backbone_retrieval_comparison\resnet18_vs_swin_tiny
```

## Result

| Backbone | Classification Val Acc | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 69.67% | 512 | 58.25% | 80.75% | 87.42% | 47.76% |
| Swin-Tiny | 待跑 | 768 | 待跑 | 待跑 | 待跑 | 待跑 |

## Decision

结果待 Swin-Tiny 训练和 retrieval evaluation 完成后填写。判断时不能只看 Swin 是否提高 classification accuracy，还要看 Recall@1 和 mAP 是否同步提升；如果二者不同步，后续需要用 SupCon 或 CE+SupCon 直接优化 embedding space。
