# CE vs SupCon vs Joint Training

## Motivation

本实验比较 CE only、SupCon only 和 CE + SupCon 三种训练目标对 CUB-200-2011 classification 与 retrieval representation 的影响。此前 CE feature 已经具备基础检索能力，但 mAP 仍不高；Swin-Tiny 提升 backbone 后 mAP 只小幅提高，说明仅靠分类目标不足以充分优化 embedding geometry。

## Protocol

Phase 5.1 先固定 backbone 为 ResNet-18。三组使用相同 train/validation split、224 输入、ImageNet pretrained initialization、full fine-tuning、two-view augmentation、validation retrieval split、排除 query 自身和 L2-normalized cosine similarity。主要变化是训练目标：CE only、SupCon only、CE + 0.1 SupCon。SupCon only 的正式 Accuracy 使用 linear probe：冻结 SupCon encoder，重置并训练一个线性分类头。

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_only.yaml `
  --device cuda
```

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --device cuda
```

```powershell
.\.venv\Scripts\python.exe scripts\train_linear_probe.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_supcon_only\<supcon_run_id>\best.pt `
  --experiment-name phase5_resnet18_supcon_only_linear_probe `
  --device cuda
```

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon.yaml `
  --device cuda
```

## Result

| Group | Loss | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | CE only | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| B | SupCon only | 待跑，linear probe | 待跑 | 待跑 | 待跑 | 待跑 |
| C | CE + 0.1 SupCon | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

## Decision

结果待三组训练和 retrieval evaluation 完成后填写。判断时不能只看 retrieval，也不能只看 accuracy：如果 SupCon only retrieval 高但 linear probe accuracy 低，说明 embedding 对检索有利但分类决策不够直接；如果 CE+SupCon 同时保持 accuracy 并提升 mAP，它将成为 Phase 5 的主候选方法。后续 Experiment 5.2 再做 lambda ablation，例如 0.05、0.1、0.25、0.5。
