# Phase 5 Day 1：CE vs SupCon vs Joint Training

## 目标

本阶段开始直接研究训练目标对 classification performance 和 retrieval representation 的影响。Phase 1 到 Phase 4 已经说明 CrossEntropy feature 具备基础检索能力，但 mAP 仍有提升空间；Swin-Tiny 提升了 backbone 表达能力，但 classification accuracy 的提升没有等比例转化为 mAP 提升。因此 Phase 5 的核心问题是：如果直接约束 embedding space 的类内紧致性和类间分离，retrieval 是否会更好。

## Research Question

```text
不同训练目标如何影响分类性能与 Retrieval Representation？
```

## Experiment 5.1 设计

三组实验至少包括 CE only、SupCon only 和 CE + SupCon。为了控制变量，Phase 5.1 先使用 ResNet-18，而不是直接使用 Swin-Tiny；ResNet 更轻，Phase 2 已有完整 CE retrieval baseline，适合判断 SupCon 本身是否有效。三组都使用固定 train/validation split、224 输入、ImageNet pretrained initialization、ResNet-18 full fine-tuning、two-view augmentation、validation retrieval protocol、排除 query 自身和 L2-normalized cosine similarity。

| Group | Config | Loss | Checkpoint Selection | Accuracy Protocol | Retrieval Protocol |
| --- | --- | --- | --- | --- | --- |
| A | `configs/phase5_resnet18_ce_only.yaml` | CE | best val accuracy | classifier head | encoder feature |
| B | `configs/phase5_resnet18_supcon_only.yaml` | SupCon | lowest train loss | linear probe | encoder feature |
| C | `configs/phase5_resnet18_ce_supcon.yaml` | CE + 0.1 SupCon | best val accuracy | classifier head | encoder feature |

纯 SupCon 不直接训练 classifier head，因此它的正式 Accuracy 不能使用 contrastive 训练过程中的随机分类头输出。正确做法是：SupCon encoder 训练完成后，冻结 encoder，重置并训练一个 linear classifier，也就是 linear probe，再报告 validation accuracy。Retrieval 则直接使用 SupCon encoder 的 classifier 前 feature。

## 已完成代码准备

本阶段新增 `SupConLoss`、two-view transform、带 projection head 的 contrastive classifier、`scripts/train_contrastive.py` 和 `scripts/train_linear_probe.py`。`scripts/evaluate_ce_retrieval.py` 已允许在加载 checkpoint 时忽略 `projection_head`，因此 CE+SupCon 或 SupCon checkpoint 可以继续用同一个 retrieval evaluation 脚本提取 encoder feature。

## Smoke Test

已在 CPU 上完成 1 batch smoke test，确认 CE+SupCon 训练可以 forward/backward、保存 checkpoint 和 log；也确认 linear probe 可以加载 contrastive checkpoint、冻结 encoder 并训练分类头。单元测试结果为 `35 passed`。

## 正式训练指令

A 组 CE only：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_only.yaml `
  --device cuda
```

B 组 SupCon only：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --device cuda
```

B 组 linear probe 在 SupCon only 跑完后执行，把 `<supcon_run_id>` 替换成实际 run id：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\train_linear_probe.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_supcon_only\<supcon_run_id>\best.pt `
  --experiment-name phase5_resnet18_supcon_only_linear_probe `
  --device cuda
```

C 组 CE + SupCon：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon.yaml `
  --device cuda
```

如果只能用 CPU，可以把 `--device cuda` 改成 `--device cpu`，并临时加 `--batch-size 4`。正式实验仍应在结果表中记录 device 和 batch size，因为 SupCon 对 batch size 更敏感。

## Retrieval Evaluation

每组训练完成后，用同一个 retrieval 脚本评估 validation split。以 CE+SupCon 为例，把 `<joint_run_id>` 替换成实际 run id：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_ce_supcon_lambda0_1\<joint_run_id>\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --device cpu `
  --output-dir outputs\embeddings\phase5_retrieval
```

## Metrics

正式表格同时报告 Accuracy、Recall@1、Recall@5、Recall@10 和 mAP。CE only 与 CE+SupCon 的 Accuracy 来自训练 checkpoint 的 validation accuracy；SupCon only 的 Accuracy 来自 linear probe。三组 retrieval 都使用 encoder feature，而不是 projection head feature；projection head 只用于训练 SupCon loss。

## Hypothesis

CE only 应该分类较强，但 retrieval 排序仍受限于分类边界目标。SupCon only 可能提升 retrieval，但如果没有 linear probe，不能直接说明分类能力。CE + SupCon 预期取得更好的任务平衡：尽量保留分类 accuracy，同时提升 Recall@K 和 mAP。
