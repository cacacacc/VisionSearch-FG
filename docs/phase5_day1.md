# Phase 5 Day 1：CE vs SupCon vs 联合训练

## 今日目标

Phase 5 开始从“更换 backbone”进入“改进 representation learning objective”。本阶段不再只问模型能不能分类，而是研究不同 loss 如何改变 embedding space，并进一步影响图像检索。

核心问题：

```text
不同训练目标如何影响分类性能与检索表征？
```

## 是什么

本阶段比较三种训练目标：

| 组别 | 训练目标 | 含义 |
| --- | --- | --- |
| A | CE only | 只优化分类边界 |
| B | SupCon only | 只优化同类更近、异类更远的 embedding space |
| C | CE + 0.1 SupCon | 同时优化分类能力和特征空间结构 |

CE 是分类任务最常见的 loss。SupCon 是 supervised contrastive learning，它利用 label 构造正样本对和负样本对：同类别样本互为正样本，不同类别样本互为负样本。

## 为什么需要

此前实验已经说明：分类 accuracy 提升不一定等价于 retrieval mAP 大幅提升。原因是 CE 主要学习 decision boundary，而 retrieval 更依赖 embedding space 的距离结构。

对于 image retrieval，我们希望：

- 同类别鸟类图片在 embedding space 中距离更近
- 不同类别但外观相似的鸟类距离更远
- Top-K neighbor 排序更稳定
- embedding 不只是能被 classifier head 分开，而是本身具有可检索性

所以 Phase 5 引入 SupCon，直接约束 representation。

## 在项目中的位置

本阶段位于整个项目 pipeline 的 representation improvement 部分：

```text
Image -> Backbone Encoder -> Embedding -> Classifier / Retrieval Index
```

CE 主要通过 classifier head 反向影响 encoder。SupCon 则通过 projection head 训练 encoder 的 feature geometry。正式 retrieval evaluation 使用 encoder feature，不使用 projection head feature；projection head 只服务于训练阶段。

## 如何实现

### A 组：CE only

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_only.yaml `
  --device cuda
```

Run ID：

```text
20260823_084541_phase5_resnet18_ce_only
```

### B 组：SupCon only

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --device cuda
```

Run ID：

```text
20260823_084548_phase5_resnet18_supcon_only
```

SupCon only 不直接训练可用的 classifier head，因此需要 linear probe：

```powershell
.\.venv\Scripts\python.exe scripts\train_linear_probe.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_supcon_only\20260823_084548_phase5_resnet18_supcon_only\best.pt `
  --experiment-name phase5_resnet18_supcon_only_linear_probe `
  --device cuda
```

Linear probe run ID：

```text
20260823_091524_phase5_resnet18_supcon_only_linear_probe
```

### C 组：CE + SupCon

```powershell
.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon.yaml `
  --device cuda
```

Run ID：

```text
20260823_091041_phase5_resnet18_ce_supcon_lambda0_1
```

## Tensor shape

对于 batch size 为 `B`、类别数为 `200`、encoder embedding dim 为 `512`、projection dim 为 `128`：

| Tensor | Shape | 作用 |
| --- | --- | --- |
| image | `[B, 3, 224, 224]` | 输入图片 |
| embedding | `[B, 512]` | encoder 输出，用于检索 |
| logits | `[B, 200]` | classifier head 输出，用于 CE |
| projection | `[B, 128]` | projection head 输出，用于 SupCon |

two-view augmentation 会为同一张图片构造两个 view。SupCon 使用这些 view 和 labels 构造正样本关系与负样本关系。

## 评价指标

正式实验需要同时报告：

- Accuracy
- Recall@1
- Recall@5
- Recall@10
- mAP

当前已经完成 classification 和 linear probe。Phase 5 retrieval evaluation 尚未完成，因此 Recall@K 和 mAP 仍为“待评估”。

## 结果汇总

| 组别 | 损失函数 | 最优 epoch | Accuracy | Top-5 Acc | Macro-F1 | Retrieval |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | CE only | 6 | 67.17% | 89.58% | 66.74% | 待评估 |
| B | SupCon only + linear probe | 19 | 43.33% | 74.25% | 41.03% | 待评估 |
| C | CE + 0.1 SupCon | 6 | 66.17% | 89.75% | 65.47% | 待评估 |

CE + SupCon 在 epoch 6 和 epoch 7 都达到 66.17% validation accuracy；由于 checkpoint 保存逻辑只在严格提升时更新，所以正式 best checkpoint 对应 epoch 6。

SupCon-only contrastive training 的 encoder checkpoint 按 minimum train loss 选择，最低 train loss 出现在 epoch 29，train loss 为 0.2693。它训练过程中的 classifier accuracy 不作为正式分类结果，因为 pure SupCon 不优化分类头。

## 结果分析

当前 classification 侧结论是：

```text
CE only > CE + SupCon >> SupCon only linear probe
```

CE only 最强，说明 CE 对 200 类分类边界仍然最直接有效。CE + 0.1 SupCon 只比 CE only 低 1.00 percentage point，说明 SupCon auxiliary loss 没有明显破坏分类能力。SupCon only 的 linear probe accuracy 明显较低，说明当前 SupCon encoder 的线性分类可分性不足。

但这个结果还不能作为 Phase 5.1 的最终科研结论。因为本实验真正关心的是 representation 是否更适合 retrieval，而不是只看分类头 accuracy。下一步必须补齐 Recall@K 和 mAP。

## 取舍分析

CE 的优势是分类效果强、训练稳定、checkpoint selection 清晰。缺点是 embedding space 不一定天然适合 nearest-neighbor retrieval。

SupCon 的优势是直接优化 feature distance，理论上更适合 retrieval。缺点是对 batch size、temperature、positive pair 数量和 augmentation 更敏感；在当前 batch size 16 下，每个 batch 内同类 positive pair 可能不足。

CE + SupCon 的优势是同时保留分类监督和 representation 约束。缺点是需要调 `lambda`，如果 SupCon weight 太小可能没有效果，太大可能损害分类。

## 科研价值

这个实验的价值不在于一次性超过 baseline，而在于把项目从“工程训练模型”推进到“研究训练目标如何改变 representation”。它回答的是科研问题：

```text
分类目标、度量学习目标、多任务目标，哪一种更适合细粒度图像检索？
```

这个问题和 Supervised Contrastive Learning、metric learning、class center learning 都直接相关，也为后续 lambda ablation、embedding visualization 和 qualitative retrieval analysis 提供依据。

## Debug 记录

这次使用 `train_linear_probe.py` 时遇到两个命令行问题：

1. `--checkpoint` 不能保留 `<supcon_run_id>` 占位符。Windows 路径中 `<` 和 `>` 是非法字符，必须替换为真实 run id。
2. `train_linear_probe.py` 必须提供 `--config`。该脚本需要 config 来构建 dataset、model 和 training protocol。

另一个环境问题是：当前执行环境中的 `.venv\Scripts\python.exe` 指向了不存在的 base Python 路径：

```text
C:\Users\admin\AppData\Local\Python\pythoncore-3.10-64
```

因此我没有在当前工具环境里补跑 retrieval evaluation。你本地终端如果可以正常运行 `.venv`，可以直接执行下面的 retrieval commands。

## 下一步任务

补齐三组 retrieval metrics：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_only.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_ce_only\20260823_084541_phase5_resnet18_ce_only\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --device cpu `
  --output-dir outputs\embeddings\phase5_retrieval
```

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_supcon_only.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_supcon_only\20260823_084548_phase5_resnet18_supcon_only\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --device cpu `
  --output-dir outputs\embeddings\phase5_retrieval
```

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon.yaml `
  --checkpoint outputs\checkpoints\phase5_resnet18_ce_supcon_lambda0_1\20260823_091041_phase5_resnet18_ce_supcon_lambda0_1\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --device cpu `
  --output-dir outputs\embeddings\phase5_retrieval
```

判断规则：

```text
如果 CE + SupCon 的 retrieval 指标提升，同时 accuracy 只小幅下降，它就是 Phase 5 的主候选方法。
如果 SupCon only retrieval 提升但 linear probe 较低，说明 retrieval geometry 和 classification boundary 是不同目标。
如果 retrieval 没有提升，则进入 lambda / temperature / batch size / augmentation ablation。
```
