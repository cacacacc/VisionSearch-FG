# 实验 5.3：Projection Head Ablation

## 实验动机

在 SimCLR 和 SupCon 这类 contrastive learning 方法中，projection head 是一个关键设计：训练时通常不是直接对 backbone feature `h` 施加 contrastive loss，而是先通过 projection head 得到 `z`，再在 `z` 上计算 SupCon loss。

这个设计背后的核心思想是：projection head 可以承接 contrastive objective 的优化压力，让 backbone feature `h` 保留更通用、更适合下游分类和检索任务的 representation。

## 研究问题

```text
Projection Head 是否真正改善 Representation Learning？
```

具体比较：

```text
Backbone -> SupCon
vs
Backbone -> Projection Head -> SupCon
```

正式 retrieval evaluation 一律使用 backbone feature `h`，而不是 projection feature `z`。这样才能判断 projection head 是否帮助 backbone 学到更好的通用表征。

## 实验设置

本实验固定其他变量，只改变 SupCon loss 的输入位置。

| 组别 | 结构 | SupCon loss 输入 | Retrieval 使用特征 | 目的 |
| --- | --- | --- | --- | --- |
| A | Backbone -> SupCon | backbone feature `h` | backbone feature `h` | 不使用 projection head |
| B | Backbone -> Projection Head -> SupCon | projection feature `z` | backbone feature `h` | 使用 projection head |

固定变量：

- Dataset：CUB-200-2011
- Split：固定 train / validation IDs
- Backbone：ResNet-18
- 输入尺寸：224
- Augmentation：random resized crop + horizontal flip + color jitter
- Batch size：16
- Optimizer：AdamW
- Temperature：0.07
- CE weight：1.0
- SupCon weight：0.1
- Checkpoint selection：best validation accuracy
- Retrieval metric：L2-normalized cosine similarity

## 训练指令

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_no_projection.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_with_projection.yaml `
  --device cuda
```

## 评估指令

Experiment 5.3 的正式 retrieval evaluation 使用 backbone feature `h`，因此这里使用默认的 `--feature embedding`。

```powershell
cd D:\code\VisionSearch-FG

$runNoProjection = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_no_projection_lambda0_1 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_no_projection.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_no_projection_lambda0_1\$runNoProjection\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\phase5_projection_head_ablation

$runWithProjection = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_with_projection_lambda0_1 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_with_projection.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_with_projection_lambda0_1\$runWithProjection\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\phase5_projection_head_ablation
```

## 工程核查

本实验过程中发现过一个重要工程问题：旧版 `scripts/train_contrastive.py` 固定使用 `output.projection` 计算 SupCon loss，即使配置中写了：

```yaml
training:
  supcon_feature: embedding
```

因此旧 run `20260823_113354_phase5_resnet18_ce_supcon_no_projection_lambda0_1` 不能作为严格 no-projection 结果。

该问题已经修复。训练脚本现在会读取 `training.supcon_feature`，模型也支持 `model.projection_head: identity`。最终 no-projection 使用的有效 run 是：

```text
20260823_120818_phase5_resnet18_ce_supcon_no_projection_lambda0_1
```

metadata 已确认：

```text
projection_head = identity
supcon_feature = embedding
total_parameters = 11,279,112
```

## 最终实验表

| 组别 | Projection Head | SupCon 输入 | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | 无 | `h` | 64.50% | 49.50% | 75.58% | 84.50% | 38.23% |
| B | 有 | `z` | 67.25% | 51.83% | 79.00% | 86.33% | 39.98% |

详细 run 信息：

| 组别 | Run ID | 最优 epoch | Top-5 Acc | Macro-F1 | Train SupCon Loss | 训练轮数 | 训练耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A：no projection | `20260823_120818_phase5_resnet18_ce_supcon_no_projection_lambda0_1` | 7 | 88.75% | 63.37% | 0.2156 | 12 | 8.34 min |
| B：with projection | `20260823_113401_phase5_resnet18_ce_supcon_with_projection_lambda0_1` | 6 | 88.75% | 66.97% | 0.1870 | 11 | 9.45 min |

参数量：

| 组别 | Total Params | 说明 |
| --- | ---: | --- |
| A：no projection | 11,279,112 | 使用 identity projection head，无额外 MLP projection 参数 |
| B：with projection | 11,607,432 | 使用 MLP projection head |

## 结果分析

with projection 在分类和检索上都优于 no projection。

分类侧：

- Accuracy 从 64.50% 提升到 67.25%，提升 2.75 percentage points。
- Macro-F1 从 63.37% 提升到 66.97%，提升 3.60 percentage points。
- Train SupCon Loss 从 0.2156 降到 0.1870，说明 projection space 更容易优化 SupCon objective。

检索侧：

- Recall@1 从 49.50% 提升到 51.83%，提升 2.33 percentage points。
- Recall@5 从 75.58% 提升到 79.00%，提升 3.42 percentage points。
- Recall@10 从 84.50% 提升到 86.33%，提升 1.83 percentage points。
- mAP 从 38.23% 提升到 39.98%，提升 1.75 percentage points。

这个结果说明，在当前 CUB-200-2011、ResNet-18、batch size 16、lambda=0.1、temperature=0.07 的设置下，projection head 不只是论文中的默认结构，而是确实改善了 backbone feature `h` 的下游表现。

## 科研结论

本实验支持 SimCLR / SupCon 类论文中的 projection head 设计思想：

```text
SupCon loss 作用在 projection space z 上，
比直接作用在 backbone feature h 上更有利于学习可迁移表征。
```

更具体地说，projection head 可能承担了 contrastive loss 对局部几何结构的优化压力，使 backbone feature `h` 不被过度扭曲，从而同时改善分类 accuracy 和 retrieval quality。

## 后续方向

当前实验只比较了“有无 projection head”。下一步可以继续做 projection dimension ablation：

```text
projection_dim = 64 / 128 / 256 / 512
```

需要观察 projection dimension 是否影响：

- Accuracy
- Recall@K
- mAP
- 参数量
- 训练稳定性

如果后续算力允许，还应该对 no projection 和 with projection 做多 seed 验证，避免单次 run 的随机性影响结论。
