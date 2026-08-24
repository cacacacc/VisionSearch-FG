# 实验 5.5：Temperature Ablation

## 实验动机

Experiment 5.2 研究了 SupCon loss 权重 `lambda`，Experiment 5.3 研究了 projection head，Experiment 5.4 研究了 projection embedding dimension。Experiment 5.5 进一步研究 SupCon 中 temperature `tau` 的影响。

在 SupCon 中，temperature 控制 similarity logits 的缩放：

```text
logits = sim(z_i, z_j) / tau
```

较小的 `tau` 会放大相似度差异，使模型更强烈地区分 hard positive 和 hard negative；较大的 `tau` 会让分布更平滑，训练信号更温和。

本实验优先级低于 `lambda`、Projection Head 和 Embedding Dimension Ablation，在算力允许时完成。

## 研究问题

```text
SupCon temperature 如何影响 embedding distribution？
```

具体关注：

- 类内样本是否更紧凑
- 类间样本是否更分离
- retrieval Recall@K 和 mAP 是否提升
- 分类 Accuracy 是否受到影响
- SupCon loss 是否更稳定

## 实验假设

`tau=0.05` 可能让 contrastive objective 更尖锐，增强 hard negative 区分能力，但也可能导致训练不稳定。

`tau=0.07` 是 SupCon / SimCLR 类方法常用默认值，也是前面实验使用的 baseline。

`tau=0.1` 可能带来更平滑的优化过程，在小 batch size 下更稳定。

`tau=0.2` 可能过于平滑，削弱 SupCon 对 embedding geometry 的约束。

## 实验设置

为了控制变量，本实验固定 5.4 中的科研主配置，只改变 temperature。

固定变量：

- Dataset：CUB-200-2011
- Split：固定 train / validation IDs，seed=42
- Backbone：ResNet-18
- Pretraining：ImageNet pretrained
- Fine-tuning：full fine-tuning
- 输入尺寸：224
- Augmentation：random resized crop + horizontal flip + color jitter
- Batch size：16
- Optimizer：AdamW
- CE weight：1.0
- SupCon weight：0.1
- Projection head：MLP
- Projection dim：512
- SupCon feature：projection feature `z`
- Checkpoint selection：best validation accuracy
- Retrieval metric：L2-normalized cosine similarity
- Retrieval feature：projection feature `z`

唯一改变：

```text
tau = 0.05 / 0.07 / 0.1 / 0.2
```

## 配置文件

| Temperature | 配置文件 |
| ---: | --- |
| 0.05 | `configs/phase5_resnet18_ce_supcon_temp0_05.yaml` |
| 0.07 | `configs/phase5_resnet18_ce_supcon_temp0_07.yaml` |
| 0.1 | `configs/phase5_resnet18_ce_supcon_temp0_1.yaml` |
| 0.2 | `configs/phase5_resnet18_ce_supcon_temp0_2.yaml` |

说明：`tau=0.07` 与 5.4 中的 512-D baseline 设置等价。若想节省算力，可以复用 5.4 的 `projection_dim=512` run 作为 `tau=0.07` 对照；若想保持实验目录完全独立，则重新运行 `temp0_07` 配置。

## 训练指令

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_05.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_07.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_1.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_2.yaml `
  --device cuda
```

## 评估指令

本实验研究 projection embedding distribution，因此 retrieval evaluation 必须使用：

```text
--feature projection
```

```powershell
cd D:\code\VisionSearch-FG

$run005 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_05 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_05.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_05\$run005\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_temperature_ablation

$run007 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_07 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_07.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_07\$run007\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_temperature_ablation

$run01 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_1 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_1.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_1\$run01\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_temperature_ablation

$run02 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_2 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_2.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_2\$run02\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_temperature_ablation
```

## 最终实验表

| Tau | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Train SupCon Loss | Query Time ms/query | Run ID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 66.67% | 40.67% | 70.25% | 80.25% | 31.91% | 0.1956 | 0.0552 | `20260823_221843_phase5_resnet18_ce_supcon_temp0_05` |
| 0.07 | 67.00% | 40.83% | 68.92% | 79.75% | 32.07% | 0.2019 | 0.0545 | `20260823_221852_phase5_resnet18_ce_supcon_temp0_07` |
| 0.1 | 68.00% | 43.42% | 72.42% | 82.67% | 34.89% | 0.1939 | 0.0545 | `20260823_222949_phase5_resnet18_ce_supcon_temp0_1` |
| 0.2 | 66.75% | 49.08% | 74.67% | 83.33% | 38.85% | 0.3907 | 0.0543 | `20260823_222957_phase5_resnet18_ce_supcon_temp0_2` |

评估确认：

```text
feature = projection
embedding_dim = 512
metric = cosine
split = validation split from official train
num_samples = 1200
```

## 训练阶段详细结果

| Tau | Run ID | 最优 epoch | Accuracy | Top-5 Acc | Macro-F1 | Train CE Loss | Train SupCon Loss | 训练轮数 | 训练耗时 | Total Params |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | `20260823_221843_phase5_resnet18_ce_supcon_temp0_05` | 7 | 66.67% | 88.58% | 66.51% | 0.1111 | 0.1956 | 12 | 10.39 min | 11,804,424 |
| 0.07 | `20260823_221852_phase5_resnet18_ce_supcon_temp0_07` | 7 | 67.00% | 87.92% | 66.94% | 0.1106 | 0.2019 | 12 | 10.37 min | 11,804,424 |
| 0.1 | `20260823_222949_phase5_resnet18_ce_supcon_temp0_1` | 7 | 68.00% | 88.33% | 67.78% | 0.1120 | 0.1939 | 12 | 10.41 min | 11,804,424 |
| 0.2 | `20260823_222957_phase5_resnet18_ce_supcon_temp0_2` | 7 | 66.75% | 88.50% | 66.80% | 0.1102 | 0.3907 | 12 | 10.41 min | 11,804,424 |

## 结果分析

从分类结果看，`tau=0.1` 当前表现最好，Accuracy 为 68.00%，Macro-F1 为 67.78%，均高于默认的 `tau=0.07`。

`tau=0.07` 的 Accuracy 为 67.00%，是当前第二高结果。`tau=0.05` 和 `tau=0.2` 分别为 66.67% 和 66.75%，说明过低或过高的 temperature 都没有在分类侧带来优势。

从 retrieval 结果看，`tau=0.2` 在所有检索指标上都最优：

```text
Recall@1  = 49.08%
Recall@5  = 74.67%
Recall@10 = 83.33%
mAP       = 38.85%
```

这说明较大的 temperature 虽然没有带来最高分类 Accuracy，但明显改善了 projection embedding 的检索结构。与默认 `tau=0.07` 相比，`tau=0.2` 的提升为：

```text
Recall@1  +8.25 percentage points
Recall@5  +5.75 percentage points
Recall@10 +3.58 percentage points
mAP       +6.78 percentage points
```

`tau=0.1` 是分类和检索之间的折中方案。它的 Accuracy 最高，并且 retrieval 指标也明显优于 `tau=0.05` 和 `tau=0.07`，但仍低于 `tau=0.2`。

从 SupCon loss 看，`tau=0.2` 的 Train SupCon Loss 明显最高，为 0.3907，而其他三个设置都在 0.19 到 0.21 附近。这个数值不能直接解释为“训练更差”，因为 temperature 会改变 SupCon logits 的尺度和 loss 数值范围。更合理的判断是：在较平滑的 contrastive objective 下，embedding distribution 反而更适合 retrieval。

Query Time 在四组之间几乎一致，约 0.054 到 0.055 ms/query。原因是本实验固定 projection_dim=512，检索向量维度相同，因此 temperature 不应显著影响检索耗时。

## 科研结论

本实验显示，temperature 对分类性能和 retrieval representation 的影响并不一致：

```text
分类最优：tau = 0.1
检索最优：tau = 0.2
默认 tau = 0.07 不是当前最优
```

如果目标是 classification + retrieval 的综合平衡，`tau=0.1` 是合理候选。

如果目标是优化 projection embedding 的 retrieval quality，当前应优先选择：

```text
tau = 0.2
```

这个结果也说明，在当前 CUB-200-2011、batch size 16、projection_dim=512 的设置下，过于尖锐的 contrastive logits 并不一定带来更好的检索分布。较大的 temperature 可能缓解小 batch 中 positive / negative 采样不足带来的不稳定性，使 embedding space 更平滑、更适合 Top-K retrieval。

最终建议：

```text
科研主检索配置：tau = 0.2
分类折中配置：tau = 0.1
不建议继续默认固定 tau = 0.07 而不说明原因
```

## 科研意义

Temperature ablation 可以帮助解释 SupCon 对 embedding distribution 的影响：

```text
tau smaller
  -> sharper contrastive logits
  -> stronger local separation pressure

tau larger
  -> smoother contrastive logits
  -> weaker but potentially more stable optimization
```

它补充了 Phase 5 的完整研究链条：

- 5.2：SupCon loss 权重应该多大
- 5.3：projection head 是否必要
- 5.4：projection embedding 维度应该多大
- 5.5：temperature 如何影响 embedding distribution
