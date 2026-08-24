# 实验 5.4：Embedding Dimension Ablation

## 实验动机

Experiment 5.3 证明了 projection head 对 CE + SupCon joint training 有帮助。Experiment 5.4 进一步研究 projection embedding 的维度应该设置多大。

本实验关注的是：

```text
Representation Capacity
vs
Retrieval Efficiency
```

维度越高，projection space 的表达能力通常越强，但检索时需要更多存储和更多相似度计算。维度越低，检索更轻量，但可能损失细粒度鸟类识别所需的局部差异信息。

## 研究问题

```text
在 CE + SupCon joint training 中，
64 / 128 / 256 / 512 维 projection embedding
会如何影响分类性能、检索质量、存储成本和搜索速度？
```

## 关键定义

本实验中的 Embedding Dimension 指用于 SupCon 和 retrieval evaluation 的 projection feature `z` 维度，而不是 backbone feature `h` 维度。

```text
image
  -> ResNet-18 backbone
  -> backbone feature h: 512-D
  -> projection head
  -> projection feature z: 64 / 128 / 256 / 512-D
```

注意：

```text
评估 5.4 时必须使用 --feature projection。
```

如果不加 `--feature projection`，脚本默认评估 backbone feature `h`，维度会始终是 512-D，无法回答本实验问题。

## 实验假设

`projection_dim=64` 的存储和搜索成本最低，但可能因为表达容量不足导致 Recall@K 和 mAP 下降。

`projection_dim=128` 可能是较好的平衡点：维度足够表达细粒度类别差异，同时检索成本明显低于 512-D。

`projection_dim=256` 和 `projection_dim=512` 可能进一步提升 retrieval quality，但收益可能递减，同时 memory 和 query time 增加。

## 实验设置

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
- Temperature：0.07
- Projection head：MLP
- SupCon feature：projection feature `z`
- Checkpoint selection：best validation accuracy
- Retrieval metric：L2-normalized cosine similarity
- Retrieval split：validation split，排除 query 自身

唯一改变：

```text
projection_dim = 64 / 128 / 256 / 512
```

## 配置文件

| Projection Dim | 配置文件 |
| ---: | --- |
| 64 | `configs/phase5_resnet18_ce_supcon_projection_dim64.yaml` |
| 128 | `configs/phase5_resnet18_ce_supcon_projection_dim128.yaml` |
| 256 | `configs/phase5_resnet18_ce_supcon_projection_dim256.yaml` |
| 512 | `configs/phase5_resnet18_ce_supcon_projection_dim512.yaml` |

## 训练命令

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim64.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim128.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim256.yaml `
  --device cuda

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim512.yaml `
  --device cuda
```

## 评估命令

下面命令会自动读取每个 checkpoint 目录下最新的 run，避免手动把 `<run_id>` 写进命令导致路径错误。

```powershell
cd D:\code\VisionSearch-FG

$run64 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim64 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim64.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim64\$run64\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_embedding_dimension_ablation

$run128 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim128 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim128.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim128\$run128\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_embedding_dimension_ablation

$run256 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim256 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim256.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim256\$run256\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_embedding_dimension_ablation

$run512 = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim512 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_projection_dim512.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_projection_dim512\$run512\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cpu `
  --output-dir outputs\embeddings\phase5_embedding_dimension_ablation
```

每次评估会输出：

```text
summary.json
embeddings.npy
records.csv
top_results.json
```

其中 `summary.json` 包含：

- Accuracy 需要从训练日志 `history.json` / `summary.json` 同步
- Recall@1
- Recall@5
- Recall@10
- mAP
- embedding_dim
- storage_mib_float32
- search_latency_ms_per_query

## 最终实验表

| Projection Dim | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Memory MiB | Query Time ms/query | Run ID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 66.17% | 41.00% | 66.83% | 77.33% | 31.46% | 0.293 | 0.0549 | `20260823_123152_phase5_resnet18_ce_supcon_projection_dim64` |
| 128 | 66.33% | 40.67% | 68.58% | 79.17% | 30.88% | 0.586 | 0.0534 | `20260823_123200_phase5_resnet18_ce_supcon_projection_dim128` |
| 256 | 66.08% | 41.17% | 68.08% | 77.83% | 31.52% | 1.172 | 0.0632 | `20260823_130455_phase5_resnet18_ce_supcon_projection_dim256` |
| 512 | 66.83% | 40.25% | 69.50% | 79.08% | 32.72% | 2.344 | 0.0561 | `20260823_133027_phase5_resnet18_ce_supcon_projection_dim512` |

评估确认：

```text
feature = projection
metric = cosine
split = validation split from official train
num_samples = 1200
```

## 训练阶段详细结果

| Projection Dim | Run ID | 最优 epoch | Accuracy | Top-5 Acc | Macro-F1 | Train SupCon Loss | 训练轮数 | 训练耗时 | Total Params |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | `20260823_123152_phase5_resnet18_ce_supcon_projection_dim64` | 8 | 66.17% | 88.50% | 65.62% | 0.1869 | 13 | 10.03 min | 11,574,600 |
| 128 | `20260823_123200_phase5_resnet18_ce_supcon_projection_dim128` | 6 | 66.33% | 88.58% | 65.83% | 0.1907 | 11 | 8.61 min | 11,607,432 |
| 256 | `20260823_130455_phase5_resnet18_ce_supcon_projection_dim256` | 8 | 66.08% | 88.25% | 65.44% | 0.1984 | 13 | 8.90 min | 11,673,096 |
| 512 | `20260823_133027_phase5_resnet18_ce_supcon_projection_dim512` | 7 | 66.83% | 88.25% | 66.82% | 0.2009 | 12 | 8.25 min | 11,804,424 |

## 结果分析

从分类结果看，`projection_dim=512` 的 Accuracy 最高，为 66.83%，Macro-F1 也最高，为 66.82%。这说明更高维 projection space 在分类侧没有带来明显负面影响，反而略有优势。

从 retrieval 结果看，512-D 的 mAP 最高，为 32.72%，Recall@5 也最高，为 69.50%。这说明更大的 projection dimension 确实带来了更好的整体排序质量。

但是 512-D 并不是所有 Recall 指标都最高。256-D 的 Recall@1 最高，为 41.17%；128-D 的 Recall@10 最高，为 79.17%。这说明当前单 seed 实验中，embedding dimension 与 retrieval accuracy 不是严格单调关系。

从效率看，memory 与维度严格线性增长：

```text
64-D:  0.293 MiB
128-D: 0.586 MiB
256-D: 1.172 MiB
512-D: 2.344 MiB
```

Query Time 在当前 1,200 张 validation gallery 上并不严格随维度增加而上升。128-D 最快，为 0.0534 ms/query；256-D 最慢，为 0.0632 ms/query。原因是当前数据规模较小，排序和 Python / NumPy 调度开销会稀释维度本身带来的计算差异，因此 Query Time 更适合作为小规模验证结果，而不是大规模部署结论。

## 科研结论

本实验回答了 Representation Capacity 与 Retrieval Efficiency 之间的 trade-off。

如果以 retrieval mAP 和分类 Accuracy 作为主要目标，当前最佳选择是：

```text
projection_dim = 512
```

如果更重视效率，128-D 是当前最合理的轻量候选：

```text
projection_dim = 128
```

它的 memory 只有 512-D 的 25%，Recall@5 为 68.58%，只比 512-D 低 0.92 percentage point；Recall@10 为 79.17%，反而略高于 512-D。但它的 mAP 为 30.88%，比 512-D 低 1.84 percentage points，说明前排整体排序质量有所损失。

64-D 虽然最省 memory，但 Recall@5、Recall@10 和 mAP 都不占优，因此不建议作为默认检索维度。

256-D 的 Recall@1 最高，但整体 mAP 仍低于 512-D，说明它对 Top-1 命中有帮助，但排序稳定性不如 512-D。

综合判断：

```text
科研主结果：projection_dim = 512
效率折中方案：projection_dim = 128
不推荐默认使用：projection_dim = 64
```

## 实验意义

该实验可以建立本项目在 joint training setting 下的典型效率权衡：

```text
Embedding Dimension
  -> Representation Capacity
  -> Retrieval Accuracy
  -> Memory / Query Time
```

它补充了 Experiment 5.2 和 Experiment 5.3：

- 5.2 回答 SupCon loss 权重如何影响 CE 与 retrieval 的平衡。
- 5.3 回答 projection head 是否有必要。
- 5.4 回答 projection embedding 应该多大才合适。
