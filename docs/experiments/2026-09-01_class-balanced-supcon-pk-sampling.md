# 实验 10.1：Class-balanced SupCon / P×K Sampling

## 实验动机

Phase 5 中 SupCon 没有稳定提升 retrieval，一个关键原因可能不是 loss 本身无效，而是 batch composition 不适合 supervised contrastive learning。

CUB 有 200 个类别。如果随机采样 `batch_size=16`，同一个 batch 内很少出现同类别的不同图片。这样 SupCon 虽然通过 two-view augmentation 至少能拿到同一图片的另一个 view 作为 positive，但它没有充分利用“同一鸟种的不同图片也应该互相靠近”这个监督信号。

P×K sampling 的目标是让每个 batch 固定包含：

```text
P 个类别
每个类别 K 张图片
batch_size = P × K
```

这样每个 anchor 都能看到同类别的不同图片，SupCon 才真正利用 class label 约束类内紧致性。

## 研究问题

```text
在 CE + SupCon 目标不变的情况下，
把随机 batch 改成 class-balanced P×K batch，
是否能提升 retrieval embedding 的 Recall@K 和 mAP？
```

## 对照设计

| 组别 | Config | Sampling | P | K | Batch | 目的 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | `configs/phase5_resnet18_ce_supcon.yaml` | random | - | - | 16 | 原始 CE + SupCon baseline |
| B | `configs/phase5_resnet18_ce_supcon_pk8x2.yaml` | P×K | 8 | 2 | 16 | 更多 negative class，较少 positive |
| C | `configs/phase5_resnet18_ce_supcon_pk4x4.yaml` | P×K | 4 | 4 | 16 | 更多 class 内 positive，较少 negative class |

固定变量：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Backbone：ResNet-18
- Objective：`CE + 0.1 × SupCon`
- Augmentation：`rrc_hflip_colorjitter`
- Projection head：MLP，projection dim = 128
- Temperature：0.07
- Epoch：30
- Batch size：16
- Evaluation：validation retrieval，query/gallery 同为 validation split，并排除 query 自身

唯一主要变化：

```text
sampling.strategy: random -> pk
```

## 实现说明

本实验新增 `PKBatchSampler`。旧配置不写 `data.sampling` 时仍使用原来的随机 shuffle；只有配置中明确写：

```yaml
data:
  sampling:
    strategy: pk
    classes_per_batch: 8
    samples_per_class: 2
```

训练脚本才会启用 P×K batch。

## 训练指令

### B 组：P=8, K=2

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_pk8x2.yaml `
  --device cuda
```

### C 组：P=4, K=4

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_contrastive.py `
  --config configs\phase5_resnet18_ce_supcon_pk4x4.yaml `
  --device cuda
```

如果没有 CUDA，可以把 `--device cuda` 改成 `--device cpu`，但这会比较慢。

## 检索评估指令

### P=8, K=2

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_pk8x2 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_pk8x2.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_pk8x2\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cuda `
  --output-dir outputs\embeddings\phase5_pk_sampling
```

### P=4, K=4

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\phase5_resnet18_ce_supcon_pk4x4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\phase5_resnet18_ce_supcon_pk4x4.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_pk4x4\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature projection `
  --device cuda `
  --output-dir outputs\embeddings\phase5_pk_sampling
```

如果 CUDA 评估不稳定，把 retrieval 指令里的 `--device cuda` 改成 `--device cpu`。

## 结果记录表

| 方法 | Run ID | Sampling | P | K | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE + SupCon random | `20260823_093524_phase5_resnet18_ce_supcon_lambda0_1` | random | - | - | 67.33% | 66.60% | 88.92% | 50.67% | 78.42% | 85.08% | 39.43% |
| CE + SupCon P8K2 | `20260901_124416_phase5_resnet18_ce_supcon_pk8x2` | pk | 8 | 2 | 67.83% | 67.17% | 89.75% | 42.08% | 71.67% | 81.08% | 35.49% |
| CE + SupCon P4K4 | `20260901_152335_phase5_resnet18_ce_supcon_pk4x4` | pk | 4 | 4 | 65.83% | 64.65% | 87.58% | 39.17% | 71.83% | 80.75% | 34.03% |

训练时间记录：

| 方法 | 实际设备 | 完成 epoch | Best epoch | 总训练时间 |
| --- | --- | ---: | ---: | ---: |
| CE + SupCon P8K2 | CPU | 14 | 9 | 2.59 h |
| CE + SupCon P4K4 | CPU | 20 | 15 | 4.47 h |

## 预期解释

如果 `P8K2` 更好，说明 retrieval 更依赖丰富 negative class，多类别区分比类内多 positive 更重要。

如果 `P4K4` 更好，说明当前主要瓶颈是类内不够紧致，同类别不同姿态、背景、尺度的样本需要更强 positive 约束。

如果两者都没有提升，说明 Phase 5 的主要问题可能不只是 batch composition，而是 ResNet-18 backbone、projection dim、temperature、augmentation 强度或 CE/SupCon 权重共同限制了 retrieval geometry。

如果 P×K 在 ResNet-18 上有效，下一步再迁移到更强的 Swin/BBox setting，避免直接在昂贵模型上盲目训练。

## 当前结果解释

P×K sampling 在当前 ResNet-18 CE + SupCon 设置下没有改善 retrieval。`P8K2` 的 Val Acc 从 random baseline 的 67.33% 小幅提高到 67.83%，但 mAP 从 39.43% 下降到 35.49%，Recall@1 从 50.67% 下降到 42.08%。`P4K4` 进一步降低了分类和检索结果。

这说明当前瓶颈不只是 batch 中缺少同类样本。可能原因包括：

- 每个类别训练样本较少，P×K replacement 会让部分 batch 的样本多样性不足。
- `P4K4` 降低了 negative class 数量，使类别间分离信号变弱。
- 当前 SupCon 作用在 128-D projection 上，但最终 retrieval geometry 仍受 ResNet backbone 表达能力限制。
- `rrc_hflip_colorjitter` 对细粒度局部差异可能过强，和 SupCon 的拉近目标存在冲突。

因此，本实验不支持继续在 ResNet-18 上扩大 P×K 搜索。下一步更合理的是把检索优化转向更强的 Swin/BBox 表示，或者在 Swin/BBox 上加入更轻量的 metric learning，而不是继续调 ResNet 的 P×K 组合。
