# 实验 5.1：CE vs SupCon vs 联合训练

## 实验动机

本实验研究不同训练目标对 CUB-200-2011 细粒度分类性能与检索表征的影响。Phase 1 到 Phase 4 已经说明：CE 训练得到的 ResNet-18 特征具备基础检索能力，但特征空间结构仍有提升空间；Swin-Tiny 提升了分类性能，但检索 mAP 没有等比例提升。

因此 Phase 5 开始直接约束特征空间，观察 SupCon 是否能让同类样本更紧凑、异类样本更分离。

## 研究问题

```text
不同训练目标如何影响分类性能与检索表征？
```

## 实验假设

CE only 预计分类性能更强，因为它直接优化 200 类分类边界。

SupCon only 预计可能产生更适合检索的 embedding，但需要通过 linear probe 才能公平报告分类能力。

CE + SupCon 预计在分类准确率与检索指标之间取得更好的任务平衡。

## 实验方法

本实验固定 backbone、数据划分、输入尺寸、训练协议和检索协议，只改变训练目标。

| 组别 | 训练目标 | 配置文件 | checkpoint 选择 | 分类评估方式 | 检索评估方式 |
| --- | --- | --- | --- | --- | --- |
| A | CE only | `configs/phase5_resnet18_ce_only.yaml` | 最高验证准确率 | classifier head | encoder feature |
| B | SupCon only | `configs/phase5_resnet18_supcon_only.yaml` | 最低训练损失 | frozen encoder + linear probe | encoder feature |
| C | CE + 0.1 SupCon | `configs/phase5_resnet18_ce_supcon.yaml` | 最高验证准确率 | classifier head | encoder feature |

固定变量：

- Backbone：ResNet-18
- 预训练：ImageNet pretrained
- 微调方式：full fine-tuning
- 输入尺寸：224
- Batch size：16
- 数据增强：random resized crop + horizontal flip + color jitter
- 数据划分：固定 CUB train / validation IDs
- 检索距离：L2-normalized cosine similarity
- 检索图库：validation split，排除 query 自身

## 评价指标

正式结果表需要同时报告：

- Accuracy
- Recall@1
- Recall@5
- Recall@10
- mAP

当前已经完成三组训练和 SupCon linear probe；Phase 5 的检索评估尚未生成，因此 Recall@K 和 mAP 暂时记录为“待评估”。

## 实验结果

| 组别 | 损失函数 | Run ID | 最优 epoch | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | CE only | `20260823_084541_phase5_resnet18_ce_only` | 6 | 67.17% | 待评估 | 待评估 | 待评估 | 待评估 |
| B | SupCon only + linear probe | `20260823_091524_phase5_resnet18_supcon_only_linear_probe` | 19 | 43.33% | 待评估 | 待评估 | 待评估 | 待评估 |
| C | CE + 0.1 SupCon | `20260823_091041_phase5_resnet18_ce_supcon_lambda0_1` | 6 | 66.17% | 待评估 | 待评估 | 待评估 | 待评估 |

补充分类指标：

| 组别 | 验证 Top-5 Acc | 验证 Macro-F1 | 观察 |
| --- | ---: | ---: | --- |
| A：CE only | 89.58% | 66.74% | 当前分类 accuracy 最高 |
| B：SupCon linear probe | 74.25% | 41.03% | 线性分类可分性明显弱于 CE |
| C：CE + 0.1 SupCon | 89.75% | 65.47% | Top-5 略高于 CE，Top-1 略低于 CE |

SupCon-only 的 encoder checkpoint 按最低训练损失选择。最低训练损失出现在 epoch 29，train loss 为 0.2693。它训练过程中的 classifier 输出不作为正式分类结果，因为 pure SupCon 不优化分类头。

CE + SupCon 在 epoch 6 和 epoch 7 都达到 66.17% validation accuracy。由于 checkpoint 保存逻辑只在严格提升时更新，正式 `best.pt` 对应第一次达到最高准确率的 epoch 6。

## 结果分析

从当前分类侧结果看：

```text
CE only > CE + SupCon >> SupCon only linear probe
```

CE only 达到最高 Top-1 Accuracy：67.17%。这符合实验假设，因为 CE 直接优化类别决策边界。

CE + 0.1 SupCon 的 Accuracy 为 66.17%，只比 CE only 低 1.00 percentage point。这说明辅助 SupCon loss 没有明显破坏分类能力。它的 Top-5 Accuracy 为 89.75%，略高于 CE only 的 89.58%，说明模型可能仍然保持了较好的类别候选排序能力。

SupCon only 的 linear probe Accuracy 为 43.33%，明显低于 CE。这个结果说明当前 SupCon encoder 对 200 类线性分类并不够友好。但这还不能证明 SupCon 对检索无效，因为 SupCon 的核心目标是优化特征空间几何结构，而不是直接优化 classifier head。

## 当前限制

Phase 5.1 的关键检索指标还没有补齐。没有 Recall@1、Recall@5、Recall@10 和 mAP，就不能最终回答 SupCon 是否改善了检索表征。

下一步需要执行三组检索评估：

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

## 判断规则

如果 CE + SupCon 提升 Recall@K 或 mAP，同时 Accuracy 只比 CE only 小幅下降，那么它就是 Phase 5 当前最好的平衡方案。

如果 SupCon only 的检索指标提升，但 linear probe accuracy 较低，说明检索表征几何结构和分类线性可分性是两个不同目标。

如果 SupCon only 和 CE + SupCon 都没有提升检索指标，那么当前 `lambda=0.1`、batch size 16、temperature 0.07 或 two-view augmentation 可能还不够合适，下一步应做 lambda ablation 或 SupCon 训练协议消融。
