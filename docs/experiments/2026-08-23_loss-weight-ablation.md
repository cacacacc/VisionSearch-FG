# 实验 5.2：Loss Weight Ablation

## 实验动机

Experiment 5.1 比较了 CE only、SupCon only 和 CE + SupCon。Experiment 5.2 进一步研究联合损失中 SupCon 权重 `lambda` 的影响。

本实验关注的问题不是“是否使用 SupCon”，而是：

```text
SupCon 应该以多大的权重加入 CE，才能在分类目标和检索表征目标之间取得平衡？
```

## 研究问题

```text
分类目标和 representation learning 目标之间如何平衡？
```

## 实验假设

当 `lambda=0` 时，模型退化为 CE only，分类性能预计较强，但 embedding space 不一定最适合检索。

当 `lambda` 较小时，例如 `0.1` 或 `0.25`，SupCon 可能在不明显破坏分类准确率的情况下改善检索表征。

当 `lambda` 较大时，例如 `0.5` 或 `1.0`，SupCon 对训练目标的影响增强，可能提升 retrieval geometry，但也可能削弱 CE 对分类边界的优化，导致 accuracy 下降。

## 方法

联合损失定义为：

```text
L = L_CE + lambda * L_SupCon
```

本实验固定其他变量，只改变 `lambda`：

| lambda | 配置文件 | 训练目标 |
| ---: | --- | --- |
| 0 | `configs/phase5_resnet18_ce_only.yaml` / lambda0 run | CE only |
| 0.1 | `configs/phase5_resnet18_ce_supcon.yaml` | CE + 0.1 SupCon |
| 0.25 | `configs/phase5_resnet18_ce_supcon_lambda0_25.yaml` | CE + 0.25 SupCon |
| 0.5 | `configs/phase5_resnet18_ce_supcon_lambda0_5.yaml` | CE + 0.5 SupCon |
| 1.0 | `configs/phase5_resnet18_ce_supcon_lambda1_0.yaml` | CE + 1.0 SupCon |

固定变量：

- Backbone：ResNet-18
- 预训练：ImageNet pretrained
- 微调方式：full fine-tuning
- 输入尺寸：224
- Batch size：16
- Temperature：0.07
- 数据增强：random resized crop + horizontal flip + color jitter
- Checkpoint selection：最高 validation accuracy
- 检索评估：validation split，排除 query 自身，使用 L2-normalized cosine similarity

## 最终实验表

| lambda | Accuracy | Recall@5 | Recall@10 | mAP |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 66.33% | 78.67% | 85.75% | 39.24% |
| 0.1 | 67.33% | 78.42% | 85.08% | 39.43% |
| 0.25 | 66.83% | 77.33% | 85.25% | 38.27% |
| 0.5 | 66.17% | 77.83% | 85.17% | 39.27% |
| 1.0 | 64.92% | 74.75% | 84.08% | 37.41% |

这是本项目非常有价值的一张 ablation study 表。它同时展示了分类性能和检索性能随 SupCon 权重变化的趋势。

## 详细结果

| lambda | Run ID | 最优 epoch | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `20260823_093502_phase5_resnet18_ce_supcon_lambda0` | 6 | 66.33% | 52.33% | 78.67% | 85.75% | 39.24% |
| 0.1 | `20260823_093524_phase5_resnet18_ce_supcon_lambda0_1` | 6 | 67.33% | 50.67% | 78.42% | 85.08% | 39.43% |
| 0.25 | `20260823_094907_phase5_resnet18_ce_supcon_lambda0_25` | 7 | 66.83% | 50.50% | 77.33% | 85.25% | 38.27% |
| 0.5 | `20260823_102535_phase5_resnet18_ce_supcon_lambda0_5` | 5 | 66.17% | 51.33% | 77.83% | 85.17% | 39.27% |
| 1.0 | `20260823_102549_phase5_resnet18_ce_supcon_lambda1_0` | 8 | 64.92% | 48.67% | 74.75% | 84.08% | 37.41% |

补充分类指标：

| lambda | Top-5 Accuracy | Macro-F1 | 观察 |
| ---: | ---: | ---: | --- |
| 0 | 88.83% | 65.63% | CE-only 对照组 |
| 0.1 | 88.92% | 66.60% | 当前分类 accuracy 最高 |
| 0.25 | 87.58% | 65.96% | Accuracy 略低于 0.1，但仍高于 lambda=0 |
| 0.5 | 87.75% | 66.06% | Accuracy 下降，Macro-F1 仍接近 0.25 |
| 1.0 | 87.50% | 64.36% | SupCon 权重过大后分类性能明显下降 |

## 重复 run 说明

当前日志中存在重复 run：

| lambda | 其他 run | Accuracy | 说明 |
| ---: | --- | ---: | --- |
| 0.1 | `20260823_091041_phase5_resnet18_ce_supcon_lambda0_1` | 66.17% | Experiment 5.1 中的首次联合训练结果 |
| 0.5 | `20260823_094923_phase5_resnet18_ce_supcon_lambda0_5` | 66.25% | 另一次 lambda=0.5 训练结果 |

上方主表采用较新的 ablation run 作为记录对象。正式论文式报告中应避免只挑最好的一次结果；如果后续要写更严谨的结论，应固定随机种子并对每个 `lambda` 重复运行多次，报告均值和标准差。

## 结果分析

从分类指标看，当前最优 `lambda` 是 `0.1`，Accuracy 达到 67.33%，略高于 `lambda=0` 的 66.33%。这说明小权重 SupCon 没有破坏分类学习，甚至可能起到轻微正则化作用。

从检索指标看，`lambda=0` 的 Recall@5 和 Recall@10 最高，分别为 78.67% 和 85.75%。这说明在当前训练设置下，加入 SupCon 并没有提升 Top-K 检索召回。

`lambda=0.1` 的 mAP 最高，为 39.43%，只比 `lambda=0` 的 39.24% 高 0.19 percentage point。这个提升非常小，不能单独作为强结论；更合理的解释是小权重 SupCon 对排序质量有轻微正则化效果，但当前证据还不充分。

`lambda=0.5` 的 mAP 为 39.27%，也略高于 `lambda=0`，但 Recall@5 和 Recall@10 都低于 `lambda=0`。这说明较大 SupCon 权重并没有稳定改善检索表征。

`lambda=1.0` 在分类和检索上都最弱：Accuracy 为 64.92%，Recall@5 为 74.75%，mAP 为 37.41%。这说明 SupCon 权重过大后，模型既没有保持分类性能，也没有获得更好的检索指标。

当前结果可以概括为：

```text
分类最优：lambda = 0.1
Recall@5 / Recall@10 最优：lambda = 0
mAP 最优：lambda = 0.1，但提升很小
lambda = 1.0 明显不可取
```

## 科研结论

在当前 CUB-200-2011、ResNet-18、batch size 16、temperature 0.07、projection dim 128 的设置下，SupCon 权重消融没有证明“更大的 SupCon 权重会带来更好的 retrieval representation”。

更准确的结论是：

```text
小权重 SupCon 可以保持甚至略微提升分类 accuracy，
但对 Recall@K 和 mAP 的提升并不稳定。
```

因此当前不应继续盲目增大 `lambda`。下一步更有价值的是检查 SupCon 本身的训练条件：

- batch size 是否过小，导致 positive pairs 不足
- temperature 是否需要消融
- two-view augmentation 是否适合细粒度鸟类
- projection head 是否真的有帮助
- SupCon loss 是否应该作用于 projection feature 而不是 backbone feature

这也自然引出 Experiment 5.3：Projection Head Ablation。

## 判断规则

如果只选择分类模型，当前可优先考虑 `lambda=0.1`。

如果只选择 Recall@K 最强的检索模型，当前 `lambda=0` 更稳。

如果希望寻找 classification 与 retrieval 的折中方案，`lambda=0.1` 是当前最合理候选，但它的 retrieval 优势非常有限，后续必须通过 projection head、temperature、batch size 或多 seed 实验进一步验证。
