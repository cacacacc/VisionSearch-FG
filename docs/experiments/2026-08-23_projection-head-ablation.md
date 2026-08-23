# 实验 5.3：Projection Head Ablation

## 实验动机

在 SimCLR 和 SupCon 这类 contrastive learning 方法中，projection head 是一个非常关键的设计：训练时通常不是直接对 backbone feature `h` 施加 contrastive loss，而是先通过一个 projection head 得到 `z`，再在 `z` 上计算对比学习损失。

这个设计背后的核心思想是：projection head 可以承接 contrastive objective 的优化压力，让 backbone feature `h` 保留更通用、更适合下游任务的 representation。

因此本实验要验证：

```text
Projection Head 是否真的改善了 Representation Learning？
```

## 研究问题

```text
在 SupCon / CE + SupCon 训练中，是否需要 Projection Head？
```

更具体地说：

```text
Backbone -> SupCon
vs
Backbone -> Projection Head -> SupCon
```

哪一种能得到更好的分类性能和检索表征？

## 实验假设

如果 projection head 有效，那么：

- 使用 projection head 训练时，encoder feature `h` 的检索指标应该更好。
- 分类 accuracy 不应明显下降。
- SupCon loss 可以在 projection space `z` 中优化局部几何结构，而不直接扭曲 backbone feature `h`。

如果 projection head 无效，那么：

- `Backbone -> SupCon` 和 `Backbone -> Projection Head -> SupCon` 的 Recall@K / mAP 差异应该很小。
- 甚至直接在 backbone feature 上计算 SupCon 可能更简单、更稳定。

## 方法

本实验固定其他训练变量，只改变 SupCon loss 的输入位置。

| 组别 | 结构 | SupCon loss 输入 | Retrieval 使用特征 | 目的 |
| --- | --- | --- | --- | --- |
| A | Backbone -> SupCon | backbone feature `h` | backbone feature `h` | 不使用 projection head |
| B | Backbone -> Projection Head -> SupCon | projection feature `z` | backbone feature `h` | 使用 projection head |

两个分支的区别：

```text
A 组：
Image -> Backbone -> h -> SupCon
                     |
                     +-> Classifier -> CE

B 组：
Image -> Backbone -> h -> Projection Head -> z -> SupCon
                     |
                     +-> Classifier -> CE
```

注意：正式 retrieval evaluation 一律使用 backbone feature `h`，而不是 projection feature `z`。这是为了回答 projection head 是否帮助 backbone 学到更好的通用 representation。

## 固定变量

- Dataset：CUB-200-2011
- Split：固定 train / validation IDs
- Backbone：ResNet-18
- 输入尺寸：224
- Augmentation：random resized crop + horizontal flip + color jitter
- Batch size：16
- Optimizer：AdamW
- Temperature：0.07
- CE weight：1.0
- SupCon weight：使用 Experiment 5.2 选出的最佳 `lambda`
- Checkpoint selection：best validation accuracy
- Retrieval metric：L2-normalized cosine similarity

如果 Experiment 5.2 的 retrieval 指标尚未补齐，临时可以先使用当前分类侧最优的 `lambda=0.1` 作为默认值。

## 评价指标

正式结果表需要报告：

- Accuracy
- Recall@1
- Recall@5
- Recall@10
- mAP
- Train SupCon Loss
- 参数量

## 结果表

当前该实验尚未运行，表格先作为正式记录模板。

| 组别 | Projection Head | SupCon 输入 | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | 无 | `h` | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |
| B | 有 | `z` | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |

补充工程指标：

| 组别 | 额外参数量 | 训练耗时 | 显存占用 | 观察 |
| --- | ---: | ---: | ---: | --- |
| A：无 projection head | 待统计 | 待统计 | 待统计 | 结构更简单 |
| B：有 projection head | 待统计 | 待统计 | 待统计 | 更符合 SimCLR / SupCon 设计 |

## 当前工程状态

当前代码中的 `ContrastiveClassifier` 默认总是创建 projection head：

```text
embedding h -> projection_head -> projection z
```

训练脚本 `scripts/train_contrastive.py` 当前也默认使用：

```text
output.projection
```

来计算 SupCon loss。

因此，要真正运行 Experiment 5.3，需要先做一个小的工程改动：让配置文件可以控制 SupCon loss 使用 `embedding` 还是 `projection`。

建议新增配置项：

```yaml
model:
  projection_head: mlp   # 可选：mlp / identity

training:
  supcon_feature: projection  # 可选：projection / embedding
```

对应两组实验：

```yaml
# A 组：Backbone -> SupCon
model:
  projection_head: identity
training:
  supcon_feature: embedding
```

```yaml
# B 组：Backbone -> Projection Head -> SupCon
model:
  projection_head: mlp
training:
  supcon_feature: projection
```

## 推荐配置文件

建议新增：

```text
configs/phase5_resnet18_ce_supcon_no_projection.yaml
configs/phase5_resnet18_ce_supcon_with_projection.yaml
```

其中 with projection 可以复用当前 CE + SupCon 配置；no projection 需要把 SupCon loss 改为直接作用在 encoder embedding `h` 上。

## 预期结果解释

如果 B 组优于 A 组：

```text
Projection Head 对 SupCon 有帮助。
它可能让 contrastive loss 主要塑造 projection space，
同时保护 backbone feature h 的通用性。
```

这将支持 SimCLR / SupCon 论文中的设计思想。

如果 A 组优于 B 组：

```text
直接优化 backbone feature 更适合当前小数据集和轻量模型。
Projection Head 可能引入了额外优化难度或信息瓶颈。
```

这说明论文中的设计不一定在当前 CUB + ResNet-18 + 小 batch 设置下最优。

如果两组接近：

```text
Projection Head 不是当前性能瓶颈。
后续应优先关注 batch size、lambda、temperature、augmentation 或 backbone。
```

## 科研价值

这个实验直接对应 SimCLR / SupCon 类论文中的核心设计问题：

```text
为什么 contrastive loss 通常作用在 projection head 输出上，
而不是直接作用在 backbone representation 上？
```

它能帮助项目从“照搬论文结构”进入“验证论文设计是否适合自己的任务”。对于科研项目来说，这比单纯使用 SupCon 更有说服力。

## 下一步

1. 在模型和训练脚本中增加 `supcon_feature` 配置。
2. 新增 no-projection 和 with-projection 两个配置文件。
3. 使用 Experiment 5.2 选出的最佳 `lambda` 训练两组模型。
4. 在 validation split 上评估 Accuracy、Recall@K 和 mAP。
5. 如果结果稳定，再对最终候选模型做 official test 评估。
