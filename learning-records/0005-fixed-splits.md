# 0005：固定 split 是正式对照实验的前置条件

## 背景

全局训练协议已经确定：official test 不能用于调参，所有模型选择应基于 official train 内部分出的 validation。

## 学到什么

正式实验需要先生成固定 split 文件：

```text
cub_train_ids_seed42.txt
cub_val_ids_seed42.txt
cub_test_ids.txt
```

训练脚本再通过这些 image id 文件筛选样本。

## 为什么重要

如果 frozen、fine-tuning、Swin、SupCon 使用不同的 train/validation 划分，那么结果差异可能来自数据划分，而不是模型或方法。

## 后续行动

用 `configs/baseline_resnet18_frozen_protocol.yaml` 和 `configs/baseline_resnet18_unfrozen_protocol.yaml` 跑正式对照实验。
