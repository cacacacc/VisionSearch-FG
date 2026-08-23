# 0016：公平比较不是强迫所有超参数相同

## 背景

Phase 4 开始比较 ResNet-18 与 Swin-Tiny。ResNet 是 CNN backbone，Swin-Tiny 是 Transformer-based backbone，二者结构不同，因此完全相同的 learning rate 未必公平。

## 学到什么

科研上的公平是固定数据、split、输入分辨率、augmentation、训练预算和评价协议，并为每个模型选择合理超参数。强迫 Swin-Tiny 使用只适合 ResNet-18 的 LR，可能反而让比较失真。

## 为什么重要

如果 Swin-Tiny 表现更好，需要确认提升来自 backbone 表达能力，而不是来自不同数据或评价协议。如果 Swin-Tiny 表现不好，也要先检查训练配置是否合理，而不是直接否定 Transformer-based backbone。

## 后续行动

已运行 `configs/baseline_swin_tiny_protocol.yaml` 原配置，并将 Swin-Tiny 与 ResNet-18 Full FT + HFlip 放入同一张对照表。Swin-Tiny 在 RTX 4070 上训练，best epoch 为 4，Val Acc 为 75.50%，Val Macro-F1 为 75.28%，best checkpoint 对应 Val Top-5 为 93.42%。相比 ResNet-18 的 69.67% Val Acc 和 89.33% Val Top-5，Swin-Tiny 分类性能更好，但参数量也从 11.28M 增加到 27.67M。

下一步不是直接宣布 Swin 全面优于 ResNet，而是用同一个 validation retrieval protocol 比较 ResNet feature 和 Swin feature。这个实验会回答：classification accuracy 更高的 backbone，是否一定产生更好的 retrieval embedding。
