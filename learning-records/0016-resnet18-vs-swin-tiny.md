# 0016：公平比较不是强迫所有超参数相同

## 背景

Phase 4 开始比较 ResNet-18 与 Swin-Tiny。ResNet 是 CNN backbone，Swin-Tiny 是 Transformer-based backbone，二者结构不同，因此完全相同的 learning rate 未必公平。

## 学到什么

科研上的公平是固定数据、split、输入分辨率、augmentation、训练预算和评价协议，并为每个模型选择合理超参数。强迫 Swin-Tiny 使用只适合 ResNet-18 的 LR，可能反而让比较失真。

## 为什么重要

如果 Swin-Tiny 表现更好，需要确认提升来自 backbone 表达能力，而不是来自不同数据或评价协议。如果 Swin-Tiny 表现不好，也要先检查训练配置是否合理，而不是直接否定 Transformer-based backbone。

## 后续行动

运行 `configs/baseline_swin_tiny_protocol.yaml`，记录 Accuracy、Macro-F1、Top-5 Accuracy、Parameter Count 和 Training Time，并与 ResNet-18 Full FT + HFlip baseline 放入同一张对照表。
