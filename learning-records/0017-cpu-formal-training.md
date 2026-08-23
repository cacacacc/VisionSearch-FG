# 0017：CPU 也可以正式训练，但必须报告计算条件

## 背景

当前环境没有可用 GPU，但用户希望继续用 CPU 跑 Swin-Tiny 正式训练。CPU 训练会非常慢，但只要协议清楚、变量固定、结果完整记录，就可以作为项目内正式实验。

## 学到什么

正式实验不等于必须使用 GPU。正式实验的核心是数据 split、输入分辨率、augmentation、训练预算、评价协议和记录方式可复现。CPU-only formal run 必须明确报告 `device=cpu`、`batch_size`、epoch 数、early stopping、parameter count 和 training time。

## 为什么重要

如果不报告计算条件，Swin-Tiny 的 training time 会被误解。CPU 训练得到的 Accuracy 和 Macro-F1 可以与 ResNet validation 结果比较，但 training time 不能与 GPU 训练结果混合解释。

## 后续行动

使用 `configs/baseline_swin_tiny_cpu_formal.yaml` 跑 Swin-Tiny CPU 正式训练。跑完后，把 `history.json` 和 `metadata.json` 发回，用于填写 ResNet-18 vs Swin-Tiny 正式对照表。
