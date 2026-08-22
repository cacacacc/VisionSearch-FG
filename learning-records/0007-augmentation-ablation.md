# 0007：Ablation 只能改变一个核心变量

## 背景

项目进入 augmentation ablation。目标是比较 Basic、Horizontal Flip、RandomResizedCrop、ColorJitter 等训练增强是否改善 CUB validation accuracy。

## 学到什么

真正的 ablation 必须控制变量。比较 augmentation 时，dataset split、backbone、optimizer、learning rate、batch size、epoch 和 evaluation protocol 都应保持一致，只改变 `data.augmentation`。

## 为什么重要

如果同时改变 augmentation 和 learning rate，那么结果变化无法归因于 augmentation。实验表面上是在比较增强，实际可能比较的是优化策略或训练预算。

## 后续行动

依次运行 4 个 ablation 配置，跑完后把每个 run 的 `history.json` 发给我，我会整理正式 ablation 表。
