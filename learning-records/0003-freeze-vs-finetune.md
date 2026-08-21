# 0003：冻结 backbone 与 fine-tuning 的区别

## 背景

项目已经完成冻结 backbone 的 ResNet-18 baseline，并开始进入非冻结 backbone 的小规模 probe。

## 学到什么

冻结 backbone 时，只训练分类头，计算成本低，适合先建立稳定 baseline。非冻结 backbone 时，整个 ResNet-18 都参与梯度更新，模型有机会适配 CUB 的细粒度差异，但训练成本更高。

## 为什么重要

fine-tuning 不是简单地“把所有参数打开”。合理做法通常是：

- 先冻结 backbone，确认数据、标签、loss、metrics、checkpoint 全链路正确。
- 再用较小学习率解冻 backbone，避免破坏 ImageNet 预训练特征。
- 用相同验证协议比较 frozen 与 unfrozen，避免只看不可比较的小样本结果。

## 后续复习

复习 `docs/phase1_day5.md`，重点理解 `requires_grad` 如何决定参数是否被 optimizer 更新。
