# 0019：Backbone 对比中需要控制 embedding dimension

## 背景

Phase 4.2 比较 ResNet-18 与 Swin-Tiny retrieval embedding 时，两个 backbone 的默认 feature dimension 不同：ResNet-18 是 512-D，Swin-Tiny 是 768-D。直接比较是合理的，因为它反映真实 backbone pipeline；但如果要更严格地说明 Swin 优势来自 representation，而不是仅来自更高维度，需要补充同维度控制实验。

## 学到什么

使用 PCA 将 Swin 768-D feature 压缩到 512-D 后，retrieval 没有明显下降。Swin 768-D 的 Recall@1、Recall@5、Recall@10、mAP 分别为 61.25%、84.33%、90.67%、49.04%；Swin 512-D PCA 的对应结果为 61.67%、84.75%、90.50%、49.22%。PCA 512-D 保留 99.06% explained variance，并将 float32 embedding storage 从 3.516 MiB 降到 2.344 MiB。

## 为什么重要

这个补实验说明 Swin 在 Phase 4.2 中优于 ResNet，不能简单归因于 Swin 使用了 768-D 更高维 feature。即使压缩到 512-D，Swin retrieval 仍然保持主要性能，并且 mAP 略有提升。这为后续论文表述提供了更严谨的控制变量证据。

## 后续行动

Phase 4 的 backbone comparison 可以报告两个层次：默认 pipeline 对比使用 ResNet 512-D vs Swin 768-D；控制维度对比补充 Swin PCA-to-512-D。下一阶段进入 SupCon 或 CE+SupCon 时，也应明确 embedding dimension，避免把 representation learning 的收益和维度变化混在一起解释。
