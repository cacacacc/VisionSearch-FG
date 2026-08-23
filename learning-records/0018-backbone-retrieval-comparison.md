# 0018：分类更强不一定代表检索 embedding 更强

## 背景

Phase 4.1 比较 ResNet-18 和 Swin-Tiny 的分类性能。Phase 4.2 进一步比较两个 backbone 的 retrieval embedding，使用完全相同 validation query/gallery protocol。

## 学到什么

Classification accuracy 和 retrieval quality 是相关但不等价的指标。分类训练优化的是 logits 和类别决策边界；retrieval 依赖 embedding space 的几何结构，包括类内紧致性和类间间隔。因此可能出现 Swin Accuracy 高于 ResNet，但 Swin Recall@10 或 mAP 与 ResNet 接近。

## 为什么重要

如果只看 classification accuracy，可能会错误判断 backbone 的 retrieval 价值。Backbone retrieval comparison 能直接回答“更强分类 backbone 是否自然带来更好的检索表征”。这个结果会决定后续是否需要 SupCon、metric learning 或专门的 projection head。

## 后续行动

Swin-Tiny classification 训练完成后，用 `scripts/evaluate_ce_retrieval.py` 提取 Swin feature 并计算 Recall@1/5/10 和 mAP，再用 `scripts/compare_backbone_retrieval.py` 汇总 ResNet 与 Swin 的 retrieval 对照表。
