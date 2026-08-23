# 0018：分类更强不一定代表检索 embedding 更强

## 背景

Phase 4.1 比较 ResNet-18 和 Swin-Tiny 的分类性能。Phase 4.2 进一步比较两个 backbone 的 retrieval embedding，使用完全相同 validation query/gallery protocol。

## 学到什么

Classification accuracy 和 retrieval quality 是相关但不等价的指标。分类训练优化的是 logits 和类别决策边界；retrieval 依赖 embedding space 的几何结构，包括类内紧致性和类间间隔。因此可能出现 Swin Accuracy 高于 ResNet，但 Swin Recall@10 或 mAP 与 ResNet 接近。

## 为什么重要

如果只看 classification accuracy，可能会错误判断 backbone 的 retrieval 价值。Backbone retrieval comparison 能直接回答“更强分类 backbone 是否自然带来更好的检索表征”。这个结果会决定后续是否需要 SupCon、metric learning 或专门的 projection head。

## 后续行动

已完成 Swin-Tiny retrieval evaluation，并用 `scripts/compare_backbone_retrieval.py` 汇总 ResNet 与 Swin 的 retrieval 对照表。结果显示 Swin-Tiny 的 classification Val Acc 为 75.50%，高于 ResNet-18 的 69.67%；retrieval 上 Swin-Tiny 也更好，Recall@1 为 61.25%，Recall@5 为 84.33%，Recall@10 为 90.67%，mAP 为 49.04%。ResNet-18 对应为 Recall@1 58.25%，Recall@5 80.75%，Recall@10 87.42%，mAP 47.76%。

这个结果说明更强 backbone 对 retrieval embedding 有帮助，但提升并不和 classification accuracy 等比例增长。Swin-Tiny 的 Val Acc 提升 5.83 percentage points，而 mAP 只提升 1.28 percentage points，因此后续仍需要 SupCon 或 CE+SupCon 来直接优化 embedding geometry。
