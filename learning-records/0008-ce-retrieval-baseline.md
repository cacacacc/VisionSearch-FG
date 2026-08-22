# 0008：CE 分类特征可以直接作为检索 baseline

## 背景

Phase 1 已经得到 ResNet-18 CrossEntropy 分类模型。Phase 2 的第一步不需要重新训练，而是直接提取 classifier 前的 512-D feature 评估检索能力。

## 学到什么

分类模型的 embedding 可以作为 retrieval baseline。流程是提取 feature、L2 normalize、计算 cosine similarity，并排除 query 自己后计算 Recall@K 和 mAP。

## 为什么重要

这个 baseline 回答一个核心问题：只用分类监督学到的 feature，本身有多少检索能力。后续 SupCon、Swin 或 CE + SupCon 的 retrieval 结果都必须超过它，才说明新增方法确实改善了 embedding space。

## 后续行动

运行 `scripts/evaluate_ce_retrieval.py`，使用当前最佳 ResNet-18 HFlip checkpoint，在 validation split 上得到 CE Retrieval Baseline。
