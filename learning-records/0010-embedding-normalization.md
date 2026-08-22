# 0010：Embedding norm 会影响检索几何

## 背景

Phase 2 已经证明 CE-trained ResNet-18 feature 可以作为 retrieval baseline，并且 cosine similarity 优于 euclidean distance。下一步需要单独观察 embedding magnitude 是否会影响检索排序。

## 学到什么

Cosine similarity 本质上会先去除向量长度影响，只比较方向；dot product 则同时包含方向和长度。因此比较 Raw Embedding 和 L2-normalized Embedding 时，严谨做法是比较 Raw Dot Product 与 L2-normalized Cosine，而不是比较两个都会内部归一化的 cosine 分支。

本次实验中，Raw Dot Product 的 Recall@1 / Recall@5 / Recall@10 / mAP 分别为 46.17% / 73.00% / 81.58% / 33.61%，L2-normalized Cosine 分别为 58.25% / 80.75% / 87.42% / 47.76%。L2 normalization 带来了显著提升，尤其是 Recall@1 和 mAP。

## 为什么重要

如果不理解 normalization，就很容易把 metric 的变化误解释成 representation 的变化。检索实验必须明确 feature 是否归一化，否则不同方法之间的 Recall@K 和 mAP 不具备严格可比性。

## 后续行动

后续 retrieval 实验固定使用 L2-normalized embedding 与 cosine similarity。Raw Dot Product 只作为分析 feature norm 影响时的补充实验。
