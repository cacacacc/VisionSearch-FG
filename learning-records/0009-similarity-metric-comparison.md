# 0009：Representation 和 retrieval metric 是两个变量

## 背景

Phase 2 Day 1 已经得到 CE-trained ResNet-18 的 512-D embedding retrieval baseline。下一步需要确认，同一批 embedding 在 cosine similarity 和 euclidean distance 下是否产生不同检索结果。

## 学到什么

检索系统至少包含两个独立环节：模型学到的 representation，以及 retrieval 阶段使用的 metric。为了做公平实验，比较 metric 时必须固定 embedding 文件、数据 split、query/gallery 设置和指标定义，只改变排序依据。

本次实验中，cosine similarity 的 Recall@1 / Recall@5 / Recall@10 / mAP 分别为 58.25% / 80.75% / 87.42% / 47.76%，euclidean distance 分别为 54.33% / 77.67% / 85.42% / 43.69%。Cosine 在所有指标上都更好。

## 为什么重要

如果没有固定 embedding，就无法判断结果变化来自 metric 还是来自重新提取特征时的其他差异。科研实验中的 ablation 要做到“只改变一个变量”，这次实验就是把 retrieval metric 单独拿出来观察。

## 后续行动

后续 retrieval 实验默认使用 cosine similarity，并固定该 metric 比较 SupCon、Swin 或 CE + SupCon。Euclidean distance 只在需要分析 metric sensitivity 时作为补充对照。
