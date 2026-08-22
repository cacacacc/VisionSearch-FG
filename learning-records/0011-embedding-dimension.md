# 0011：Embedding 维度是性能和成本的权衡

## 背景

Phase 2 已经固定了 retrieval 协议：L2-normalized embedding + cosine similarity。下一步需要观察 embedding dimension 对检索性能、存储量和搜索延迟的影响。

## 学到什么

Embedding dimension 不是越高越绝对正确。高维通常保留更多信息，但存储和相似度计算成本更高；低维更轻量，但可能丢失细粒度视觉差异。为了公平比较，降维实验必须固定原始 embedding、metric、split、query/gallery 协议和指标，只改变维度。

本次实验中，512 维的 Recall@1 / Recall@5 / Recall@10 / mAP 为 58.25% / 80.75% / 87.42% / 47.76%；128 维 PCA 为 57.50% / 80.67% / 87.83% / 47.59%。128 维把 storage 降到 512 维的 25%，但 mAP 只下降 0.17 个百分点，是当前最合理的轻量候选。

## 为什么重要

检索系统最终不仅要看 Recall@K 和 mAP，还要看部署成本。对于大规模图库，embedding storage 和 query-gallery similarity latency 会直接影响系统可用性。维度实验能帮助判断项目应该坚持 512 维，还是可以接受 256/128 维的轻量版本。

## 后续行动

科研主结果继续报告 512 维，保证与原始 CE feature baseline 可比；轻量检索配置可以优先考虑 128 维 PCA。64 维暂时不作为默认配置，因为 Recall@1 和 mAP 损失更明显。
