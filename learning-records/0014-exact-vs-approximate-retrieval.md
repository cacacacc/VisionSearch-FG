# 0014：ANN 的核心是 Accuracy 与 Speed 权衡

## 背景

Phase 3.1 比较了 Brute Force 与 FAISS exact search。Phase 3.2 进入 approximate nearest neighbor search，比较 Flat、IVF、HNSW 等 index。

## 学到什么

Exact search 追求找到真实 top-K nearest neighbors；approximate search 允许候选结果与 exact top-K 有差异，用这种误差换取更低 query time 或更适合大规模图库的 index 结构。Search Recall@K 衡量 ANN 找回 exact top-K 的比例，Retrieval Recall@K 衡量这种近似误差是否真的影响类别检索任务。

本次实验中，Flat Exact 的 Retrieval Recall@1/5/10 为 58.25% / 80.75% / 87.42%。HNSW ef=16 的 Search R@10 为 99.91%，Retrieval Recall@1/5/10 与 Flat Exact 完全一致，同时 query time 从 0.0037 ms/image 降到 0.0021 ms/image。IVF nprobe=1 的 Search R@10 只有 64.59%，Retrieval Recall@1 降到 52.17%，损失过大。

## 为什么重要

Representation 和 search backend 是两个变量。ANN 不能被简单理解为“更准的模型”，它通常不是提升 embedding 质量，而是在系统层面优化检索成本。报告 ANN 结果时必须同时报告 Search Recall、Query Time 和 Memory，否则无法判断 accuracy-speed trade-off 是否值得。

## 后续行动

科研报告继续以 Flat Exact 作为 reference；工程候选可以优先考虑 HNSW ef=16。当前 validation gallery 只有 1,200 张，后续到 official test 或更大 gallery 时需要重新测 latency 和 memory。
