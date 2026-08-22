# 0013：FAISS exact search 不应改变 retrieval 结果

## 背景

Phase 2 已经固定了 retrieval representation 和 metric：L2-normalized embedding + cosine similarity。Phase 3 开始研究搜索系统本身，第一步是比较 Brute Force 与 FAISS exact search。

## 学到什么

FAISS 不是神经网络训练，`Training Epoch = 0`。它负责把 embedding 放进 index，并执行 nearest neighbor search。对于 L2-normalized embedding，FAISS `IndexFlatIP` 的 inner product search 等价于 cosine similarity exact search；理论上它应该与 Brute Force 得到相同 ranking 和相同 Recall@K / mAP。

## 为什么重要

搜索后端和 embedding representation 是两个变量。Experiment 3.1 的目标不是提升 embedding 质量，而是验证搜索实现是否能降低 search latency 或提供更好的工程接口，同时不改变科研指标。

## 后续行动

运行 `scripts/compare_bruteforce_faiss.py`，在 validation gallery 上比较 Brute Force NumPy 与 FAISS `IndexFlatIP` 的 Recall@K、mAP、indexing time 和 query latency。
