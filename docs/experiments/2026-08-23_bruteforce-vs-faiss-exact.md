# Brute Force vs FAISS Exact Search

## Motivation

本实验比较 Brute Force NumPy search 与 FAISS exact search。目标是验证搜索算法优化不会改变 embedding 本身，也不应该改变 Recall@K 和 mAP；如果结果一致，就可以把 FAISS 作为后续更大规模检索实验的搜索后端。

## Protocol

输入 embedding 固定为 Phase 2 Day 1 生成的 CE-trained ResNet-18 embedding。开发阶段使用 validation split，共 1,200 张图片；每张图片作为 query 时，从 gallery 中排除自身。所有 embedding 先做 L2 normalization，再使用 inner product search，因此检索语义等价于 cosine similarity。Brute Force 使用 NumPy similarity matrix + argsort；FAISS 使用 `IndexFlatIP` exact search。

## Command

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_bruteforce_faiss.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --latency-repeats 5 `
  --output-dir outputs\experiments\faiss_exact_search\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

## Result

| Method | Indexing ms | Search ms / Query | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brute Force NumPy | 0.000 | 0.0541 | 58.25% | 80.75% | 87.42% | 47.76% |
| FAISS IndexFlatIP | 1.537 | 0.0222 | 58.25% | 80.75% | 87.42% | 47.76% |

## Decision

FAISS `IndexFlatIP` 与 Brute Force NumPy 的 Recall@1、Recall@5、Recall@10 和 mAP 完全一致，`metric_delta` 全部为 0。FAISS 的平均单 query search latency 为 0.0222 ms，低于 Brute Force 的 0.0541 ms；代价是需要 1.537 ms indexing time。`rankings_identical=false` 表示完整排序中存在同分或极小数值差异导致的 tie-order 不完全一致，但不影响检索指标。结论是 FAISS exact search 可以作为后续检索后端，因为它优化搜索成本但不改变 embedding retrieval 评价结果。
