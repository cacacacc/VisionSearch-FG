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
| Brute Force NumPy | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| FAISS IndexFlatIP | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## Decision

结果待填。预期 FAISS `IndexFlatIP` 与 Brute Force 的 ranking 和 retrieval metrics 完全一致，因为二者都是 exact inner-product search。若一致，后续 Phase 3 可以继续使用 FAISS；若不一致，需要先排查 self-exclusion、normalization 和 score direction。
