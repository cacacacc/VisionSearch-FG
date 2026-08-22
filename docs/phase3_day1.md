# Phase 3 Day 1：Brute Force vs FAISS Exact Search

## 目标

Phase 3 不训练神经网络，`Training Epoch = 0`。FAISS 负责把已有 embedding 建成 index，并执行 nearest neighbor search。本阶段比较 Brute Force 和 FAISS Exact Search，在保持 embedding、metric 和检索协议不变的情况下，验证搜索后端优化是否能降低向量搜索成本，同时不改变 retrieval 结果。

## Research Question

```text
在保持检索结果相同的情况下，FAISS 能否降低向量搜索成本？
```

## Controlled Variables

固定项包括 CE-trained ResNet-18 embedding、validation split、query/gallery 协议、排除 query 自身、L2 normalization、inner product search、Recall@K 和 mAP。唯一变化是搜索后端：Brute Force 使用 NumPy 全量 similarity matrix + argsort；FAISS 使用 `IndexFlatIP` 做 exact inner-product search。因为 embedding 已经 L2 normalize，inner product 等价于 cosine similarity。

## Method

开发阶段继续使用 validation gallery，不立即使用 official test：

```text
Query = Validation Image
Gallery = Validation Images - Query Itself
Validation split = data/processed/splits/cub_val_ids_seed42.txt
Current validation size = 1,200
```

运行正式对照实验：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_bruteforce_faiss.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --latency-repeats 5 `
  --output-dir outputs\experiments\faiss_exact_search\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

输出文件：

```text
outputs/experiments/faiss_exact_search/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.json
outputs/experiments/faiss_exact_search/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.csv
```

## Metrics

本实验记录 Recall@1、Recall@5、Recall@10、mAP、indexing time、总 query latency 和平均单 query latency。FAISS exact search 不应该改变检索指标；如果指标变化，优先检查 normalization、metric、self-exclusion 或 ranking 实现是否不一致。

## 待记录结果

| Method | Indexing ms | Search ms / Query | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brute Force NumPy | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| FAISS IndexFlatIP | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 判断标准

如果 `rankings_identical = true` 且 metric delta 全部为 0，说明 FAISS exact search 没有改变 embedding retrieval 结果，只改变搜索实现。若 FAISS latency 更低，可以作为后续检索默认后端；若当前 validation gallery 上优势不明显，也不代表 FAISS 无价值，因为 1,200 张 gallery 太小，大规模图库下 index/search backend 的差异会更明显。
