# Phase 3 Day 2：Exact vs Approximate Retrieval

## 目标

本阶段比较 FAISS exact search 与 approximate search。Phase 3 仍然 `Training Epoch = 0`，不训练神经网络，只改变向量搜索 index。实验目标是建立典型 ANN trade-off：在允许少量搜索误差的情况下，能否换取更低 query time 或更合理的 index memory。

## Research Question

```text
能够牺牲多少 Retrieval Accuracy 换取多少搜索速度？
```

## Controlled Variables

固定项包括 CE-trained ResNet-18 embedding、validation split、query/gallery 协议、排除 query 自身、L2 normalization、inner product search 和 top-K 评价。唯一变化是 FAISS index：`IndexFlatIP` 是 exact search baseline；`IndexIVFFlat` 和 `IndexHNSWFlat` 是 approximate search。因为 embedding 已经 L2 normalize，inner product 等价于 cosine similarity。

## Method

开发阶段继续使用 validation gallery：

```text
Query = Validation Image
Gallery = Validation Images - Query Itself
Validation size = 1,200
Training Epoch = 0
```

运行正式对照实验：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_faiss_exact_approx.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --latency-repeats 5 `
  --ivf-nlist 32 `
  --hnsw-m 32 `
  --output-dir outputs\experiments\faiss_exact_vs_approx\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

输出文件：

```text
outputs/experiments/faiss_exact_vs_approx/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.json
outputs/experiments/faiss_exact_vs_approx/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.csv
```

## Metrics

本实验记录 Search Recall@1/5/10、Retrieval Recall@1/5/10、Query Time、Indexing Time 和 Index Memory。Search Recall@K 指 ANN top-K 与 exact top-K 的重合比例，直接衡量近似搜索是否找回 exact nearest neighbors；Retrieval Recall@K 衡量近似搜索对类别检索任务的实际影响。这里不记录 mAP，因为 ANN 通常只返回 top-K 候选，而 mAP 需要完整排序。

## 本次结果

| Method | Search R@10 | Retrieval R@1 | Retrieval R@5 | Retrieval R@10 | Query ms / Image | Memory MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat Exact | 100.00% | 58.25% | 80.75% | 87.42% | 0.0037 | 2.344 |
| IVF nprobe=1 | 64.59% | 52.17% | 70.92% | 76.42% | 0.0039 | 2.416 |
| IVF nprobe=4 | 92.59% | 58.17% | 80.00% | 86.67% | 0.0042 | 2.416 |
| IVF nprobe=8 | 98.23% | 58.33% | 80.83% | 87.58% | 0.0048 | 2.416 |
| HNSW ef=16 | 99.91% | 58.25% | 80.75% | 87.42% | 0.0021 | 2.655 |
| HNSW ef=32 | 100.00% | 58.25% | 80.75% | 87.42% | 0.0032 | 2.655 |
| HNSW ef=64 | 100.00% | 58.25% | 80.75% | 87.42% | 0.0051 | 2.655 |

补充：Flat Exact 的 indexing time 为 0.665 ms；IVF nprobe=1/4/8 的 indexing time 分别为 8.570 / 6.890 / 7.303 ms；HNSW ef=16/32/64 的 indexing time 分别为 9.901 / 8.670 / 6.839 ms。当前 gallery 只有 1,200 张，因此绝对 latency 都非常小，不能直接外推到大规模图库。

## 结论

HNSW ef=16 是当前最合理的 approximate 候选：Search R@10 达到 99.91%，Retrieval Recall@1/5/10 与 Flat Exact 完全一致，同时 query time 从 0.0037 ms/image 降到 0.0021 ms/image。HNSW ef=32 和 ef=64 可以达到 100% Search R@10，但速度优势下降，当前规模下没有必要作为默认。IVF nprobe=1 损失过大，不能作为默认；IVF nprobe=8 的 Retrieval Recall 略高于 exact，这是近似候选偶然替换出了同类样本，不能解释为 representation 变好。
