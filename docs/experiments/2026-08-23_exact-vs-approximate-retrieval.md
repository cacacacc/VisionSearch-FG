# Exact vs Approximate Retrieval

## Motivation

本实验比较 FAISS exact search 与 approximate search，建立 ANN 的核心权衡：Accuracy 与 Speed。搜索 index 不改变 embedding 本身，只改变 nearest neighbor search 的实现方式；因此该实验关注的是搜索成本和近似误差，而不是 representation learning。

## Protocol

输入 embedding 固定为 Phase 2 CE Retrieval Baseline 的 512-D feature，统一 L2 normalize 后使用 inner product search。开发阶段使用 validation split，共 1,200 张图片；每张 query 从 gallery 中排除自身。Exact baseline 使用 FAISS `IndexFlatIP`；approximate index 使用 `IndexIVFFlat` 和 `IndexHNSWFlat` 的多个参数配置。

## Command

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

## Result

| Method | Search R@10 | Retrieval R@1 | Retrieval R@5 | Retrieval R@10 | Query ms / Image | Memory MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat Exact | 100.00% | 58.25% | 80.75% | 87.42% | 0.0037 | 2.344 |
| IVF nprobe=1 | 64.59% | 52.17% | 70.92% | 76.42% | 0.0039 | 2.416 |
| IVF nprobe=4 | 92.59% | 58.17% | 80.00% | 86.67% | 0.0042 | 2.416 |
| IVF nprobe=8 | 98.23% | 58.33% | 80.83% | 87.58% | 0.0048 | 2.416 |
| HNSW ef=16 | 99.91% | 58.25% | 80.75% | 87.42% | 0.0021 | 2.655 |
| HNSW ef=32 | 100.00% | 58.25% | 80.75% | 87.42% | 0.0032 | 2.655 |
| HNSW ef=64 | 100.00% | 58.25% | 80.75% | 87.42% | 0.0051 | 2.655 |

## Decision

Flat Exact 仍作为科研指标 reference，因为它定义了 exact top-K。HNSW ef=16 是当前最合理的 approximate 候选：Search R@10 为 99.91%，Retrieval Recall@1/5/10 与 Flat Exact 完全相同，query time 从 0.0037 ms/image 降到 0.0021 ms/image，但 memory 从 2.344 MiB 增加到 2.655 MiB。IVF nprobe=1 不可用，因为 Search R@10 只有 64.59%，Retrieval Recall@1 从 58.25% 降到 52.17%。IVF nprobe=8 的 retrieval recall 略高于 exact 属于 ANN 候选扰动带来的任务指标波动，不应解释为 embedding 变强。
