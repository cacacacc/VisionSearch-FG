# Phase 2 Day 3：Embedding Normalization

## 目标

本阶段比较 Raw Embedding 和 L2-normalized Embedding 对 retrieval 的影响。这里不重新训练模型，也不重新提取 feature，而是复用 Phase 2 Day 1 保存的同一批 `embeddings.npy`。唯一变化是 retrieval 前是否对 embedding 做 L2 normalization，因此这是一个关于 embedding geometry 的受控实验。

## Research Question

```text
去除 embedding magnitude 的影响是否改善 retrieval？
```

## 重要定义

严格来说，cosine similarity 本身就是 normalized dot product，因此如果两个分支都使用真正的 cosine similarity，Raw Embedding 和 L2-normalized Embedding 的排序会完全相同。为了让实验有科研意义，本实验把两组定义为：Raw Embedding 使用 dot-product similarity，保留 magnitude 影响；L2-normalized Embedding 使用 normalized dot-product similarity，等价于 cosine similarity，去除 magnitude 影响。

## Controlled Variables

固定项包括 checkpoint、embedding 文件、validation split、query/gallery 集合、排除 query 自身的协议、Recall@K 和 mAP 的计算方式。唯一变化是相似度计算前是否做 L2 normalization。

## Method

输入文件沿用 CE Retrieval Baseline：

```text
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/embeddings.npy
outputs/embeddings/ce_retrieval/20260822_144519_ablation_resnet18_fullft_aug_hflip/records.csv
```

运行正式对照实验：

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_embedding_normalization.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --output-dir outputs\experiments\embedding_normalization\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

输出文件：

```text
outputs/experiments/embedding_normalization/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.json
outputs/experiments/embedding_normalization/20260822_144519_ablation_resnet18_fullft_aug_hflip/summary.csv
```

## 本次结果

| Variant | Samples | Embedding Dim | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Dot Product | 1,200 | 512 | 46.17% | 73.00% | 81.58% | 33.61% |
| L2-normalized Cosine | 1,200 | 512 | 58.25% | 80.75% | 87.42% | 47.76% |

L2-normalized Cosine 相对 Raw Dot Product 的提升为：Recall@1 +12.08 个百分点，Recall@5 +7.75 个百分点，Recall@10 +5.83 个百分点，mAP +14.15 个百分点。

## 结论

L2-normalized Cosine 明显优于 Raw Dot Product，说明当前 CE-trained ResNet-18 embedding 的 magnitude 没有稳定帮助检索，反而显著干扰排序。后续 retrieval 实验应默认使用 L2-normalized embedding 与 cosine similarity，并在实验协议中固定这一点；Raw Dot Product 不应作为默认检索 metric。
