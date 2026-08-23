# Swin-Tiny Embedding Dimension Control

## Motivation

Backbone retrieval comparison 中，ResNet-18 feature 是 512-D，而 Swin-Tiny feature 是 768-D。虽然这代表两个 backbone 的默认实际 pipeline，但如果要更严格地判断 Swin 的优势是否主要来自 backbone 表达能力，而不是更高 embedding dimension，需要补充一个小控制实验：将 Swin 768-D feature 压缩到 512-D，再执行完全相同 retrieval protocol。

## Protocol

本实验不重新训练模型，也不重新提取 feature。输入固定为 Phase 4.2 已保存的 Swin-Tiny validation embedding：`outputs/embeddings/backbone_retrieval/20260823_012929_baseline_swin_tiny_protocol/cosine/embeddings.npy`。query/gallery 固定为 validation split 的 1,200 张图片，检索时排除 query 自身，metric 固定为 L2-normalized cosine similarity。唯一变化是 embedding dimension：原始 768-D 对比 PCA 压缩后的 512-D。

## Command

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\compare_embedding_dimensions.py `
  --embeddings outputs\embeddings\backbone_retrieval\20260823_012929_baseline_swin_tiny_protocol\cosine\embeddings.npy `
  --records outputs\embeddings\backbone_retrieval\20260823_012929_baseline_swin_tiny_protocol\cosine\records.csv `
  --dimensions 768 512 `
  --latency-repeats 5 `
  --output-dir outputs\experiments\swin_embedding_dimension\20260823_012929_baseline_swin_tiny_protocol
```

## Result

| Variant | Projection | Explained Var | Storage MiB | Latency / Query ms | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin 768-D | Original | 100.00% | 3.516 | 0.0533 | 61.25% | 84.33% | 90.67% | 49.04% |
| Swin 512-D | PCA | 99.06% | 2.344 | 0.0521 | 61.67% | 84.75% | 90.50% | 49.22% |

输出文件位于 `outputs/experiments/swin_embedding_dimension/20260823_012929_baseline_swin_tiny_protocol/summary.json` 和 `summary.csv`。

## Decision

Swin-Tiny 压缩到 512-D 后没有出现 retrieval 退化：Recall@1 提升 0.42 percentage points，Recall@5 提升 0.42 percentage points，Recall@10 下降 0.17 percentage points，mAP 提升 0.19 percentage points。这个结果说明 Phase 4.2 中 Swin 优于 ResNet 的结论不能简单归因于 768-D 维度更高；在相同 512-D 级别下，Swin feature 仍然保留了主要检索能力。512-D PCA 同时把 float32 embedding storage 从 3.516 MiB 降到 2.344 MiB，下降约 33.33%，因此后续如果需要和 ResNet 做同维度比较，可以报告 Swin 512-D PCA 作为补充结果。
