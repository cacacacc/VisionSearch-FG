# Phase 0 Day 1：项目初始化

## Motivation：为什么做

科研项目不是只写一个训练脚本，而是要能持续迭代、复现实验、比较方法并解释结果。因此第一天先搭建项目结构。

## 为什么需要这些模块

| 模块 | 为什么存在 | Pipeline 位置 |
| --- | --- | --- |
| `configs/` | 固化实验设置，避免每次手改代码造成实验不可复现 | 训练、验证、检索前读取 |
| `data/` | 放原始数据和预处理结果，但不提交大文件 | Dataset / DataLoader 输入 |
| `src/visionsearch_fg/data/` | 负责 CUB 数据读取、划分、transform | 输入 image 和 label |
| `src/visionsearch_fg/models/` | 负责 backbone、classification head、embedding head | image tensor -> logits / embedding |
| `src/visionsearch_fg/engine/` | 负责 train / validate loop | batch -> loss / metrics |
| `src/visionsearch_fg/retrieval/` | 负责 embedding index 和 Recall@K / mAP | embedding -> nearest neighbors |
| `src/visionsearch_fg/explainability/` | 负责 Grad-CAM、attention、t-SNE | model + image / embedding -> visualization |
| `scripts/` | 放可执行入口，调用 `src` 中的可复用模块 | 命令行运行实验 |
| `outputs/` | 保存实验产物，不提交到 Git | checkpoints / logs / figures |
| `tests/` | 检查关键模块行为，降低后续重构风险 | 开发过程 |

## 第一个 Baseline 的 Tensor 流

Phase 1 的 ResNet-18 baseline 会遵循下面的数据流：

```text
image file 图片文件
-> PIL image
-> transform
-> image tensor: [B, 3, 224, 224]
-> ResNet-18 backbone
-> feature: [B, 512]
-> classifier
-> logits: [B, 200]
-> cross entropy loss
```

## 当前决策

使用 ResNet-18 作为第一个 baseline，因为它计算量低、迁移学习成熟、便于在轻薄本上验证完整 pipeline。

暂时不直接上 ViT / Swin，因为 Transformer 对数据量和算力更敏感，适合在 baseline 稳定后作为对比实验。
