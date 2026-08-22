# Retrieval Qualitative Analysis

## Motivation

本实验补充 Phase 2 和 Phase 3 的数值指标。Retrieval 结果不能只看 Recall@K 或 latency，还需要观察 query 与 top-k 图片之间的视觉关系，判断模型是按类别语义检索，还是被背景、姿态、颜色、局部纹理等因素干扰。

## Protocol

输入 embedding 固定为 Phase 2 CE Retrieval Baseline 的 512-D feature，使用 L2-normalized cosine similarity。开发阶段使用 validation split，共 1,200 张图片；每张 query 从 gallery 中排除自身。脚本自动选取 Top-1 正确、Top-1 错但 Top-5 有同类、Top-5 错但 Top-10 有同类、Top-10 完全失败四组案例。

## Command

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\generate_retrieval_qualitative_report.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --cases-per-group 4 `
  --output-dir outputs\experiments\retrieval_qualitative\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

## Output

```text
outputs/experiments/retrieval_qualitative/20260822_144519_ablation_resnet18_fullft_aug_hflip/retrieval_qualitative_report.html
```

本次已生成 16 个案例，四组各 4 个：

```text
Top-1 正确：4
Top-1 错误，但 Top-5 有同类：4
Top-5 错误，但 Top-10 有同类：4
Top-10 完全失败：4
```

## Analysis Checklist

| 观察维度 | 需要记录的问题 |
| --- | --- |
| 同类别结果 | 同类样本是否集中在 Top-1、Top-5 或 Top-10？ |
| 外观相似但类别不同 | 错误结果是否在颜色、体型、纹理上与 query 接近？ |
| 背景相似导致错误 | 水面、树枝、天空、草地等背景是否主导了相似度？ |
| 姿态相似 | 飞行、侧身、站立、低头等姿态是否压过类别信息？ |
| 局部特征相似 | 头部、喙、翅膀、腹部颜色等细节是否被正确利用？ |

## Decision

已生成 HTML 报告，下一步需要人工逐例查看并填写视觉失败模式。该分析将决定后续 representation 改进方向：如果背景干扰强，优先考虑 foreground/crop/attention；如果局部特征混淆强，优先考虑更高分辨率、part-aware feature 或 SupCon；如果同类经常在 Top-5/Top-10 但不在 Top-1，说明需要增强 intra-class compactness 和 inter-class margin。
