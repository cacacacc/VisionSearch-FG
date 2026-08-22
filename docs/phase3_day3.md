# Phase 3 Day 3：Retrieval Qualitative Analysis

## 目标

本阶段不只看 Recall@K、mAP、latency 等数字，而是选择若干 query，展示 Query、Top-1、Top-5、Top-10 检索结果，并分析模型到底依赖了什么视觉线索。这个步骤是后续改进 representation 的重要证据，因为数值指标只能说明“是否检索成功”，不能解释“为什么成功或失败”。

## Research Question

```text
CE-trained embedding 的检索结果主要由类别语义、整体外观、背景、姿态还是局部特征驱动？
```

## Method

仍然使用 validation retrieval protocol：

```text
Query = Validation Image
Gallery = Validation Images - Query Itself
Metric = L2-normalized cosine similarity
Training Epoch = 0
```

脚本会自动选取四类案例：Top-1 正确、Top-1 错但 Top-5 有同类、Top-5 错但 Top-10 有同类、Top-10 完全失败。每个案例展示 query 和 top-10，绿色边框表示同类别，红色边框表示类别不同。页面会保留五个分析维度：同类别结果、外观相似但类别不同、背景相似导致错误、姿态相似、局部特征相似。

## 运行指令

```powershell
cd D:\code\VisionSearch-FG
.\.venv\Scripts\python.exe scripts\generate_retrieval_qualitative_report.py `
  --embeddings outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\embeddings.npy `
  --records outputs\embeddings\ce_retrieval\20260822_144519_ablation_resnet18_fullft_aug_hflip\records.csv `
  --cases-per-group 4 `
  --output-dir outputs\experiments\retrieval_qualitative\20260822_144519_ablation_resnet18_fullft_aug_hflip
```

输出文件：

```text
outputs/experiments/retrieval_qualitative/20260822_144519_ablation_resnet18_fullft_aug_hflip/retrieval_qualitative_report.html
outputs/experiments/retrieval_qualitative/20260822_144519_ablation_resnet18_fullft_aug_hflip/retrieval_qualitative_cases.json
outputs/experiments/retrieval_qualitative/20260822_144519_ablation_resnet18_fullft_aug_hflip/retrieval_qualitative_cases.csv
```

本次已生成 16 个案例：Top-1 正确、Top-1 错但 Top-5 有同类、Top-5 错但 Top-10 有同类、Top-10 完全失败四组各 4 个。

## 分析规则

同类别结果用于判断 embedding 是否形成类别簇。外观相似但类别不同通常说明 CE feature 捕捉了整体颜色、体型或纹理，但还不够细粒度。背景相似导致错误说明模型可能过度依赖非鸟体区域。姿态相似导致错误说明模型对 pose invariance 不够稳定。局部特征相似用于观察模型是否关注了头部、喙、翅膀、腹部颜色等真正有判别力的细节。

## 后续使用

如果失败案例集中在背景相似，后续应考虑更强 augmentation、crop、attention 或 foreground-aware 方法。如果失败案例集中在局部细节混淆，后续 SupCon、part-aware representation 或更高分辨率输入更有研究价值。如果同类样本经常出现在 Top-5/Top-10 但不是 Top-1，说明 embedding 已经有语义结构，但 intra-class compactness 和 inter-class margin 仍需优化。
