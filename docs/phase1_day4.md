# Phase 1 Day 4：Baseline 结果分析与错误样本可视化

## 目标

Day 4 的目标是把 baseline 结果从一个总体 accuracy 拆成可检查的样本级证据。

训练完成后，只知道 `val_accuracy=24.70%` 还不够。科研上更重要的问题是：

```text
模型预测错了哪些图？
错得是否很自信？
真实类别是否出现在 top-5 里？
错误是否集中在外观相近的鸟类之间？
```

因此本阶段新增一个分析脚本，用 checkpoint 对验证集做推理，并导出预测明细、汇总指标和高置信错误样本图。

## What：新增什么

新增模块：

```text
src/visionsearch_fg/analysis/predictions.py
```

它负责把一批 logits 转成样本级预测记录：

```text
logits: [batch_size, num_classes]
labels: [batch_size]
-> prediction records
```

每条记录包含：

| 字段 | 含义 |
| --- | --- |
| `image_id` | CUB 原始图像 id |
| `path` | 图像路径 |
| `true_class` | 真实类别名 |
| `pred_class` | top-1 预测类别名 |
| `correct` | top-1 是否正确 |
| `top_k_correct` | 真实类别是否进入 top-k |
| `true_probability` | 模型给真实类别的概率 |
| `pred_probability` | 模型给 top-1 预测的概率 |
| `top_k_classes` | top-k 预测类别 |

新增脚本：

```text
scripts/analyze_baseline_predictions.py
```

它负责加载配置、加载 checkpoint、遍历验证集并写出分析产物。

## Why：为什么要做

分类 baseline 的第一层价值是建立可复现分数；第二层价值是暴露模型行为。

如果只看 accuracy，我们不知道模型是：

```text
大多数类别都略懂一点；
只记住了少数容易类别；
经常把相似类别混淆；
还是对错误预测也非常自信。
```

错误样本分析能帮助决定下一步策略：

| 观察 | 后续方向 |
| --- | --- |
| top-5 高但 top-1 低 | 说明表征有信号，fine-tuning 或分类头优化可能有效 |
| 高置信错误集中在相似鸟类 | 需要局部细节、bbox/crop、Grad-CAM 或更强 backbone |
| 很多错误置信度都低 | 训练还不充分，先延长训练或调整学习率 |
| 某些类别长期完全错误 | 检查类别样本数、标签映射和数据增强 |

## How：如何运行

使用第一个真实冻结 backbone baseline 的 best checkpoint：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_baseline_predictions.py `
  --config configs\baseline_resnet18_frozen.yaml `
  --checkpoint outputs\checkpoints\baseline_resnet18_frozen\20260821_200139_baseline_resnet18_frozen\best.pt `
  --device cpu
```

输出目录：

```text
outputs/figures/baseline_error_analysis/<run_id>/
├── summary.json
├── predictions.json
├── predictions.csv
└── high_confidence_wrong_examples.jpg
```

## Tensor Shape

验证集一个 batch 的核心 shape：

```text
image:  [B, 3, 224, 224]
label:  [B]
logits: [B, 200]
probs:  [B, 200]
top5:   [B, 5]
```

`softmax(logits)` 把每类分数转成概率分布，`topk(k=5)` 找到概率最高的 5 个类别。

## 本阶段判断标准

Day 4 完成后，需要能回答：

```text
这个 baseline 的 top-1、top-5 是否和训练日志一致？
预测明细是否能追溯到原图？
错误样本是否能被人工查看？
下一步优化是训练策略问题，还是模型/表征问题？
```
