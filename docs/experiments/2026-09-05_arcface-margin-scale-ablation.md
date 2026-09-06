# 实验 11.3：ArcFace Margin / Scale Ablation

## 实验动机

进行本实验前，当前最强结果来自：

```text
ConvNeXt V2 Tiny BBox448 + ArcFace + Flip TTA + Query Expansion
```

其中基础 ArcFace 的 mAP 已经达到 `81.33%`，后处理后达到 `82.81%`。这说明 angular margin objective 非常有效。

但是，如果要把这个结果写成科研结论，还需要回答一个问题：

```text
ArcFace 的提升是否依赖某个偶然超参数？
```

ArcFace 主要有两个关键超参数：

- `margin`：控制类间角度间隔，margin 越大，分类边界越严格。
- `scale`：控制 softmax logits 的温度/锐度，scale 越大，分类分布越尖锐。

## Research Question

```text
ArcFace margin 和 scale 如何影响 classification performance 与 retrieval embedding quality？
```

## 实验设计

固定变量：

- Dataset：CUB-200-2011
- Split：`cub_train_ids_seed42.txt` / `cub_val_ids_seed42.txt`
- Backbone：ConvNeXt V2 Tiny
- Pretraining：ImageNet-22K/1K fine-tuned timm checkpoint
- Input：BBox crop 448，margin = 0.15
- Augmentation：Horizontal Flip
- Optimizer：AdamW
- Backbone LR：`5e-5`
- Margin Head LR：`5e-4`
- Epoch：30
- Early stopping：5
- Retrieval metric：cosine

变量：

| 组别 | Config | Margin | Scale | 目的 |
| --- | --- | ---: | ---: | --- |
| A | `configs/foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30.yaml` | 0.3 | 30 | 较弱 angular constraint |
| B | `configs/foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30.yaml` | 0.4 | 30 | 中等偏弱 margin |
| C | `configs/foreground_convnextv2_tiny_bbox448_arcface.yaml` | 0.5 | 30 | 当前最强 baseline |
| D | `configs/foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30.yaml` | 0.6 | 30 | 更强 angular constraint |
| E | `configs/foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml` | 0.5 | 16 | 较低 scale |
| F | `configs/foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64.yaml` | 0.5 | 64 | 较高 scale |

## 运行指令

### 1. 一次性训练所有新增 ablation

当前 `m=0.5, scale=30` 已经训练完成，因此这里只训练新增的 5 组。

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

$experiments = @(
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64.yaml" }
)

foreach ($exp in $experiments) {
  Write-Host "Training $($exp.Name)" -ForegroundColor Cyan
  .\.venv\Scripts\python.exe scripts\train_margin_classifier.py `
    --config $exp.Config `
    --device cuda
}
```

### 2. 一次性评估基础 retrieval

```powershell
cd D:\code\VisionSearch-FG

$experiments = @(
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64.yaml" }
)

foreach ($exp in $experiments) {
  $run = Get-ChildItem "outputs\checkpoints\$($exp.Name)" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty Name

  Write-Host "Evaluating retrieval $($exp.Name) / $run" -ForegroundColor Cyan
  .\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
    --config $exp.Config `
    --checkpoint "outputs\checkpoints\$($exp.Name)\$run\best.pt" `
    --split train `
    --ids-path data\processed\splits\cub_val_ids_seed42.txt `
    --metric cosine `
    --feature embedding `
    --device cuda `
    --output-dir outputs\embeddings\angular_margin_ablation
}
```

### 3. 一次性评估 Flip TTA + Query Expansion

```powershell
cd D:\code\VisionSearch-FG

$experiments = @(
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml" },
  @{ Name = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64"; Config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64.yaml" }
)

foreach ($exp in $experiments) {
  $run = Get-ChildItem "outputs\checkpoints\$($exp.Name)" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty Name

  Write-Host "Post-processing $($exp.Name) / $run" -ForegroundColor Cyan
  .\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
    --config $exp.Config `
    --checkpoint "outputs\checkpoints\$($exp.Name)\$run\best.pt" `
    --split train `
    --ids-path data\processed\splits\cub_val_ids_seed42.txt `
    --feature embedding `
    --views bbox bbox_flip `
    --qe-top-k 0 3 5 10 `
    --qe-alpha 0.0 0.1 0.2 0.5 `
    --device cuda `
    --output-dir outputs\embeddings\arcface_margin_scale_ablation_postprocessing
}
```

## 结果记录表

| Margin | Scale | Run ID | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP | 最优后处理 mAP |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 30 | `20260905_123114_foreground_convnextv2_tiny_bbox448_arcface_m0_3_s30` | 87.17% | 87.14% | 96.75% | 83.58% | 93.58% | 96.00% | 78.85% | 80.11% |
| 0.4 | 30 | `20260905_130442_foreground_convnextv2_tiny_bbox448_arcface_m0_4_s30` | 86.92% | 86.70% | 96.42% | 84.67% | 93.83% | 96.83% | 80.08% | 81.58% |
| 0.5 | 30 | `20260905_110809_foreground_convnextv2_tiny_bbox448_arcface` | 87.83% | 87.69% | 95.92% | 85.75% | 93.33% | 96.67% | 81.33% | 82.81% |
| 0.6 | 30 | `20260905_134056_foreground_convnextv2_tiny_bbox448_arcface_m0_6_s30` | 87.08% | 86.94% | 96.00% | 84.83% | 92.42% | 96.17% | 80.30% | 81.64% |
| 0.5 | 16 | `20260905_141401_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16` | 87.17% | 87.08% | 96.42% | 86.33% | 93.58% | 96.42% | 82.72% | 84.21% |
| 0.5 | 64 | `20260905_150755_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s64` | 87.17% | 86.65% | 95.75% | 86.17% | 93.58% | 97.00% | 81.38% | 82.49% |

## 训练记录

| Margin | Scale | Best Epoch | 实际训练 Epoch | 训练时间 |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 30 | 6 | 11 | 33.27 min |
| 0.4 | 30 | 7 | 12 | 36.02 min |
| 0.5 | 30 | 7 | 12 | 36.23 min |
| 0.6 | 30 | 7 | 11 | 32.87 min |
| 0.5 | 16 | 13 | 18 | 53.65 min |
| 0.5 | 64 | 5 | 10 | 29.86 min |

## 最优后处理设置

| Margin | Scale | 最优 mAP 设置 | Recall@1 | Recall@5 | Recall@10 | mAP | 最优 Recall@1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 30 | QE top3 alpha0.5 | 83.67% | 93.33% | 96.75% | 80.11% | 84.42% |
| 0.4 | 30 | QE top3 alpha0.5 | 84.83% | 93.58% | 96.33% | 81.58% | 85.50% |
| 0.5 | 30 | QE top3 alpha0.5 | 86.25% | 92.42% | 96.33% | 82.81% | 86.67% |
| 0.6 | 30 | QE top3 alpha0.5 | 84.92% | 92.33% | 96.00% | 81.64% | 85.50% |
| 0.5 | 16 | QE top3 alpha0.5 | 87.50% | 93.33% | 96.67% | 84.21% | 87.83% |
| 0.5 | 64 | QE top3 alpha0.5 | 86.67% | 93.08% | 96.83% | 82.49% | 86.75% |

## 结果分析

在固定 `scale=30` 时，margin 呈现清晰的先升后降趋势。`margin=0.3` 的基础 mAP 为 `78.85%`，`margin=0.4` 提升到 `80.08%`，`margin=0.5` 达到 `81.33%`，但 `margin=0.6` 回落到 `80.30%`。这说明 angular margin 存在合适强度：太小不能充分拉开类间角度，太大则可能让训练约束过强，损害类内 ranking structure。

scale ablation 的结论更有价值。`scale=16` 的 Val Acc 为 `87.17%`，略低于 `scale=30` 的 `87.83%`，但基础 retrieval mAP 提升到 `82.72%`，后处理 mAP 提升到 `84.21%`，成为当前全项目最强结果。这说明更低的 scale 可能让 softmax 分布没有过早变得过尖锐，从而保留了更适合 nearest-neighbor ranking 的 embedding geometry。

`scale=64` 的基础 mAP 为 `81.38%`，接近 `scale=30`，但后处理 mAP 为 `82.49%`，低于 `scale=16` 和 `scale=30`。这说明过高 scale 并不会带来更好的 retrieval 排序，可能使训练更偏向分类边界而不是样本间连续结构。

最终报告时不要只看 Val Acc。当前分类最优仍是 `margin=0.5, scale=30`，Val Acc 为 `87.83%`；但 retrieval 最优是 `margin=0.5, scale=16`，基础 mAP 为 `82.72%`，后处理 mAP 为 `84.21%`。这再次证明 classification performance 和 retrieval representation quality 相关但不等价。

## 官方测试集最终结果

模型选择和超参数分析均在 official train 内部 validation split 上完成。最终确定的 `margin=0.5, scale=16` 配置随后只在 CUB official test set 上进行一次最终评估。

测试设置：

- Checkpoint：`20260905_141401_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16`
- Split：`data/processed/splits/cub_test_ids.txt`
- Samples：`5,794`
- Feature：`embedding`，768 维
- Metric：cosine
- Views：`bbox` + `bbox_flip`

| 设置 | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| BBox + Flip TTA | 88.16% | 94.24% | 95.51% | 83.06% |
| BBox + Flip TTA + QE (top-3, alpha=0.5) | 88.30% | 93.63% | 94.99% | 83.59% |

官方测试结果汇总文件：

```text
outputs/embeddings/final_test/20260905_141401_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16/bbox_bbox_flip/embedding/summary.json
```

Query Expansion 使 mAP 从 `83.06%` 提升到 `83.59%`，但 Recall@5 和 Recall@10 略有下降。因此，最终主结果应报告 QE 配置的 mAP，同时保留无 QE 的 TTA 结果作为直接 embedding 检索对照。

## 结论

当前推荐主配置应从：

```text
ArcFace margin=0.5, scale=30
```

更新为：

```text
ArcFace margin=0.5, scale=16
```

在 validation split 上，如果以 mAP 为主指标，`ConvNeXt V2 Tiny BBox448 + ArcFace(m=0.5, s=16) + Flip TTA + QE top3 alpha0.5` 是当前最佳方法，mAP 达到 `84.21%`。在 official test set 上，该配置达到 `83.59%` mAP 和 `88.30%` Recall@1，作为当前项目的最终报告结果。
