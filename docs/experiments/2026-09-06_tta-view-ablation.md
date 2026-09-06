# 实验 11.4：TTA View Ablation

## 实验动机

当前最强结果来自：

```text
ConvNeXt V2 Tiny BBox448
+ ArcFace margin=0.5, scale=16
+ Flip TTA
+ Query Expansion top3 alpha0.5
```

该设置达到：

```text
mAP = 84.21%
Recall@1 = 87.50%
```

但是这里仍有一个需要单独验证的问题：当前使用的是 `BBox + BBox Flip` 两个视角平均。对于 CUB 细粒度检索，原图上下文是否还包含有用信息？还是只会引入背景噪声？

## Research Question

```text
在最强 ArcFace embedding 上，不同 TTA view 组合如何影响 retrieval ranking？
```

## 实验设计

固定变量：

- Dataset：CUB-200-2011
- Split：`cub_val_ids_seed42.txt`
- Backbone：ConvNeXt V2 Tiny
- Input size：448
- Training objective：ArcFace, margin = 0.5, scale = 16
- Retrieval metric：cosine
- Post-processing grid：QE top-k = 0 / 3 / 5 / 10，alpha = 0.0 / 0.1 / 0.2 / 0.5

变量：

| 组别 | Views | 目的 |
| --- | --- | --- |
| A | `bbox` | 单视角 baseline |
| B | `bbox bbox_flip` | 当前默认 Flip TTA |
| C | `original` | 原图上下文单视角 |
| D | `original original_flip` | 原图上下文 Flip TTA |
| E | `original bbox` | 原图 + 前景平均 |
| F | `original original_flip bbox bbox_flip` | 全视角平均 |

## 运行指令

本实验不需要重新训练，只评估当前最强 checkpoint。

### 1. 一次性评估所有 view 组合

```powershell
cd D:\code\VisionSearch-FG

$expName = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16"
$config = "configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml"
$run = Get-ChildItem "outputs\checkpoints\$expName" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

$viewSets = @(
  @{ Name = "bbox"; Views = @("bbox") },
  @{ Name = "bbox_bbox_flip"; Views = @("bbox", "bbox_flip") },
  @{ Name = "original"; Views = @("original") },
  @{ Name = "original_original_flip"; Views = @("original", "original_flip") },
  @{ Name = "original_bbox"; Views = @("original", "bbox") },
  @{ Name = "all_views"; Views = @("original", "original_flip", "bbox", "bbox_flip") }
)

foreach ($viewSet in $viewSets) {
  $views = $viewSet.Views
  Write-Host "Evaluating views: $($viewSet.Name)" -ForegroundColor Cyan
  .\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
    --config $config `
    --checkpoint "outputs\checkpoints\$expName\$run\best.pt" `
    --split train `
    --ids-path data\processed\splits\cub_val_ids_seed42.txt `
    --feature embedding `
    --views $views `
    --qe-top-k 0 3 5 10 `
    --qe-alpha 0.0 0.1 0.2 0.5 `
    --device cuda `
    --output-dir outputs\embeddings\arcface_tta_view_ablation
}
```

如果 CUDA 评估不稳定，把 `--device cuda` 改成 `--device cpu`。

### 2. 结果输出位置

```text
outputs\embeddings\arcface_tta_view_ablation\<run_id>\<views>\embedding\summary.json
```

其中 `<run_id>` 应为：

```text
20260905_141401_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16
```

## 结果记录表

| Views | Recall@1 | Recall@5 | Recall@10 | mAP | 最优 QE 设置 | 最优 QE Recall@1 | 最优 QE mAP |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `bbox` | 86.33% | 93.58% | 96.42% | 82.72% | top3 alpha0.5 | 86.58% | 83.20% |
| `bbox bbox_flip` | 87.33% | 94.25% | 96.92% | 83.79% | top3 alpha0.5 | 87.50% | 84.21% |
| `original` | 83.50% | 93.00% | 96.17% | 79.53% | top3 alpha0.5 | 83.08% | 80.04% |
| `original original_flip` | 84.75% | 93.33% | 96.50% | 80.79% | top3 alpha0.5 | 85.00% | 81.44% |
| `original bbox` | 86.00% | 93.50% | 96.67% | 82.77% | top3 alpha0.5 | 86.08% | 83.26% |
| `original original_flip bbox bbox_flip` | 87.75% | 94.25% | 96.83% | 83.64% | top3 alpha0.5 | 87.50% | 84.14% |

## 最优 Recall@1 设置

| Views | 最优 Recall@1 设置 | Recall@1 | mAP |
| --- | --- | ---: | ---: |
| `bbox` | QE top10 alpha0.1 | 86.67% | 82.75% |
| `bbox bbox_flip` | QE top3 alpha0.2 | 87.83% | 84.15% |
| `original` | QE top5 alpha0.2 | 83.58% | 79.82% |
| `original original_flip` | QE top3 alpha0.2 | 85.17% | 81.22% |
| `original bbox` | QE top5 alpha0.5 | 86.33% | 83.17% |
| `original original_flip bbox bbox_flip` | QE top5 alpha0.1 | 87.83% | 83.78% |

## 结果分析

`bbox bbox_flip` 是当前最稳的默认视角组合。它的 TTA-only mAP 为 `83.79%`，加入 QE top3 alpha0.5 后达到 `84.21%`，是本实验最高 mAP。相比单独 `bbox`，Flip TTA 将 mAP 从 `82.72%` 提升到 `83.79%`，说明水平翻转平均确实能降低单视角姿态扰动。

`original` 明显弱于 `bbox`。原图单视角 mAP 只有 `79.53%`，低于 BBox 单视角的 `82.72%`；原图 Flip TTA 后 mAP 提升到 `80.79%`，但仍然明显低于前景裁剪。这支持前面 foreground-aware 实验和 explainability 分析中的结论：背景上下文虽然有时有帮助，但对细粒度 retrieval 并不稳定。

`original bbox` 没有超过 `bbox bbox_flip`。原图 + 前景平均的最优 mAP 为 `83.26%`，低于 `bbox bbox_flip` 的 `84.21%`。这说明简单加入原图上下文会引入一定背景噪声，不能替代前景视角的稳定增强。

全视角平均有一个细节值得保留：TTA-only Recall@1 达到 `87.75%`，是所有 TTA-only 设置中最高；但最优 QE mAP 为 `84.14%`，仍略低于 `bbox bbox_flip` 的 `84.21%`。因此如果主指标是 mAP，应选择 `bbox bbox_flip + QE top3 alpha0.5`；如果单独强调 Recall@1，可以把全视角平均作为补充结果报告。

## 分析重点

实验结果显示，`bbox bbox_flip` 仍然是 mAP 最强的视角组合，说明当前模型的判别信息主要来自前景鸟体，原图背景更多是噪声。

全视角平均在 TTA-only Recall@1 上有轻微优势，说明背景或上下文在部分类别上仍有辅助作用，但这种帮助没有稳定转化为更好的整体排序 mAP。

`original` 明显弱于 `bbox`，进一步支持前面 Grad-CAM / Attention 分析中的结论：背景 shortcut 对 fine-grained retrieval 不稳定，foreground-aware input 是必要组件。

正式报告时应同时保留 TTA-only 与 TTA+QE 结果，因为 QE 提升 mAP 的同时可能降低 Recall@5 / Recall@10。

## 结论

当前推荐默认 retrieval pipeline 保持为：

```text
ConvNeXt V2 Tiny BBox448
+ ArcFace margin=0.5, scale=16
+ BBox + BBox Flip feature averaging
+ Query Expansion top3 alpha0.5
```

该设置 mAP 为 `84.21%`。如果需要报告 Recall@1 最强设置，可以补充：

```text
BBox + BBox Flip + QE top3 alpha0.2
```

或：

```text
Original + Original Flip + BBox + BBox Flip + QE top5 alpha0.1
```

二者 Recall@1 均为 `87.83%`。
