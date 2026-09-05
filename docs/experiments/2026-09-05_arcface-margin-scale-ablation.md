# 实验 11.3：ArcFace Margin / Scale Ablation

## 实验动机

当前最强结果来自：

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
| 0.3 | 30 | 待训练 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| 0.4 | 30 | 待训练 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| 0.5 | 30 | `20260905_110809_foreground_convnextv2_tiny_bbox448_arcface` | 87.83% | 87.69% | 95.92% | 85.75% | 93.33% | 96.67% | 81.33% | 82.81% |
| 0.6 | 30 | 待训练 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| 0.5 | 16 | 待训练 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |
| 0.5 | 64 | 待训练 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 | 待记录 |

## 分析重点

如果 `margin=0.4/0.5/0.6` 呈现先升后降，说明 angular margin 存在最优强度：太小不能充分拉开类间角度，太大则训练过难或损害类内结构。

如果 `scale=16` 明显下降，说明 logits 过软，分类边界约束不足。

如果 `scale=64` 明显下降，说明 logits 过尖锐，训练可能过度关注 hard boundary，损害 retrieval 排序。

最终报告时不要只看 Val Acc。ArcFace 的研究重点是 retrieval geometry，因此主结论应优先依据 `Recall@1 / Recall@5 / Recall@10 / mAP`，分类指标作为辅助证据。
