# 实验 11.2：ArcFace 后处理与强 Backbone 迁移

## 实验动机

实验 11.1 表明，`Swin-Tiny BBox 224 + ArcFace` 已经达到 `77.30%` mAP，超过此前 `ConvNeXt V2 Tiny BBox 448 + Flip TTA + Query Expansion` 的 `76.82%`。这说明 retrieval 的关键不只是 backbone capacity 和输入分辨率，也包括训练目标是否与 cosine retrieval metric 对齐。

因此下一步需要回答两个问题：

1. ArcFace checkpoint 是否还能通过 TTA / Query Expansion 继续提升？
2. ArcFace 的收益能否迁移到更强的 ConvNeXt V2 Tiny + BBox 448 上？

## Research Question

```text
Angular margin objective、强 backbone、高分辨率输入和 retrieval post-processing 是否能够叠加？
```

## 实验设计

| 组别 | 方法 | 输入 | 训练目标 | 后处理 | 目的 |
| --- | --- | --- | --- | --- | --- |
| A | Swin-Tiny ArcFace | BBox 224 | ArcFace | 无 | 当前 ArcFace baseline |
| B | Swin-Tiny ArcFace + Flip TTA | BBox 224 + flip | ArcFace | TTA | 验证 test-time feature averaging |
| C | Swin-Tiny ArcFace + Flip TTA + QE | BBox 224 + flip | ArcFace | TTA + QE | 验证 query expansion 是否继续提升排序 |
| D | ConvNeXt V2 Tiny ArcFace | BBox 448 | ArcFace | 无 | 验证强 backbone + angular margin 是否叠加 |

## 当前已知 Baseline

| 方法 | Val Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny BBox 224 CE | 80.00% | 68.42% | 86.50% | 93.08% | 55.51% |
| Swin-Tiny BBox 224 ArcFace | 84.92% | 82.33% | 91.58% | 95.17% | 77.30% |
| ConvNeXt V2 Tiny BBox 448 CE | 86.42% | 82.75% | 93.75% | 96.92% | 75.06% |
| ConvNeXt V2 Tiny BBox 448 CE + Flip TTA + QE | 86.42% | 82.83% | 93.83% | 96.58% | 76.82% |

## 运行指令

### 1. Swin-Tiny ArcFace：TTA + Query Expansion

```powershell
cd D:\code\VisionSearch-FG

$runArc = "20260904_105853_foreground_swin_tiny_bbox224_arcface"

.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\foreground_swin_tiny_bbox224_arcface.yaml `
  --checkpoint "outputs\checkpoints\foreground_swin_tiny_bbox224_arcface\$runArc\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views bbox bbox_flip `
  --qe-top-k 0 3 5 10 `
  --qe-alpha 0.0 0.1 0.2 0.5 `
  --device cuda `
  --output-dir outputs\embeddings\phase11_arcface_postprocessing
```

如果 CUDA 显存紧张，把 `--device cuda` 改成 `--device cpu`。

### 2. ConvNeXt V2 Tiny BBox448 + ArcFace：训练

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_margin_classifier.py `
  --config configs\foreground_convnextv2_tiny_bbox448_arcface.yaml `
  --device cuda
```

### 3. ConvNeXt V2 Tiny BBox448 + ArcFace：基础检索评估

```powershell
cd D:\code\VisionSearch-FG

$runConvArc = Get-ChildItem outputs\checkpoints\foreground_convnextv2_tiny_bbox448_arcface |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\foreground_convnextv2_tiny_bbox448_arcface.yaml `
  --checkpoint "outputs\checkpoints\foreground_convnextv2_tiny_bbox448_arcface\$runConvArc\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cuda `
  --output-dir outputs\embeddings\angular_margin
```

### 4. ConvNeXt V2 Tiny BBox448 + ArcFace：TTA + Query Expansion

```powershell
cd D:\code\VisionSearch-FG

$runConvArc = Get-ChildItem outputs\checkpoints\foreground_convnextv2_tiny_bbox448_arcface |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\foreground_convnextv2_tiny_bbox448_arcface.yaml `
  --checkpoint "outputs\checkpoints\foreground_convnextv2_tiny_bbox448_arcface\$runConvArc\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views bbox bbox_flip `
  --qe-top-k 0 3 5 10 `
  --qe-alpha 0.0 0.1 0.2 0.5 `
  --device cuda `
  --output-dir outputs\embeddings\phase11_arcface_postprocessing
```

## 结果记录表

| 方法 | Run ID | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny BBox224 ArcFace | `20260904_105853_foreground_swin_tiny_bbox224_arcface` | 84.92% | 84.50% | 95.17% | 82.33% | 91.58% | 95.17% | 77.30% |
| Swin-Tiny BBox224 ArcFace + Flip TTA | `20260904_105853_foreground_swin_tiny_bbox224_arcface` | 84.92% | 84.50% | 95.17% | 82.67% | 91.33% | 95.50% | 77.71% |
| Swin-Tiny BBox224 ArcFace + Flip TTA + QE | `20260904_105853_foreground_swin_tiny_bbox224_arcface` | 84.92% | 84.50% | 95.17% | 83.00% | 90.33% | 94.25% | 78.62% |
| ConvNeXt V2 Tiny BBox448 ArcFace | `20260905_110809_foreground_convnextv2_tiny_bbox448_arcface` | 87.83% | 87.69% | 95.92% | 85.75% | 93.33% | 96.67% | 81.33% |
| ConvNeXt V2 Tiny BBox448 ArcFace + Flip TTA | `20260905_110809_foreground_convnextv2_tiny_bbox448_arcface` | 87.83% | 87.69% | 95.92% | 86.50% | 93.25% | 96.75% | 82.34% |
| ConvNeXt V2 Tiny BBox448 ArcFace + Flip TTA + QE | `20260905_110809_foreground_convnextv2_tiny_bbox448_arcface` | 87.83% | 87.69% | 95.92% | 86.25% | 92.42% | 96.33% | 82.81% |

## Post-processing Grid

### Swin-Tiny BBox224 ArcFace

| 方法 | QE Top-K | Alpha | Recall@1 | Recall@5 | Recall@10 | mAP | Query Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flip TTA | 0 | 0.0 | 82.67% | 91.33% | 95.50% | 77.71% | 0.0560 ms/query |
| Flip TTA + QE | 3 | 0.1 | 82.67% | 91.17% | 95.42% | 78.06% | 0.0561 ms/query |
| Flip TTA + QE | 3 | 0.2 | 83.17% | 90.58% | 95.25% | 78.38% | 0.0561 ms/query |
| Flip TTA + QE | 3 | 0.5 | 83.00% | 90.33% | 94.25% | 78.62% | 0.0554 ms/query |
| Flip TTA + QE | 5 | 0.1 | 82.67% | 91.17% | 95.58% | 77.96% | 0.0553 ms/query |
| Flip TTA + QE | 5 | 0.2 | 82.50% | 90.83% | 95.42% | 78.11% | 0.0579 ms/query |
| Flip TTA + QE | 5 | 0.5 | 82.92% | 90.67% | 94.75% | 78.32% | 0.0577 ms/query |
| Flip TTA + QE | 10 | 0.1 | 82.83% | 91.25% | 95.50% | 77.77% | 0.0566 ms/query |
| Flip TTA + QE | 10 | 0.2 | 82.75% | 90.83% | 95.42% | 77.77% | 0.0556 ms/query |
| Flip TTA + QE | 10 | 0.5 | 82.50% | 90.42% | 95.33% | 77.79% | 0.0559 ms/query |

### ConvNeXt V2 Tiny BBox448 ArcFace

| 方法 | QE Top-K | Alpha | Recall@1 | Recall@5 | Recall@10 | mAP | Query Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flip TTA | 0 | 0.0 | 86.50% | 93.25% | 96.75% | 82.34% | 0.0561 ms/query |
| Flip TTA + QE | 3 | 0.1 | 86.67% | 92.83% | 96.83% | 82.57% | 0.0562 ms/query |
| Flip TTA + QE | 3 | 0.2 | 86.33% | 92.75% | 96.50% | 82.65% | 0.0566 ms/query |
| Flip TTA + QE | 3 | 0.5 | 86.25% | 92.42% | 96.33% | 82.81% | 0.0567 ms/query |
| Flip TTA + QE | 5 | 0.1 | 86.50% | 93.25% | 96.83% | 82.46% | 0.0564 ms/query |
| Flip TTA + QE | 5 | 0.2 | 86.33% | 93.00% | 96.67% | 82.53% | 0.0576 ms/query |
| Flip TTA + QE | 5 | 0.5 | 85.42% | 92.50% | 96.58% | 82.57% | 0.0595 ms/query |
| Flip TTA + QE | 10 | 0.1 | 86.25% | 93.17% | 96.83% | 82.28% | 0.0578 ms/query |
| Flip TTA + QE | 10 | 0.2 | 85.67% | 92.83% | 96.75% | 82.21% | 0.0558 ms/query |
| Flip TTA + QE | 10 | 0.5 | 84.92% | 92.58% | 96.50% | 82.01% | 0.0556 ms/query |

## 训练记录

| 方法 | Best Epoch | 实际训练 Epoch | 训练时间 | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt V2 Tiny BBox448 ArcFace | 7 | 12 | 36.23 min | 28.02M |

## 已完成结果分析

Swin-Tiny ArcFace 的 Flip TTA 带来小幅但稳定的 mAP 提升：从 `77.30%` 提升到 `77.71%`。这说明 ArcFace embedding 对水平翻转平均仍然受益，但提升幅度不大，表示单视角 embedding 已经比较稳定。

Query Expansion 继续提升完整排序质量。最佳 mAP 来自 `top3 alpha0.5`，达到 `78.62%`，成为当前项目整体最高 mAP。最佳 Recall@1 来自 `top3 alpha0.2`，达到 `83.17%`，与此前 ConvNeXt V2 Tiny BBox448 + TTA + QE 的 Recall@1 持平。

需要注意的是，QE 提升 mAP 的同时会降低 Recall@5 / Recall@10。`top3 alpha0.5` 的 mAP 最高，但 Recall@5 从 `91.33%` 下降到 `90.33%`。这说明 Query Expansion 更擅长改善相关样本的整体排序位置，但过强的 expansion 也会带来一定 query drift。

ConvNeXt V2 Tiny BBox448 + ArcFace 进一步刷新主结果。基础检索 mAP 为 `81.33%`，已经明显超过 Swin-Tiny BBox224 ArcFace + TTA + QE 的 `78.62%`。这说明 strong backbone、foreground-aware high-resolution input 和 angular margin objective 可以叠加。

Flip TTA 将 ConvNeXt ArcFace 的 mAP 从 `81.33%` 提升到 `82.34%`，同时 Recall@1 从 `85.75%` 提升到 `86.50%`。继续加入 Query Expansion 后，最佳 mAP 来自 `top3 alpha0.5`，达到 `82.81%`；最佳 Recall@1 来自 `top3 alpha0.1`，达到 `86.67%`。与 Swin ArcFace 类似，QE 主要提升 mAP，但会牺牲部分 Recall@5，因此正式报告时应同时列出 TTA-only 和 TTA+QE 两种设置。

## 预期解释

Swin ArcFace + TTA/QE 已经继续提升，说明 Angular Margin 学到的 embedding 仍可被后处理进一步优化。

ConvNeXt ArcFace 已经超过 Swin ArcFace，说明强 backbone、高分辨率输入和 angular margin 可以叠加。这是当前最值得写入主线论文故事的方向。

后续如果继续优化，应优先围绕 ConvNeXt V2 Tiny BBox448 ArcFace 做更细的 ArcFace margin / scale ablation、TTA view ablation，以及 hard negative 或 local part-aware re-ranking。
