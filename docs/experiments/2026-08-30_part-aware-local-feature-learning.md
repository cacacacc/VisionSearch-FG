# 方向 8：Part-aware / Local Feature Learning

## 方向动机

Phase 6 的 Grad-CAM、Swin attention-style visualization 和 retrieval qualitative analysis 已经给出一致证据：模型通常能看到鸟主体，但对喙、眼睛、头部、翼部纹理等细粒度局部区域并不稳定。错误 top-k 经常由整体颜色、背景、姿态或局部部位混淆造成。

因此，下一步不应只依赖更大的 backbone，而应显式研究：

```text
模型是否能稳定定位并利用 discriminative local regions？
```

这与近年 fine-grained recognition / fine-grained retrieval 的思路一致。Hierarchical Attention Vision Transformer 使用 attention cropping、attention dropping、hierarchical attention selection、token filtering 和 fusion 做细粒度分类；Fine-Grained Image Retrieval 的相关工作也强调，仅靠 image-level label 学到的 CNN 特征容易关注最强判别 patch，但可能忽略局部 patch 之间的关系，需要 relation extraction 和 descriptor aggregation。

CUB-200-2011 还提供 15 个 part annotations，包括 beak、eye、wing、tail、belly、breast 等。因此本项目可以先做一个很清楚的量化问题：

```text
模型热区是否靠近人工标注的关键 bird parts？
```

注意：part annotations 优先用于 evaluation / supervision study。不要让最终 test pipeline 永远依赖人工 part coordinates，否则会变成 oracle setting，不能和普通输入 baseline 直接比较。

## 实验 8.1：Part Annotation Alignment Evaluation

### 研究问题

```text
Grad-CAM / Swin attention 的最高响应区域，是否靠近 beak、eye、wing 等细粒度关键部位？
```

### 方法

本实验不训练模型，只重新生成带 `.npy` 原始 heatmap 的 explainability report，然后把 heatmap peak 与 CUB `parts/part_locs.txt` 对齐。

评估对象：

| Group | Parts |
| --- | --- |
| beak | beak |
| eye | left eye, right eye |
| head | beak, crown, forehead, left eye, right eye, nape, throat |
| wing | left wing, right wing |
| body | back, belly, breast, left wing, right wing, tail |

核心指标：

| 指标 | 含义 |
| --- | --- |
| `peak_hit_rate` | heatmap peak 是否落在某组 part 的半径范围内 |
| `mean_peak_distance_px` | heatmap peak 到最近 part 的平均像素距离 |
| `mean_heat_mass` | part 附近区域占总 heatmap 响应的比例 |

默认半径：

```text
radius = 24 pixels
image_size = 224
```

### 运行指令：重新生成 Swin heatmap

不要覆盖旧的人工标注目录，输出到新目录：

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\generate_swin_attention_visualization.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --checkpoint outputs\checkpoints\baseline_swin_tiny_protocol\20260823_012929_baseline_swin_tiny_protocol\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu `
  --num-correct 12 `
  --num-wrong 12 `
  --output-dir outputs\explainability\swin_attention_part_alignment_source
```

### 运行指令：计算 Swin part alignment

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\evaluate_part_heatmap_alignment.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --records outputs\explainability\swin_attention_part_alignment_source\20260823_012929_baseline_swin_tiny_protocol\attention_records.csv `
  --radius-pixels 24 `
  --output-dir outputs\explainability\part_alignment
```

### 可选：BBox crop 模型的 part alignment

BBox crop 模型应更少关注背景，但不一定更稳定覆盖喙、眼睛、翼部等局部 parts。该对照可以验证 foreground crop 是否真的改善局部证据。

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\generate_swin_attention_visualization.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --checkpoint outputs\checkpoints\foreground_swin_tiny_bbox224\20260830_141148_foreground_swin_tiny_bbox224\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu `
  --num-correct 12 `
  --num-wrong 12 `
  --output-dir outputs\explainability\swin_attention_bbox_part_alignment_source

.\.venv\Scripts\python.exe scripts\evaluate_part_heatmap_alignment.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --records outputs\explainability\swin_attention_bbox_part_alignment_source\20260830_141148_foreground_swin_tiny_bbox224\attention_records.csv `
  --radius-pixels 24 `
  --output-dir outputs\explainability\part_alignment
```

### 当前结果

本轮已经完成原图 Swin 和 BBox crop Swin 的 part alignment 初步评估。注意：两组 explainability report 都是各自重新选择 `12 correct + 12 wrong` 样本，因此当前结果不是严格同一批 image_id 的 paired comparison，只能作为方向性证据。

| Method | Crop Mode | Samples | Beak Hit | Eye Hit | Head Hit | Wing Hit | Body Hit | Target Hit | Target Distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny Original | none | 24 | 12.50% | 4.17% | 25.00% | 17.39% | 45.83% | 29.17% | 46.05 px |
| Swin-Tiny BBox Crop | bbox, margin=0.15 | 24 | 0.00% | 4.35% | 8.33% | 12.50% | 29.17% | 16.67% | 54.30 px |

按预测正确与错误拆分后：

| Method | Group | Beak Hit | Eye Hit | Head Hit | Wing Hit | Body Hit | Target Hit | Target Distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny Original | Correct | 16.67% | 0.00% | 33.33% | 25.00% | 58.33% | 41.67% | 39.94 px |
| Swin-Tiny Original | Wrong | 8.33% | 8.33% | 16.67% | 8.33% | 33.33% | 16.67% | 52.16 px |
| Swin-Tiny BBox Crop | Correct | 0.00% | 8.33% | 16.67% | 8.33% | 16.67% | 16.67% | 54.55 px |
| Swin-Tiny BBox Crop | Wrong | 0.00% | 0.00% | 0.00% | 16.67% | 41.67% | 16.67% | 54.04 px |

初步结论：

```text
1. 原图 Swin 的正确样本比错误样本更接近关键 parts：
   Target Hit 从 16.67% 提升到 41.67%，Target Distance 从 52.16 px 降到 39.94 px。

2. BBox crop 虽然显著提升分类和 retrieval 数值，但当前 heatmap peak 并没有更稳定落在 beak / eye / wing 附近。

3. 这说明 BBox crop 的收益更可能来自“减少背景干扰、放大鸟主体”，而不一定来自模型已经学会稳定关注喙、眼睛、翼部等细粒度 part。

4. 因此 Part-aware / Local Feature Learning 仍然有必要：BBox crop 提升了输入质量，但没有解决 discriminative local region selection 的问题。
```

当前输出文件：

```text
outputs/explainability/part_alignment/20260823_012929_baseline_swin_tiny_protocol/summary.json
outputs/explainability/part_alignment/20260830_141148_foreground_swin_tiny_bbox224/summary.json
```

后续如果要做更严格分析，应固定同一批 image_id，例如从 validation split 中指定同一组正确/错误样本，再分别生成原图和 bbox crop heatmap。这样才能回答 BBox crop 是否真的改善 part alignment。

### Paired comparison 结果

为避免不同样本选择造成偏差，后续补充了严格的 paired comparison：固定同一批 24 个 validation image_id，同时对原图 Swin 和 BBox crop Swin 生成 heatmap 并计算 part alignment。

```text
same_set = True
num_images = 24
original correct / wrong = 12 / 12
bbox correct / wrong = 15 / 9
```

paired 输出文件：

```text
outputs/explainability/part_alignment_paired/20260823_012929_baseline_swin_tiny_protocol/summary.json
outputs/explainability/part_alignment_paired/20260830_141148_foreground_swin_tiny_bbox224/summary.json
```

整体结果：

| Method | Samples | Beak Hit | Eye Hit | Head Hit | Wing Hit | Body Hit | Target Hit | Target Distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny Original | 24 | 12.50% | 4.17% | 25.00% | 17.39% | 45.83% | 29.17% | 46.05 px |
| Swin-Tiny BBox Crop | 24 | 12.50% | 0.00% | 16.67% | 17.39% | 33.33% | 29.17% | 44.82 px |

逐图距离差异统计，`mean_delta = BBox distance - Original distance`：

| Part Group | Eligible Images | BBox 更近的图片数 | Mean Delta |
| --- | ---: | ---: | ---: |
| Beak | 24 | 12 | +8.90 px |
| Eye | 24 | 11 | +10.50 px |
| Head | 24 | 11 | +4.39 px |
| Wing | 23 | 9 | +7.55 px |
| Body | 24 | 10 | +3.70 px |
| Target = beak/eye/wing | 24 | 10 | -1.23 px |

paired 结论：

```text
1. BBox crop 让同一批图的预测正确数从 12/24 提升到 15/24，说明 foreground crop 确实改善分类判断。

2. BBox crop 没有显著提高 beak / eye / head / wing / body 的 peak hit rate。

3. Target = beak/eye/wing 的 hit rate 两者相同，都是 29.17%；BBox 的平均 target distance 略低 1.23 px，但幅度很小。

4. 因此，BBox crop 的主要收益仍应解释为减少背景干扰和放大鸟主体，而不是已经解决了稳定 part localization。

5. 这进一步支持后续做 Local Token Pooling 或 Part-aware Training：现有强结果仍没有稳定使用喙、眼睛、翼部等局部判别区域。
```

## 实验 8.2：Local Token Pooling Retrieval

### 研究问题

```text
在不重新训练的情况下，Swin 的 top-response local tokens 是否能补充 global embedding？
```

### 方法

使用同一个 Swin checkpoint，提取最后 stage 的 token / feature map：

```text
Global feature = 原始 backbone pooling
Local feature  = Top-K 高响应 token pooling
Final feature  = L2Normalize([global; local])
```

当前实现使用 `scripts/evaluate_swin_local_token_retrieval.py`，不训练模型，只做 feature extraction 和 retrieval evaluation。Top-K token 的选择规则是：

```text
1. 取 Swin final stage spatial tokens。
2. 对每个 token 计算 L2 response。
3. 选择 response 最大的 Top-K tokens。
4. 对 Top-K tokens 做 mean pooling，得到 local feature。
5. 比较 global / local_topK / global+local_topK。
```

对照组：

| Method | 是否训练 | Feature |
| --- | --- | --- |
| Swin global | 0 epoch | global backbone h |
| Swin local top1 / top3 / top5 | 0 epoch | local pooled tokens |
| Swin global + local top1 / top3 / top5 | 0 epoch | concat(global, local) |
| Swin bbox global + local top1 / top3 / top5 | 0 epoch | bbox crop 上的 concat(global, local) |

该实验可以先作为 feature extraction / retrieval evaluation，不需要训练。若 `global + local` 明显优于 `global`，说明局部 token 中确实存在可利用的细粒度信息，后续再进入真正的 part-aware training。

### 运行指令：原图 Swin

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\evaluate_swin_local_token_retrieval.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --checkpoint outputs\checkpoints\baseline_swin_tiny_protocol\20260823_012929_baseline_swin_tiny_protocol\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu `
  --metric cosine `
  --top-k-tokens 1 3 5 `
  --output-dir outputs\embeddings\swin_local_token_retrieval
```

### 运行指令：BBox crop Swin

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\evaluate_swin_local_token_retrieval.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --checkpoint outputs\checkpoints\foreground_swin_tiny_bbox224\20260830_141148_foreground_swin_tiny_bbox224\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu `
  --metric cosine `
  --top-k-tokens 1 3 5 `
  --output-dir outputs\embeddings\swin_local_token_retrieval
```

输出结构：

```text
outputs/embeddings/swin_local_token_retrieval/<run_id>/summary.csv
outputs/embeddings/swin_local_token_retrieval/<run_id>/<variant>/cosine/summary.json
outputs/embeddings/swin_local_token_retrieval/<run_id>/<variant>/cosine/embeddings.npy
outputs/embeddings/swin_local_token_retrieval/<run_id>/<variant>/cosine/top_results.json
```

结果表暂待运行后填写：

| Method | Variant | Dim | Recall@1 | Recall@5 | Recall@10 | mAP | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Swin Original | global | 768 | 61.25% | 84.33% | 90.67% | 49.04% | 原图全局 baseline |
| Swin Original | local_top1 | 768 | 39.42% | 68.92% | 79.42% | 29.80% | 单个最高响应 token 明显不足 |
| Swin Original | local_top3 | 768 | 52.08% | 79.42% | 88.75% | 41.19% | 局部 token pooling 有语义，但弱于 global |
| Swin Original | local_top5 | 768 | 56.75% | 81.00% | 88.75% | 44.18% | Top-5 local 最好，但仍低于 global |
| Swin Original | global_local_top1_concat_l2 | 1536 | 55.50% | 80.92% | 88.75% | 43.76% | 加入 top1 local 后退化 |
| Swin Original | global_local_top3_concat_l2 | 1536 | 59.58% | 82.92% | 90.50% | 46.84% | 接近 global，但没有超过 |
| Swin Original | global_local_top5_concat_l2 | 1536 | 60.75% | 83.33% | 90.58% | 47.77% | 最好 concat，但仍低于 global |
| Swin BBox Crop | global | 768 | 68.42% | 86.50% | 93.08% | 55.51% | BBox 全局 baseline |
| Swin BBox Crop | local_top1 | 768 | 47.58% | 72.92% | 83.67% | 33.94% | 单个局部 token 不稳定 |
| Swin BBox Crop | local_top3 | 768 | 61.33% | 82.92% | 89.92% | 46.53% | 前景内 local 有效但弱于 global |
| Swin BBox Crop | local_top5 | 768 | 64.67% | 84.58% | 90.42% | 50.14% | BBox local top5 最好，但仍低于 global |
| Swin BBox Crop | global_local_top1_concat_l2 | 1536 | 63.75% | 85.42% | 91.83% | 50.36% | 加入 top1 local 后退化 |
| Swin BBox Crop | global_local_top3_concat_l2 | 1536 | 66.17% | 86.42% | 92.58% | 53.28% | 接近 global，但没有超过 |
| Swin BBox Crop | global_local_top5_concat_l2 | 1536 | 67.42% | 86.33% | 92.42% | 54.20% | 最好 concat，但仍低于 BBox global |

### 8.2 结果分析

当前实验没有支持“直接选择 L2 response 最大的 Top-K tokens 能提升 retrieval”。两组输入下结论一致：

```text
Swin Original:
global mAP = 49.04%
best local = local_top5, mAP = 44.18%
best global+local = top5 concat, mAP = 47.77%

Swin BBox Crop:
global mAP = 55.51%
best local = local_top5, mAP = 50.14%
best global+local = top5 concat, mAP = 54.20%
```

这说明 Swin final-stage token 中确实包含可检索的局部语义，因为 `local_top5` 已经能达到不错的 Recall@K；但简单的 token L2 response 不是足够可靠的 discriminative part selector。它更可能选到高激活区域，而不是稳定选到喙、眼睛、翼部纹理等真正区分类别的 part。

因此 8.2 的结论不是“局部特征无效”，而是：

```text
1. naive local token pooling 不能直接替代 global feature。
2. naive global+local concat 也没有带来互补增益。
3. 后续 part-aware 方法需要更强的局部区域选择机制，而不是只按 token norm 选 Top-K。
```

下一步更合理的方向是：

```text
1. Attention / Grad-CAM weighted pooling：
   用 class-relevant heatmap 权重做 local pooling，而不是用 token L2 norm。

2. Part-guided crop evaluation：
   利用 CUB part annotations 生成 head / beak / wing 局部 crop，只作为 analysis 或 annotation-assisted upper bound。

3. Local-global training：
   显式训练 global branch + local branch，而不是只在训练后拼接 local token。
```

## 实验 8.3：Class-evidence Weighted Local Pooling

### 研究问题

```text
如果不用 token norm，而是用分类 head 对每个 token 的类别证据打分，local pooling 是否能更接近真正的 discriminative regions？
```

### 方法

本实验仍然不训练模型。对于 Swin final-stage tokens：

```text
tokens = Swin final-stage spatial tokens
target_class = predicted class
score(token_i) = token_i dot classifier_weight[target_class]
weights = softmax(score / tau)
h_local = sum_i weights_i * token_i
h_final = L2Normalize([h_global; h_local])
```

默认使用 `target_class=predicted`，因为这是 inference 时可获得的信息。`target_class=true` 只能作为 oracle analysis，不能作为正式检索 pipeline。

温度 `tau` 控制局部选择强度：

| tau | 含义 |
| ---: | --- |
| 0.5 | 更尖锐，更接近选择少数高证据 token |
| 1.0 | 中等加权 |
| 2.0 | 更平滑，覆盖更多 token |

### 运行指令：原图 Swin

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\evaluate_swin_weighted_token_retrieval.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --checkpoint outputs\checkpoints\baseline_swin_tiny_protocol\20260823_012929_baseline_swin_tiny_protocol\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu `
  --metric cosine `
  --target-class predicted `
  --temperatures 0.5 1.0 2.0 `
  --output-dir outputs\embeddings\swin_weighted_token_retrieval
```

### 运行指令：BBox crop Swin

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\evaluate_swin_weighted_token_retrieval.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --checkpoint outputs\checkpoints\foreground_swin_tiny_bbox224\20260830_141148_foreground_swin_tiny_bbox224\best.pt `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --device cpu `
  --metric cosine `
  --target-class predicted `
  --temperatures 0.5 1.0 2.0 `
  --output-dir outputs\embeddings\swin_weighted_token_retrieval
```

输出结构：

```text
outputs/embeddings/swin_weighted_token_retrieval/<run_id>/predicted/summary.csv
outputs/embeddings/swin_weighted_token_retrieval/<run_id>/predicted/<variant>/cosine/summary.json
outputs/embeddings/swin_weighted_token_retrieval/<run_id>/predicted/<variant>/cosine/embeddings.npy
```

| Method | Variant | Dim | Tau | Recall@1 | Recall@5 | Recall@10 | mAP | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Swin Original | global | 768 | - | 61.25% | 84.33% | 90.67% | 49.04% | 全局 baseline |
| Swin Original | evidence_weighted_tau0.5 | 768 | 0.5 | 63.00% | 84.75% | 91.33% | 51.75% | 明显优于 global |
| Swin Original | evidence_weighted_tau1.0 | 768 | 1.0 | 63.25% | 85.08% | 91.42% | 51.80% | 原图最佳 weighted local |
| Swin Original | evidence_weighted_tau2.0 | 768 | 2.0 | 62.67% | 85.42% | 91.08% | 51.01% | 较平滑，mAP 略低 |
| Swin Original | global+weighted_tau0.5 | 1536 | 0.5 | 63.42% | 85.50% | 91.58% | 51.40% | 融合后 Recall 略高，但 mAP 不如 local |
| Swin Original | global+weighted_tau1.0 | 1536 | 1.0 | 62.67% | 85.42% | 91.25% | 50.91% | 融合收益有限 |
| Swin Original | global+weighted_tau2.0 | 1536 | 2.0 | 61.50% | 85.50% | 91.08% | 50.21% | 趋近 global，收益下降 |
| Swin BBox Crop | global | 768 | - | 68.42% | 86.50% | 93.08% | 55.51% | BBox 全局 baseline |
| Swin BBox Crop | evidence_weighted_tau0.5 | 768 | 0.5 | 70.50% | 87.67% | 93.33% | 59.41% | 明显优于 BBox global |
| Swin BBox Crop | evidence_weighted_tau1.0 | 768 | 1.0 | 71.25% | 87.83% | 93.58% | 59.62% | 当前最佳 mAP / Recall@1 |
| Swin BBox Crop | evidence_weighted_tau2.0 | 768 | 2.0 | 70.83% | 87.58% | 93.75% | 58.58% | 更平滑，Recall@10 较高 |
| Swin BBox Crop | global+weighted_tau0.5 | 1536 | 0.5 | 70.25% | 88.00% | 93.58% | 58.74% | Recall@5 高，但 mAP 不如 local |
| Swin BBox Crop | global+weighted_tau1.0 | 1536 | 1.0 | 70.67% | 87.92% | 94.00% | 58.31% | Recall@10 达到 94.00% |
| Swin BBox Crop | global+weighted_tau2.0 | 1536 | 2.0 | 69.92% | 87.42% | 93.75% | 57.33% | 融合后仍不如 local |

### 8.3 结果分析

Class-evidence weighted local pooling 与 8.2 的 naive token norm Top-K 结果完全不同：它在原图和 BBox crop 两个设置下都超过了 global baseline。

```text
Swin Original:
global mAP = 49.04%
best evidence-weighted local = tau1.0, mAP = 51.80%
gain = +2.76 percentage points

Swin BBox Crop:
global mAP = 55.51%
best evidence-weighted local = tau1.0, mAP = 59.62%
gain = +4.11 percentage points
```

这说明问题不在于 Swin token 没有局部信息，而在于局部 token 的选择标准。简单按 token norm 选 Top-K 会选到“激活强”的区域，但不一定是“对当前类别有判别证据”的区域；使用 classifier weight 计算 token-class evidence 后，local feature 的 retrieval geometry 明显改善。

当前最强 retrieval 设置变为：

```text
Swin BBox Crop + class-evidence weighted local pooling
tau = 1.0
Recall@1 = 71.25%
Recall@5 = 87.83%
Recall@10 = 93.58%
mAP = 59.62%
```

它的 mAP 和 Recall@1 高于 `Original + BBox Fusion PCA 512-D`，但 Recall@5 / Recall@10 略低。因此后续主表应区分：

```text
Best mAP / Recall@1: BBox evidence-weighted local tau1.0
Best Recall@5 / Recall@10: Original + BBox Fusion PCA / concat
```

下一步应优先做两个验证：

```text
1. 对 evidence-weighted local feature 做 qualitative retrieval report，检查它是否减少喙、眼睛、翼部纹理混淆。
2. 对 evidence-weighted token heatmap 做 part alignment，验证它是否比普通 Swin heatmap 更接近 beak / eye / wing。
```

补充：evidence-weighted local feature 的 qualitative retrieval 人工标注已完成，记录见：

```text
docs/experiments/2026-08-31_weighted-local-retrieval-qualitative-human-notes.md
outputs/experiments/retrieval_qualitative_weighted_local/swin_bbox_evidence_weighted_tau1/retrieval_qualitative_human_notes.csv
```

## 实验 8.3b：Evidence Token Heatmap Part Alignment

### 研究问题

```text
8.3 中提升 retrieval 的 class-evidence weighted token，是否真的比普通 Swin heatmap 更接近 beak / eye / wing？
```

### 方法

本实验不训练模型。固定 8.1 paired comparison 使用的同一批 24 张图片：

```text
outputs/explainability/part_alignment/paired_image_ids_24.txt
```

对 BBox crop Swin 生成 class-evidence token heatmap：

```text
score(token_i) = token_i dot classifier_weight[predicted_class]
weights = softmax(score / tau)
tau = 1.0
```

再使用 `evaluate_part_heatmap_alignment.py` 计算 beak / eye / head / wing / body / target 的对齐指标。

### 运行指令

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\generate_swin_evidence_token_visualization.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --checkpoint outputs\checkpoints\foreground_swin_tiny_bbox224\20260830_141148_foreground_swin_tiny_bbox224\best.pt `
  --split train `
  --ids-path outputs\explainability\part_alignment\paired_image_ids_24.txt `
  --device cpu `
  --target-class predicted `
  --temperature 1.0 `
  --num-correct 24 `
  --num-wrong 24 `
  --output-dir outputs\explainability\swin_evidence_token_part_alignment_source

.\.venv\Scripts\python.exe scripts\evaluate_part_heatmap_alignment.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --records outputs\explainability\swin_evidence_token_part_alignment_source\20260830_141148_foreground_swin_tiny_bbox224\predicted_tau1\evidence_records.csv `
  --radius-pixels 24 `
  --output-dir outputs\explainability\part_alignment_evidence_token
```

### 当前结果

本实验已经完成。实际输出目录为：

```text
普通 BBox Swin heatmap:
outputs/explainability/part_alignment_paired/20260830_141148_foreground_swin_tiny_bbox224/summary.json

Evidence token heatmap:
outputs/explainability/part_alignment_evidence_token/predicted_tau1/summary.json
outputs/explainability/part_alignment_evidence_token/predicted_tau1/part_alignment_records.csv
```

与普通 BBox Swin heatmap 的 paired comparison 如下：

| Method | Samples | Beak Hit | Eye Hit | Head Hit | Wing Hit | Body Hit | Target Hit | Target Distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BBox Swin heatmap | 24 | 12.50% | 0.00% | 16.67% | 17.39% | 33.33% | 29.17% | 44.82 px |
| BBox evidence-token heatmap | 24 | 12.50% | 4.17% | 12.50% | 17.39% | 29.17% | 29.17% | 47.25 px |

按预测正确与错误样本拆分 evidence-token heatmap：

| Group | Samples | Beak Hit | Eye Hit | Head Hit | Wing Hit | Body Hit | Target Hit | Target Distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct | 15 | 6.67% | 0.00% | 6.67% | 13.33% | 26.67% | 20.00% | 54.62 px |
| Wrong | 9 | 22.22% | 11.11% | 22.22% | 25.00% | 33.33% | 44.44% | 34.97 px |

heat mass 分布显示，evidence-token heatmap 的响应仍主要集中在较大的身体区域，而不是稳定集中到细粒度部位：

| Part Group | Mean Heat Mass |
| --- | ---: |
| Beak | 5.59% |
| Eye | 2.38% |
| Head | 10.83% |
| Wing | 13.47% |
| Body | 24.14% |

结论：

```text
1. Evidence-token local feature 明显提升 retrieval：
   BBox global mAP = 55.51%，evidence-token tau1.0 mAP = 59.62%。
2. 但 evidence-token heatmap 没有同步提升 part alignment：
   Target Hit 与普通 BBox heatmap 相同，均为 29.17%；Target Distance 反而从 44.82 px 增加到 47.25 px。
3. 这说明 retrieval 提升不等价于 heatmap peak 更靠近 beak / eye / wing。
4. 当前 weighted local 更可能学到的是“类别相关的局部纹理/颜色证据”，而不是严格的几何 part localization。
5. 因此后续如果要进一步提高细粒度检索，不能只做 post-hoc token weighting，应该进入真正的 part-aware training 或 part-guided auxiliary evaluation。
```

## 实验 8.4：Part-aware Training

### 研究问题

```text
显式引导模型关注局部 discriminative regions，是否能进一步提升 Accuracy 和 retrieval mAP？
```

可选方法：

| 方法 | 是否使用 part annotation | 说明 |
| --- | --- | --- |
| Attention crop / drop | 否 | 根据模型响应自动生成局部视图 |
| Top-K token selection | 否 | 从 Swin tokens 中选择高响应局部 token |
| Local-global fusion | 否 | 融合 global feature 和 local feature |
| Part-supervised auxiliary loss | 是 | 使用 CUB part locations 做弱监督或分析性监督 |

优先级建议：

```text
先做 8.1 Part Alignment Evaluation
再做 8.2 Local Token Pooling Retrieval
再做 8.3 Class-evidence Weighted Local Pooling
最后再考虑 8.4 Part-aware Training
```

原因是：如果现有热区和关键 parts 明显不对齐，part-aware learning 的动机更强；如果 local token pooling 在 0 epoch 下已经提升 retrieval，说明局部 token 中确实有可利用信息，后续训练更值得投入。

## 预期结论形式

最终报告不只写数值，还要回答：

```text
1. 原图 Swin 是否更容易关注背景？
2. BBox crop Swin 是否更接近 beak / eye / wing？
3. 正确样本和错误样本的 part alignment 是否不同？
4. top-k retrieval 错误是否与 part alignment 差有关？
5. local feature fusion 是否能缓解喙、眼睛、翼部纹理混淆？
```

如果这些问题能被数据支持，Part-aware / Local Feature Learning 就会成为项目中比“盲目换大模型”更有说服力的优化方向。

## 参考

- Hierarchical Attention Vision Transformer for fine-grained visual classification, 2023：`https://www.sciencedirect.com/science/article/pii/S1047320323000056`
- Fine-Grained Image Retrieval with relation extraction / descriptor aggregation, 2023：`https://www.sciencedirect.com/science/article/abs/pii/S0031320323002431`
- CUB-200-2011 数据集与 part annotations：`https://data.caltech.edu/records/65de6-vp158`
