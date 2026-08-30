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

### 方法草案

使用同一个 Swin checkpoint，提取最后 stage 的 token / feature map：

```text
Global feature = 原始 backbone pooling
Local feature  = Top-K 高响应 token pooling
Final feature  = L2Normalize([global; local])
```

对照组：

| Method | 是否训练 | Feature |
| --- | --- | --- |
| Swin global | 0 epoch | global backbone h |
| Swin local top-k | 0 epoch | local pooled tokens |
| Swin global + local | 0 epoch | concat(global, local) |
| Swin bbox global + local | 0 epoch | bbox crop 上的 concat(global, local) |

该实验可以先作为 feature extraction / retrieval evaluation，不需要训练。若它有效，再进入真正的 part-aware training。

## 实验 8.3：Part-aware Training

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
最后再考虑 8.3 Part-aware Training
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
