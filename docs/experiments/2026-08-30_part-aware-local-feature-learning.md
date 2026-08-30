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
