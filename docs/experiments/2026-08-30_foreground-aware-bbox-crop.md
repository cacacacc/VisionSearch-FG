# 实验 7.1：Foreground-aware BBox Crop

## 实验动机

Phase 6 的 Grad-CAM 和 Swin attention-style visualization 显示，模型虽然通常能覆盖鸟主体，但背景、水面、树枝、天空、草地等上下文仍然频繁参与预测证据。Retrieval qualitative analysis 也显示，错误 top-k 经常受到背景、姿态、整体颜色和局部部位混淆影响。因此下一步需要验证：如果直接使用 CUB 官方 bounding box 裁剪鸟主体，是否能提升细粒度分类和检索结果。

## 研究问题

```text
减少背景区域后，模型是否能更稳定地利用鸟主体和局部细节，从而提升 Accuracy、Recall@K 和 mAP？
```

## 方法

本实验使用 CUB-200-2011 自带的 `bounding_boxes.txt`。训练和验证时先对图像执行 `bbox + margin` 裁剪，再进入原来的 resize、augmentation 和 normalization 流程。

```text
原图
-> image_id 对应 bounding box
-> bbox + 15% margin
-> clamp 到图像边界
-> Resize(224)
-> HFlip(train only)
-> Normalize
-> Swin-Tiny
```

该设置属于 annotation-assisted setting，因为它使用了 CUB 的额外 bbox 标注。结论应单独报告为 foreground-aware / bbox-assisted 结果，不能和不使用 bbox 的 baseline 混淆为完全同等输入条件。

## 对照设置

| 组别 | Config | Train 输入 | Val 输入 | 目的 |
| --- | --- | --- | --- | --- |
| A | `configs/baseline_swin_tiny_protocol.yaml` | 原图 224 | 原图 224 | 已有 Swin-Tiny baseline |
| B | `configs/foreground_swin_tiny_bbox224.yaml` | BBox crop 224 | BBox crop 224 | 验证 foreground-aware 是否提升 |

固定变量：

- Dataset：CUB-200-2011
- Split：固定 train / validation ids，seed=42
- Backbone：Swin-Tiny
- Pretraining：ImageNet pretrained
- Image size：224
- Augmentation：HFlip
- Optimizer：AdamW
- Checkpoint selection：最高 validation accuracy

唯一主要变化：

```text
crop_mode: none -> bbox
bbox_margin: 0.15
```

## 工程实现

当前已在 `CUB200Dataset` 中加入：

```yaml
data:
  crop_mode: bbox
  bbox_margin: 0.15
```

默认 `crop_mode: none`，因此旧实验协议不受影响。`train_baseline.py`、`train_contrastive.py`、`train_linear_probe.py`、`evaluate_ce_retrieval.py`、Grad-CAM 和 Swin attention visualization 脚本均会读取同一份配置，保证训练、评估和解释性分析输入一致。

## 工程验证

已完成单元测试和 CPU smoke test。CPU smoke test 使用真实 CUB 数据、`crop_mode=bbox`、`bbox_margin=0.15`、`pretrained=false`、`max_train_batches=1`、`max_val_batches=1`，用于验证数据读取、bbox crop、Swin-Tiny forward、训练 loop 和验证 loop 是否能完整跑通。该 smoke run 不作为科研指标，只作为工程连通性检查。

## 训练指令

GPU 正式训练：

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --device cuda
```

CPU smoke test：

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --pretrained false `
  --device cpu `
  --max-train-batches 1 `
  --max-val-batches 1 `
  --epochs 1 `
  --batch-size 2
```

## 检索评估指令

训练完成后，先用 validation split 做 retrieval evaluation：

```powershell
cd D:\code\VisionSearch-FG

$run = Get-ChildItem outputs\checkpoints\foreground_swin_tiny_bbox224 | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_ce_retrieval.py `
  --config configs\foreground_swin_tiny_bbox224.yaml `
  --checkpoint "outputs\checkpoints\foreground_swin_tiny_bbox224\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --metric cosine `
  --feature embedding `
  --device cpu `
  --output-dir outputs\embeddings\foreground_bbox_crop
```

## 最终结果

| Method | Run ID | Best Epoch | Val Acc | Macro-F1 | Top-5 Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny Original 224 | `20260823_012929_baseline_swin_tiny_protocol` | 4 | 75.50% | 75.28% | 93.42% | 61.25% | 84.33% | 90.67% | 49.04% |
| Swin-Tiny BBox Crop 224 | `20260830_141148_foreground_swin_tiny_bbox224` | 6 | 80.00% | 79.52% | 94.92% | 68.42% | 86.50% | 93.08% | 55.51% |

检索评估确认：

```text
split = validation split from official train
num_samples = 1200
feature = backbone embedding
embedding_dim = 768
metric = cosine
crop_mode = bbox
bbox_margin = 0.15
```

相对 Swin-Tiny 原图 baseline，BBox crop 224 的提升为：

```text
Val Acc   +4.50 percentage points
Macro-F1  +4.24 percentage points
Top-5 Acc +1.50 percentage points
Recall@1  +7.17 percentage points
Recall@5  +2.17 percentage points
Recall@10 +2.41 percentage points
mAP       +6.47 percentage points
```

## 结果解释

BBox crop 同时提升 Accuracy、Recall@K 和 mAP，说明背景 shortcut 是当前数值上限的重要限制，foreground-aware input 能直接改善分类和检索。这个结果与 Phase 6 的解释性分析一致：原图输入下，模型虽然能覆盖鸟主体，但背景、水面、枝条、天空和草地等上下文仍频繁参与预测证据。使用 bounding box 后，模型输入被约束到鸟主体及其少量上下文，因此更容易利用头部、喙、翼部和羽毛纹理等细粒度线索。

该结果不应被表述为“完全公平地打败原图 baseline”，因为 BBox crop 使用了 CUB 额外标注。更准确的说法是：在 annotation-assisted foreground-aware setting 下，Swin-Tiny 的分类和检索上限都显著提高。下一步可以继续做 `BBox crop 384`、`original + bbox feature fusion` 和 `BBox crop + SupCon / ArcFace`，分别验证更高分辨率、双视角特征融合和 metric learning 是否还能进一步提高数值。

后续 Experiment 7.2 已验证 `original + bbox feature fusion` 继续提升 retrieval：mAP 从 BBox crop 单分支的 55.51% 提升到 57.52%，Recall@1 从 68.42% 提升到 69.75%。这说明 BBox crop 是主要收益来源，但原图全局上下文仍能为检索排序提供补充信息。
