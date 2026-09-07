# VisionSearch-FG

基于 PyTorch 的细粒度视觉表征学习与图像检索项目，围绕 CUB-200-2011 数据集构建从分类 baseline、表征学习、检索评估到可解释性分析的完整实验链路。

项目目标不是单次追求最高分，而是建立一套可复现、可对照、可解释的 fine-grained visual retrieval 研究流程。

## 当前最终结果

当前推荐最终系统：

```text
ConvNeXt V2 Tiny BBox448
+ ArcFace margin=0.5, scale=16
+ BBox + BBox Flip feature averaging
+ Query Expansion top-3, alpha=0.5
```

该配置先在 CUB official train 内部 validation split 上完成模型选择和超参数选择，最后只在 official test split 上做最终评估。

| Split | Method | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | BBox + Flip TTA | 87.33% | 94.25% | 96.92% | 83.79% |
| Validation | BBox + Flip TTA + QE top3 alpha0.5 | 87.50% | 93.33% | 96.67% | 84.21% |
| Official Test | BBox + Flip TTA | 88.16% | 94.24% | 95.51% | 83.06% |
| Official Test | BBox + Flip TTA + QE top3 alpha0.5 | 88.30% | 93.63% | 94.99% | 83.59% |

最终测试文档：

```text
docs/experiments/2026-09-06_final-test-evaluation.md
```

## 研究问题

本项目主要回答以下问题：

1. ImageNet 预训练分类模型学到的 feature 是否天然适合 fine-grained retrieval？
2. 分类性能更高的 backbone 是否一定带来更好的 retrieval embedding？
3. foreground-aware input、分辨率提升、metric learning loss、post-processing 分别贡献多少？
4. 模型失败时主要混淆背景、姿态、整体颜色，还是喙、眼睛、头部等局部细节？
5. 如何在检索效果、embedding 维度、存储成本和查询延迟之间做 trade-off？

## 全局实验协议

正式实验遵守全局训练协议：

```text
docs/global_training_protocol.md
```

核心规则：

- CUB official test set 只用于最终评估。
- 模型选择、调参、best checkpoint 选择均基于 official train 内部划分出的 validation set。
- Retrieval validation 使用 validation images 同时作为 query 和 gallery，但必须排除 query 自身。
- 不同实验必须明确 controlled variables，只改变当前研究问题对应的变量。
- 训练结果、检索结果、定性分析和可解释性分析分别保存，避免混淆。

## 数据集

使用 CUB-200-2011：

```text
data/raw/CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── classes.txt
├── bounding_boxes.txt
└── parts/
```

生成固定 train / validation / test split：

```powershell
.\.venv\Scripts\python.exe scripts\create_cub_splits.py --root data\raw\CUB_200_2011 --seed 42
```

预期 split 文件：

```text
data/processed/splits/cub_train_ids_seed42.txt
data/processed/splits/cub_val_ids_seed42.txt
data/processed/splits/cub_test_ids.txt
```

## 环境配置

建议使用项目专用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

如果使用 CUDA，请以当前 PyTorch 官网命令安装匹配版本的 PyTorch。CPU 可以跑评估和小规模实验，正式训练建议使用 GPU。

## 项目结构

```text
VisionSearch-FG/
├── configs/                 # 实验配置：数据、模型、训练、输出
├── data/                    # 数据目录；真实 CUB 数据不提交
├── docs/                    # 中文实验记录、协议、阶段总结
├── outputs/                 # 训练日志、权重、embedding、分析结果
├── scripts/                 # 训练、评估、检索、可视化命令行入口
├── src/visionsearch_fg/     # 核心 Python 包
└── tests/                   # 单元测试与轻量行为检查
```

## 阶段路线

| 阶段 | 目标 | 主要内容 |
| --- | --- | --- |
| Phase 0 | 项目初始化 | 仓库结构、依赖、实验规范 |
| Phase 1 | 分类 baseline | ResNet-18 frozen / partial / full fine-tuning，augmentation ablation |
| Phase 2 | CE Retrieval Baseline | 直接抽取分类模型 backbone feature 做 retrieval |
| Phase 3 | 检索系统 | Brute force、FAISS exact、approximate retrieval |
| Phase 4 | Backbone 对比 | ResNet-18 vs Swin-Tiny，分类与检索对照 |
| Phase 5 | 表征学习目标 | CE、SupCon、CE+SupCon、loss weight、projection head、temperature、P x K sampling |
| Phase 6 | 可解释性分析 | Grad-CAM、Swin attention-style visualization、人工标注 |
| Phase 7 | Foreground-aware | CUB BBox crop、original/bbox fusion、PCA compression |
| Phase 8 | Part-aware / Local Feature | part alignment、head/wing/body oracle、local token pooling |
| Phase 9 | 输入分辨率 | BBox224 -> BBox448，验证局部细节分辨率收益 |
| Phase 10 | Backbone scaling | ConvNeXt V2、DINOv2 等强 backbone 对比 |
| Phase 11 | Angular Margin | ArcFace / CosFace，margin 与 scale ablation |
| Phase 12 | 强 backbone + SupCon | 在强 backbone 上验证 CE+SupCon 是否继续有效 |
| Phase 13 | Retrieval post-processing | Flip TTA、multi-view feature averaging、Query Expansion |
| Final | Official test evaluation | 固定最终 pipeline 后在 official test split 上一次性评估 |

## 关键实验总表

完整总表位于：

```text
outputs/experiments/overall_results/summary.csv
docs/experiments/2026-08-30_overall-experiment-results.md
```

核心结果摘录：

| 类别 | 方法 | Backbone / Input | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CNN baseline | ResNet-18 Full FT + HFlip | original 224 | 58.25% | 80.75% | 87.42% | 47.76% |
| Transformer baseline | Swin-Tiny Full FT + HFlip | original 224 | 61.25% | 84.33% | 90.67% | 49.04% |
| Foreground-aware | Swin-Tiny BBox Crop | bbox 224 | 68.42% | 86.50% | 93.08% | 55.51% |
| Fusion | Swin Original + BBox PCA | original+bbox 224 | 69.83% | 89.50% | 94.33% | 57.72% |
| Local feature | Swin BBox Evidence-weighted Local | bbox 224 | 71.25% | 87.83% | 93.58% | 59.62% |
| Strong backbone | ConvNeXt V2 Tiny BBox448 + CE | bbox 448 | 82.75% | 93.75% | 96.92% | 75.06% |
| Metric learning | ConvNeXt V2 Tiny BBox448 + ArcFace m0.5 s16 | bbox 448 | 86.33% | 93.58% | 96.42% | 82.72% |
| Final validation | ArcFace m0.5 s16 + BBox Flip TTA + QE | bbox+bbox_flip 448 | 87.50% | 93.33% | 96.67% | 84.21% |
| Final test | ArcFace m0.5 s16 + BBox Flip TTA + QE | bbox+bbox_flip 448 | 88.30% | 93.63% | 94.99% | 83.59% |

## 主要结论

1. ResNet-18 分类 baseline 可以形成基本 retrieval embedding，但 mAP 明显不足。
2. Swin-Tiny 分类精度高于 ResNet-18，但 retrieval 提升有限，说明 classification accuracy 与 retrieval quality 不等价。
3. BBox crop 是早期最稳定的提升来源，说明背景噪声会显著干扰 fine-grained retrieval。
4. Local feature 分析显示，模型确实需要更稳定地关注喙、眼睛、头部、翼部纹理等细粒度区域。
5. ConvNeXt V2 Tiny + BBox448 显著提高 retrieval，说明 foreground-aware input 与高分辨率细节是关键。
6. ArcFace 比普通 CE 更适合 cosine retrieval 场景，`margin=0.5, scale=16` 是当前最优配置。
7. Flip TTA 与 Query Expansion 能继续小幅提升 mAP，但后处理收益小于 backbone、分辨率和训练目标带来的收益。

## 常用命令

ResNet 分类训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\ablation_resnet18_fullft_aug_hflip.yaml `
  --device cuda
```

Swin-Tiny 分类训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --device cuda
```

ConvNeXt V2 Tiny BBox448 CE 训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py `
  --config configs\backbone_convnextv2_tiny_bbox448.yaml `
  --device cuda
```

ArcFace 训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_margin_classifier.py `
  --config configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml `
  --device cuda
```

Validation retrieval 评估：

```powershell
$expName = "foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16"
$run = Get-ChildItem "outputs\checkpoints\$expName" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty Name

.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml `
  --checkpoint "outputs\checkpoints\$expName\$run\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --feature embedding `
  --views bbox bbox_flip `
  --qe-top-k 0 3 `
  --qe-alpha 0.0 0.5 `
  --device cuda `
  --output-dir outputs\embeddings\arcface_tta_view_ablation
```

Official test retrieval 评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_postprocessing.py `
  --config configs\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16.yaml `
  --checkpoint outputs\checkpoints\foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16\20260905_141401_foreground_convnextv2_tiny_bbox448_arcface_m0_5_s16\best.pt `
  --split test `
  --ids-path data\processed\splits\cub_test_ids.txt `
  --feature embedding `
  --views bbox bbox_flip `
  --qe-top-k 0 3 `
  --qe-alpha 0.0 0.5 `
  --device cuda `
  --output-dir outputs\embeddings\final_test
```

## 重要文档

```text
docs/global_training_protocol.md
docs/experiments/2026-08-30_overall-experiment-results.md
docs/experiments/2026-08-30_foreground-aware-bbox-crop.md
docs/experiments/2026-08-30_part-aware-local-feature-learning.md
docs/experiments/2026-09-02_angular-margin-arcface-cosface.md
docs/experiments/2026-09-05_arcface-margin-scale-ablation.md
docs/experiments/2026-09-06_tta-view-ablation.md
docs/experiments/2026-09-06_final-test-evaluation.md
```

## 当前状态

项目已经完成从 baseline 到最终 official test evaluation 的完整闭环。后续如果继续优化，优先方向应是：

1. 更可靠的 part-aware local descriptor，而不是硬裁剪。
2. 更强的 re-ranking 或 local descriptor matching。
3. 在强 backbone 上重新系统比较 CE、ArcFace、SupCon、ArcFace+SupCon。
4. 检查最终 pipeline 在更多随机 seed 或不同 validation split 下的稳定性。
