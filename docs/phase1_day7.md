# Phase 1 Day 7：Partial Fine-tuning 配置

## 目标

Day 7 的目标是加入 ResNet-18 的 partial fine-tuning，使 Phase 1 可以正式比较三种迁移学习策略：

```text
Frozen Backbone
Partial Fine-tuning
Full Fine-tuning
```

## Partial Fine-tuning 定义

本项目第一版 partial fine-tuning 采用：

```text
conv1 / bn1 / layer1 / layer2 / layer3: frozen
layer4: trainable
classifier: trainable
```

也就是说，只让 ResNet-18 最靠后的语义层和分类头适配 CUB。

## 为什么这样设计

越靠前的卷积层越偏向边缘、纹理、颜色等通用视觉特征；越靠后的层越偏向类别语义。CUB 的细粒度差异主要体现在局部语义特征上，因此先解冻 `layer4` 是一个计算成本和适配能力之间的折中。

## 学习率

配置：

```text
Backbone LR:   1e-4
Classifier LR: 1e-3
```

原则：

```text
LR_backbone < LR_classifier
```

原因：backbone 来自 ImageNet 预训练，不希望更新过大；classifier 是新初始化层，需要更快学习。

## 运行指令

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_partial_protocol.yaml --device cpu
```

## 对照关系

正式比较时应使用同一套 split：

```text
data/processed/splits/cub_train_ids_seed42.txt
data/processed/splits/cub_val_ids_seed42.txt
```

对照表：

| Strategy | Config |
| --- | --- |
| Frozen | `configs/baseline_resnet18_frozen_protocol.yaml` |
| Partial FT | `configs/baseline_resnet18_partial_protocol.yaml` |
| Full FT | `configs/baseline_resnet18_unfrozen_protocol.yaml` |

指标以 validation best checkpoint 为准，不直接使用 official test 调参。
