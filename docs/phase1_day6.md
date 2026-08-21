# Phase 1 Day 6：固定 Train / Validation / Test Split

## 目标

Day 6 的目标是把全局训练协议落实到代码里。正式实验不能继续直接使用 official test 选择 checkpoint，而要先从 official train 中固定切出 validation。

## What：新增什么

新增脚本：

```text
scripts/create_cub_splits.py
```

它会读取 CUB 官方 train/test 标记，并生成：

```text
data/processed/splits/cub_train_ids_seed42.txt
data/processed/splits/cub_val_ids_seed42.txt
data/processed/splits/cub_test_ids.txt
data/processed/splits/cub_split_manifest_seed42.json
```

新增数据能力：

```text
CUB200Dataset(..., image_ids={...})
```

这允许同一个 official split 内部再按固定 image id 文件筛选样本。

## Why：为什么要做

如果训练过程中反复查看 official test，并根据 test accuracy 调 learning rate、epoch、augmentation 或 backbone，那么 test set 就已经参与了模型选择，最终结果不再是可信的泛化评估。

正确流程：

```text
Official Train
        ↓
Stratified Training / Validation
        ↓
Validation 用于模型选择
        ↓
Official Test 只做最终评估
```

## How：如何运行

生成固定 split：

```powershell
.\.venv\Scripts\python.exe scripts\create_cub_splits.py --root data\raw\CUB_200_2011 --seed 42
```

正式 frozen baseline 配置：

```text
configs/baseline_resnet18_frozen_protocol.yaml
```

正式 full fine-tuning 配置：

```text
configs/baseline_resnet18_unfrozen_protocol.yaml
```

运行 frozen protocol：

```powershell
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_frozen_protocol.yaml --device cpu
```

运行 full fine-tuning protocol：

```powershell
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_unfrozen_protocol.yaml --device cpu
```

## 配置语义

正式配置中的关键字段：

```yaml
data:
  train_split: train
  val_split: train
  train_ids_path: data/processed/splits/cub_train_ids_seed42.txt
  val_ids_path: data/processed/splits/cub_val_ids_seed42.txt
```

这里 `train_split: train` 表示只在 CUB official train 内部取样；`train_ids_path` 和 `val_ids_path` 决定 official train 中哪些图用于训练，哪些图用于验证。

## 判断标准

完成 Day 6 后，需要确认：

- split 文件能生成。
- train / val 没有 image id 重叠。
- train + val 等于 official train 的 5,994 张。
- test 等于 official test 的 5,794 张。
- 训练脚本 metadata 记录了使用的 split 文件。

## 当前注意

之前 Day 3 的 frozen baseline 使用 official test 做验证，属于 bootstrap 实验。它能证明工程链路可用，但从正式实验表格开始，应改用本阶段生成的 validation split。
