# Phase 1 Day 8：Augmentation Ablation

## 目标

本实验回答一个具体问题：在固定 ResNet-18 Full Fine-tuning、固定 train/val split、固定 optimizer、固定 learning rate、固定 batch size 的前提下，不同 training augmentation 对 CUB validation accuracy 有什么影响。这个实验只改变 `data.augmentation`，否则就不是严格 ablation。

## 实验设置

四组配置如下：

| Group | Augmentation | Config |
| --- | --- | --- |
| A | Basic | `configs/ablation_resnet18_fullft_aug_basic.yaml` |
| B | Basic + Horizontal Flip | `configs/ablation_resnet18_fullft_aug_hflip.yaml` |
| C | RandomResizedCrop | `configs/ablation_resnet18_fullft_aug_random_resized_crop.yaml` |
| D | RandomResizedCrop + Horizontal Flip + Mild ColorJitter | `configs/ablation_resnet18_fullft_aug_rrc_hflip_colorjitter.yaml` |

四组相同条件：

```text
dataset split: cub_train_ids_seed42 / cub_val_ids_seed42
backbone: pretrained ResNet-18
fine_tune_mode: full
optimizer: AdamW
learning_rate: 1e-4
batch_size: 16
max_epoch: 20
image_size: 224
```

唯一改变：

```text
data.augmentation
```

## 运行指令

建议按顺序跑，跑完一组再跑下一组，避免 CPU 和内存压力叠加。

```powershell
cd D:\code\VisionSearch-FG
$env:TORCH_HOME='D:\code\VisionSearch-FG\.torch_cache'

.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\ablation_resnet18_fullft_aug_basic.yaml --device cpu
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\ablation_resnet18_fullft_aug_hflip.yaml --device cpu
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\ablation_resnet18_fullft_aug_random_resized_crop.yaml --device cpu
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\ablation_resnet18_fullft_aug_rrc_hflip_colorjitter.yaml --device cpu
```

## 结果记录表

跑完后填写：

| Group | Augmentation | Run ID | Best Epoch | Best Val Acc | Val Top-5 at Best | Final Val Acc | Observation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| A | Basic | `20260822_134540_ablation_resnet18_fullft_aug_basic` | 19 | 64.83% | 87.58% | 64.08% | 无随机空间增强，表现稳定但低于 HFlip。 |
| B | HFlip | `20260822_144519_ablation_resnet18_fullft_aug_hflip` | 11 | 69.67% | 89.33% | 64.83% | 当前最优，说明左右翻转对 CUB 泛化有效。 |
| C | RandomResizedCrop | `20260822_183524_ablation_resnet18_fullft_aug_random_resized_crop` | 8 | 67.25% | 89.42% | 60.25% | best 较高但 final 下降明显，后期波动较大。 |
| D | RRC + HFlip + ColorJitter | `20260822_183526_ablation_resnet18_fullft_aug_rrc_hflip_colorjitter` | 12 | 68.08% | 89.50% | 63.92% | 次优，增强更丰富但不如单独 HFlip。 |

主指标仍然是 best validation accuracy；如果某组 final accuracy 明显低于 best accuracy，说明后期可能过拟合或训练不稳定。

## 解释原则

如果 `hflip` 优于 `basic`，说明左右翻转对鸟类分类是合理增强，因为鸟朝左或朝右通常不改变类别。如果 `random_resized_crop` 明显下降，可能说明裁剪过强，破坏了细粒度识别依赖的鸟嘴、眼睛、翅膀等局部线索。如果 `rrc_hflip_colorjitter` 优于其它组，说明适度的空间和颜色扰动提高了泛化能力；如果它下降，说明当前增强强度对 CUB 太激进。

## 本次结论

本次 ablation 中，`hflip` 是最优 augmentation，Best Val Acc 为 69.67%，高于 `basic` 的 64.83%。`random_resized_crop` 和 `rrc_hflip_colorjitter` 的 top-5 不低，但 final accuracy 下降更明显，说明较强随机裁剪可能提高了某些候选类别覆盖能力，同时也带来了更强训练波动。下一阶段建议把 `hflip` 作为 ResNet-18 CE baseline 的默认训练增强，并在后续加入 learning-rate scheduler 或 early stopping 后再重新评估更强增强是否有价值。
