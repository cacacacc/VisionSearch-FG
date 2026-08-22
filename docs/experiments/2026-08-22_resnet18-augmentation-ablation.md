# ResNet-18 Augmentation Ablation

## Motivation

本实验比较 4 种 training augmentation 对 CUB-200-2011 validation accuracy 的影响。实验固定 ResNet-18 Full Fine-tuning、protocol split、AdamW、learning rate、batch size、epoch 和 image size，只改变 `data.augmentation`，因此结果可以主要归因于 augmentation 策略。

## Protocol

四组实验均使用 `data/processed/splits/cub_train_ids_seed42.txt` 和 `data/processed/splits/cub_val_ids_seed42.txt`，训练集 4,794 张，验证集 1,200 张。模型统一为 ImageNet pretrained ResNet-18 full fine-tuning，`epochs=20`，`batch_size=16`，`learning_rate=1e-4`，`weight_decay=1e-4`。Validation 不使用随机增强，因此每轮验证输入是确定性的。

## Result

| Group | Augmentation | Run ID | Best Epoch | Best Val Acc | Val Top-5 at Best | Best Val Loss | Final Val Acc | Best Top-5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Basic | `20260822_134540_ablation_resnet18_fullft_aug_basic` | 19 | 64.83% | 87.58% | 1.4690 | 64.08% | 88.50% |
| B | HFlip | `20260822_144519_ablation_resnet18_fullft_aug_hflip` | 11 | 69.67% | 89.33% | 1.2484 | 64.83% | 89.42% |
| C | RandomResizedCrop | `20260822_183524_ablation_resnet18_fullft_aug_random_resized_crop` | 8 | 67.25% | 89.42% | 1.3593 | 60.25% | 89.42% |
| D | RRC + HFlip + ColorJitter | `20260822_183526_ablation_resnet18_fullft_aug_rrc_hflip_colorjitter` | 12 | 68.08% | 89.50% | 1.2502 | 63.92% | 89.67% |

## Analysis

`HFlip` 是当前最优策略，Best Val Acc 达到 69.67%，比 `Basic` 高 4.84 个百分点。`RandomResizedCrop` 和 `RRC + HFlip + ColorJitter` 的 top-5 表现接近或略高，但 final accuracy 下降更明显，说明较强随机裁剪和颜色扰动可能增加训练难度和后期波动。对于 CUB 这种依赖局部细节的 fine-grained 任务，裁剪过强可能破坏鸟嘴、眼睛、翅膀等判别区域，因此简单的水平翻转反而是更稳的默认增强。

## Decision

后续 ResNet-18 CE baseline 默认采用 `hflip` 作为 training augmentation。更强增强暂不作为默认策略，除非后续加入 early stopping、learning-rate scheduler 或 bbox-aware crop 后重新验证。
