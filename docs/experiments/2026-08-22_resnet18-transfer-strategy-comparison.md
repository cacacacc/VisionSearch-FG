# ResNet-18 Transfer Strategy Comparison

## Motivation

本实验比较 ResNet-18 在 CUB-200-2011 上的三种迁移学习策略：Frozen Backbone、Partial Fine-tuning 和 Full Fine-tuning。目标是判断在固定数据划分、固定评价协议下，是否值得解冻 backbone，以及 partial fine-tuning 是否能在计算成本和精度之间取得更好的折中。

## Protocol

三个实验均使用 `Global Training Protocol` 中生成的固定 split：`data/processed/splits/cub_train_ids_seed42.txt` 和 `data/processed/splits/cub_val_ids_seed42.txt`。训练集为 4,794 张，验证集为 1,200 张，official test 不参与模型选择。本表只报告 validation 结果，后续最终模型确定后再使用 official test 做一次性评估。

## Result

| Strategy | Config | Run ID | Max Epoch | Batch | Trainable Params | Best Epoch | Best Val Acc | Val Top-5 at Best | Best Val Loss | Final Val Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | `baseline_resnet18_frozen_protocol.yaml` | `20260821_215656_baseline_resnet18_frozen_protocol` | 10 | 32 | 102,600 | 10 | 50.75% | 77.92% | 1.9361 | 50.75% |
| Partial FT | `baseline_resnet18_partial_protocol.yaml` | `20260822_113017_baseline_resnet18_partial_protocol` | 25 | 16 | 8,496,328 | 6 | 61.25% | 86.08% | 1.4551 | 53.75% |
| Full FT | `baseline_resnet18_unfrozen_protocol.yaml` | `20260821_225256_baseline_resnet18_unfrozen_protocol` | 25 | 16 | 11,279,112 | 11 | 68.42% | 89.25% | 1.2416 | 62.92% |

补充观察：Partial FT 的 Val Acc 在 epoch 6、8、10 都达到 61.25%，但当前 checkpoint 保存逻辑只在 `val_accuracy > best_accuracy` 时更新，因此 `best.pt` 对应第一次达到最高值的 epoch 6。Full FT 的最高 Val Top-5 出现在 epoch 14，为 90.17%，但最高 Val Acc 出现在 epoch 11，因此以分类准确率作为主指标时应选择 epoch 11 的 `best.pt`。

## Analysis

Full Fine-tuning 是三者中最强的策略，Best Val Acc 比 Frozen 高 17.67 个百分点，比 Partial FT 高 7.17 个百分点，说明 CUB 的细粒度识别明显受益于 backbone 适配。Partial FT 也显著优于 Frozen，说明只解冻 `layer4` 已经能带来有效收益，但它在 epoch 6 后很快出现验证集波动，训练准确率接近 100%，存在过拟合迹象。Frozen 的训练成本最低、结果最稳定，但上限明显较低，更适合作为轻量 baseline，而不是最终分类模型。

## Next Step

下一步应对三者的 best checkpoint 运行同一套 validation error analysis，比较错误样本类型是否发生变化。随后可以优先对 Full FT 增加 early stopping、学习率调度或更温和的数据增强，目标是保留 epoch 11 前后的高精度，同时降低后期过拟合波动。
