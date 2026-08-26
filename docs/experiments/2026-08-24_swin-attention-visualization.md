# 实验 6.2：Swin Attention Visualization

## 实验动机

Experiment 6.1 使用 Grad-CAM 分析 ResNet-18 是否关注鸟类判别区域。Experiment 6.2 进一步分析 Swin-Tiny 的视觉证据来源。

Swin-Tiny 是 Transformer-based backbone。它通过窗口注意力和层级特征建模局部纹理与较长距离上下文，因此可能与 CNN 使用不同的视觉区域完成细粒度识别。

## 研究问题

```text
CNN 与 Transformer 是否使用不同视觉区域完成细粒度识别？
```

对于 Swin，重点分析可视化区域是否覆盖：

- 局部细节
- 鸟主体
- 背景

## 方法说明

当前项目使用的是 `torchvision.models.swin_t`。该实现默认不直接返回每一层的 attention weights，因此本实验第一版使用：

```text
gradient-weighted final-stage Swin feature map
```

作为 Swin attention-style visualization。

它不是严格的 attention rollout，但可以回答一个实用问题：

```text
对当前预测类别最有贡献的 Swin final-stage spatial tokens 位于图像哪里？
```

因此本文档中将其称为：

```text
Swin attention-style visualization
```

后续如果需要更严格的 attention rollout，可以改用直接返回 attention weights 的 timm / 自定义 Swin block 实现。

## 实验设置

本实验不训练新模型，只使用 Phase 4.1 的 Swin-Tiny best checkpoint。

```text
Training Epoch = 0
```

模型：

```text
Swin-Tiny Full Fine-tuning
Run ID = 20260823_001810_baseline_swin_tiny_protocol
```

默认分析 validation split：

```text
data/processed/splits/cub_val_ids_seed42.txt
```

Visualization target class：

```text
predicted class
```

解释错误样本时，使用 predicted class 作为反传目标。这表示我们解释的是：

```text
Swin 为什么会预测成这个错误类别？
```

## 训练指令

本实验不需要训练。

```powershell
# 无需训练，直接使用已有 Swin-Tiny best.pt checkpoint。
```

## 分析指令

```powershell
cd D:\code\VisionSearch-FG

$runSwin = Get-ChildItem outputs\checkpoints\baseline_swin_tiny_protocol | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
.\.venv\Scripts\python.exe scripts\generate_swin_attention_visualization.py `
  --config configs\baseline_swin_tiny_protocol.yaml `
  --checkpoint "outputs\checkpoints\baseline_swin_tiny_protocol\$runSwin\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --target-class predicted `
  --num-correct 12 `
  --num-wrong 12 `
  --device cpu `
  --output-dir outputs\explainability\swin_attention_phase6
```

输出：

```text
outputs/explainability/swin_attention_phase6/<run_id>/
  summary.json
  attention_records.json
  attention_records.csv
  index.html
  images/
```

其中：

- `index.html` 用于浏览原图和 Swin attention-style overlay。
- `attention_records.csv` 用于人工标注。

## 人工标注表

对每张图人工填写：

| Image ID | Correct | True Class | Pred Class | Local Detail | Bird Body | Background | Notes |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

标注建议：

- `Local Detail = yes/no/partial`
- `Bird Body = yes/no/partial`
- `Background = yes/no/partial`

`Local Detail` 包括：

- head
- beak
- wing edge
- plumage pattern
- tail / leg 等细粒度局部区域

## 最终结果表

| Model | Scanned Images | Scanned Accuracy | Correct Samples | Wrong Samples | HTML Report | Manual Annotation |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny | 1200 | 75.50% | 12 | 12 | `outputs/explainability/swin_attention_phase6/20260823_001810_baseline_swin_tiny_protocol/index.html` | 已标注 24/24 |

已生成文件：

| File | Path |
| --- | --- |
| Summary | `outputs/explainability/swin_attention_phase6/20260823_001810_baseline_swin_tiny_protocol/summary.json` |
| Records CSV | `outputs/explainability/swin_attention_phase6/20260823_001810_baseline_swin_tiny_protocol/attention_records.csv` |
| HTML Report | `outputs/explainability/swin_attention_phase6/20260823_001810_baseline_swin_tiny_protocol/index.html` |
| Images | `outputs/explainability/swin_attention_phase6/20260823_001810_baseline_swin_tiny_protocol/images/` |

运行确认：

```text
num_scanned = 1200
accuracy = 75.50%
num_selected = 24
num_selected_correct = 12
num_selected_wrong = 12
visualization_type = gradient_weighted_swin_feature_map
```

## 初步视觉观察

当前已完成 Swin attention-style visualization 图像生成，并已对 `attention_records.csv` 中的 24 个样本完成 24/24 人工标注。标注口径为：`yes` 表示 attention 明显覆盖该区域，`partial` 表示弱覆盖或局部覆盖，`no` 表示基本不是主要证据来源。

| Model | Local Detail yes/partial/no | Bird Body yes/partial/no | Background yes/partial/no |
| --- | --- | --- | --- |
| Swin-Tiny | 10 / 14 / 0 | 22 / 2 / 0 | 14 / 10 / 0 |

统计结果说明，Swin-Tiny 的 attention-style map 几乎总能覆盖鸟主体，其中 22/24 为 `bird_body=yes`，但它也频繁吸收背景上下文，`background=yes` 达到 14/24。局部细节覆盖为 10/24 `yes` 和 14/24 `partial`，说明 Swin 并不是只依赖单个细粒度部位，而更倾向于把鸟体整体、局部纹理和周围场景一起纳入视觉证据。

抽查样本：

| Image ID | Correct | True Class | Pred Class | 初步观察 |
| ---: | --- | --- | --- | --- |
| 9994 | 是 | `170.Mourning_Warbler` | `170.Mourning_Warbler` | 可视化区域覆盖鸟主体，同时扩散到枝干和背景块；表现更偏全局，不只关注单个局部细节。 |
| 7652 | 否 | `131.Vesper_Sparrow` | `076.Dark_eyed_Junco` | 热区大面积覆盖鸟主体，周围树枝也有明显响应；可能是主体颜色/姿态相似与局部背景共同干扰。 |
| 3429 | 否 | `060.Glaucous_winged_Gull` | `062.Herring_Gull` | 热区覆盖鸟主体和水面/岸边背景，下方背景响应较强，是场景 shortcut 的候选案例。 |

初步判断：

```text
Swin 的可视化区域通常覆盖较大范围的鸟主体，同时也会吸收周围背景上下文。
错误样本既可能来自细粒度类别相似，也可能来自水面、树枝等场景区域干扰。
```

## 与 ResNet Grad-CAM 对照

最终应与 Experiment 6.1 对照：

| Backbone | Visualization | Correct Focus Pattern | Wrong Focus Pattern | Background Shortcut? | 结论 |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | Grad-CAM | 主要覆盖 plumage，head 经常为 partial，beak 最弱 | 错误样本仍多覆盖鸟体，但会混入枝条、水面、草丛等上下文 | 存在背景参与，但不是唯一错误来源 | 细粒度局部线索不足，尤其是 beak。 |
| Swin-Tiny | Attention-style map | 几乎稳定覆盖鸟主体，同时覆盖较大范围的局部纹理和颜色区域 | 错误样本也通常覆盖鸟体，但背景、遮挡物和姿态轮廓参与更明显 | 存在较强背景参与，14/24 为 background=yes | Swin 更偏全局证据整合，但更强 backbone 没有自动消除背景 shortcut。 |

重点回答：

```text
ResNet 是否更依赖局部纹理或背景？
Swin 是否更容易覆盖完整鸟主体？
Swin 是否使用更多上下文区域？
错误预测时，二者是否被相同背景或相同姿态误导？
```

## 分析框架

如果 Swin 的可视化区域更完整覆盖鸟主体，而 ResNet 更集中在局部纹理，则说明 Transformer backbone 可能通过更全局的 object-level evidence 完成细粒度识别。

如果 Swin 和 ResNet 都在错误样本上关注背景，则说明错误可能来自数据集偏置或背景 shortcut，而不是 backbone 类型。

如果 Swin 正确样本关注鸟主体，但错误样本关注背景或姿态轮廓，则说明 Swin 的全局建模能力仍可能被上下文 shortcut 误导。

如果 Swin 的错误样本仍关注鸟主体细节，但预测为相似类别，则说明问题主要来自细粒度类别边界，而不是视觉注意区域错误。

## 科研意义

该实验把 Phase 4 的 backbone 性能差异和 Phase 6 的解释性分析连接起来：

```text
Swin-Tiny Accuracy > ResNet-18 Accuracy
```

不应只停留在数字比较，还要进一步解释：

```text
性能提升是否来自更合理的视觉证据？
```

如果 Swin 更稳定地覆盖鸟主体和判别局部区域，那么可以支持：

```text
Transformer-based backbone 在细粒度识别中更善于整合局部细节与主体级上下文。
```

如果 Swin 仍明显依赖背景，则说明更强 backbone 并没有自动解决 shortcut learning，后续仍需要 foreground-aware 或 part-aware 方法。
