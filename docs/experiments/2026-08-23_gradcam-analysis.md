# 实验 6.1：Grad-CAM Analysis

## 实验动机

Phase 6 不是装饰性可视化，而是用来回答：

```text
为什么模型取得这些结果？
```

前面 Phase 1 到 Phase 5 已经报告了 Accuracy、Recall@K、mAP 等数字，但这些指标不能告诉我们模型到底看到了什么。细粒度鸟类识别尤其容易受到背景、姿态、局部纹理和拍摄环境的影响，因此需要用 Grad-CAM 检查模型是否真的关注鸟类判别区域。

## 研究问题

```text
ResNet 在做细粒度预测时关注的是鸟类判别区域，还是背景 shortcut？
```

重点观察区域：

- Head
- Beak
- Wing
- Plumage
- Background

比较：

```text
正确预测
vs
错误预测
```

## 实验设置

本实验不训练新模型，只使用已经训练好的 checkpoint 做解释。

```text
Training Epoch = 0
```

默认分析 validation split：

```text
data/processed/splits/cub_val_ids_seed42.txt
```

Grad-CAM target layer：

```text
ResNet-18 backbone layer4[-1]
```

Grad-CAM target class：

```text
predicted class
```

解释错误样本时，使用 predicted class 作为反传目标。这表示我们解释的是：

```text
模型为什么会预测成这个错误类别？
```

而不是解释 true class。

## 模型选择

建议至少分析三组：

| 目的 | 模型 | 说明 |
| --- | --- | --- |
| CE baseline | `phase5_resnet18_ce_only` | 检查纯分类目标是否依赖背景 shortcut |
| 分类最优 joint model | `phase5_resnet18_ce_supcon_temp0_1` | Phase 5.5 中 Accuracy 最高 |
| 检索最优 joint model | `phase5_resnet18_ce_supcon_temp0_2` | Phase 5.5 中 Recall@K / mAP 最高 |

如果时间有限，优先分析：

```text
phase5_resnet18_ce_supcon_temp0_1
```

因为它是当前分类侧最强的 ResNet joint model。

## 训练指令

本实验不需要训练。

```powershell
# 无需训练，直接使用已有 best.pt checkpoint。
```

## 分析指令

### A. CE-only ResNet

```powershell
cd D:\code\VisionSearch-FG

$runCE = "20260823_084541_phase5_resnet18_ce_only"
.\.venv\Scripts\python.exe scripts\generate_gradcam_analysis.py `
  --config configs\phase5_resnet18_ce_only.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_only\$runCE\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --target-class predicted `
  --num-correct 12 `
  --num-wrong 12 `
  --device cpu `
  --output-dir outputs\explainability\gradcam_phase6
```

### B. 分类最优 joint model：tau=0.1

```powershell
cd D:\code\VisionSearch-FG

$runTau01 = "20260823_222949_phase5_resnet18_ce_supcon_temp0_1"
.\.venv\Scripts\python.exe scripts\generate_gradcam_analysis.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_1.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_1\$runTau01\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --target-class predicted `
  --num-correct 12 `
  --num-wrong 12 `
  --device cpu `
  --output-dir outputs\explainability\gradcam_phase6
```

### C. 检索最优 joint model：tau=0.2

```powershell
cd D:\code\VisionSearch-FG

$runTau02 = "20260823_222957_phase5_resnet18_ce_supcon_temp0_2"
.\.venv\Scripts\python.exe scripts\generate_gradcam_analysis.py `
  --config configs\phase5_resnet18_ce_supcon_temp0_2.yaml `
  --checkpoint "outputs\checkpoints\phase5_resnet18_ce_supcon_temp0_2\$runTau02\best.pt" `
  --split train `
  --ids-path data\processed\splits\cub_val_ids_seed42.txt `
  --target-class predicted `
  --num-correct 12 `
  --num-wrong 12 `
  --device cpu `
  --output-dir outputs\explainability\gradcam_phase6
```

每个 run 会输出：

```text
outputs/explainability/gradcam_phase6/<run_id>/
  summary.json
  gradcam_records.json
  gradcam_records.csv
  index.html
  images/
```

其中 `index.html` 用于浏览 Grad-CAM 图；`gradcam_records.csv` 用于人工标注。

## 人工标注表

对每张图人工填写：

| Image ID | Correct | True Class | Pred Class | Head | Beak | Wing | Plumage | Background | Notes |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | |

标注建议：

- `Head = yes/no/partial`
- `Beak = yes/no/partial`
- `Wing = yes/no/partial`
- `Plumage = yes/no/partial`
- `Background = yes/no/partial`

## 最终结果表

| Model | Scanned Images | Scanned Accuracy | Correct Samples | Wrong Samples | HTML Report | Manual Annotation |
| --- | ---: | ---: | --- | --- | --- | --- |
| CE-only ResNet | 1200 | 67.17% | 12 | 12 | `outputs/explainability/gradcam_phase6/20260823_084541_phase5_resnet18_ce_only/index.html` | 待人工标注 |
| CE + SupCon tau=0.1 | 1200 | 68.00% | 12 | 12 | `outputs/explainability/gradcam_phase6/20260823_222949_phase5_resnet18_ce_supcon_temp0_1/index.html` | 待人工标注 |
| CE + SupCon tau=0.2 | 1200 | 66.75% | 12 | 12 | `outputs/explainability/gradcam_phase6/20260823_222957_phase5_resnet18_ce_supcon_temp0_2/index.html` | 待人工标注 |

已生成文件：

| Model | Summary | Records CSV | Images |
| --- | --- | --- | --- |
| CE-only ResNet | `outputs/explainability/gradcam_phase6/20260823_084541_phase5_resnet18_ce_only/summary.json` | `outputs/explainability/gradcam_phase6/20260823_084541_phase5_resnet18_ce_only/gradcam_records.csv` | `outputs/explainability/gradcam_phase6/20260823_084541_phase5_resnet18_ce_only/images/` |
| CE + SupCon tau=0.1 | `outputs/explainability/gradcam_phase6/20260823_222949_phase5_resnet18_ce_supcon_temp0_1/summary.json` | `outputs/explainability/gradcam_phase6/20260823_222949_phase5_resnet18_ce_supcon_temp0_1/gradcam_records.csv` | `outputs/explainability/gradcam_phase6/20260823_222949_phase5_resnet18_ce_supcon_temp0_1/images/` |
| CE + SupCon tau=0.2 | `outputs/explainability/gradcam_phase6/20260823_222957_phase5_resnet18_ce_supcon_temp0_2/summary.json` | `outputs/explainability/gradcam_phase6/20260823_222957_phase5_resnet18_ce_supcon_temp0_2/gradcam_records.csv` | `outputs/explainability/gradcam_phase6/20260823_222957_phase5_resnet18_ce_supcon_temp0_2/images/` |

## 初步视觉观察

当前已完成 Grad-CAM 图像生成，但 `gradcam_records.csv` 中的人工标注字段仍为空。因此下面是少量样本抽查得到的初步观察，不作为最终统计结论。

抽查样本：

| Model | Image ID | Correct | True Class | Pred Class | 初步观察 |
| --- | ---: | --- | --- | --- | --- |
| CE-only ResNet | 6373 | 否 | `109.American_Redstart` | `021.Eastern_Towhee` | 热区主要覆盖鸟体，同时扩散到附近枝干；存在局部背景/枝干上下文参与。 |
| CE + SupCon tau=0.2 | 3310 | 是 | `058.Pigeon_Guillemot` | `058.Pigeon_Guillemot` | 热区偏向鸟体后半部、腿部和水面接触区域，不是单纯 head/beak 证据。 |
| CE + SupCon tau=0.2 | 8406 | 否 | `144.Common_Tern` | `147.Least_Tern` | 热区集中在鸟体中心和水面/轮廓附近，符合姿态与场景相似导致混淆的候选模式。 |
| CE + SupCon tau=0.1 | 3121 | 否 | `054.Blue_Grosbeak` | `014.Indigo_Bunting` | 热区覆盖鸟体和周围草丛，可能同时受 plumage 颜色和局部背景干扰。 |

初步判断：

```text
错误预测并不总是纯背景 shortcut。
不少错误样本的 CAM 仍然覆盖鸟体区域，但会混入枝干、水面、草丛等上下文。
```

这说明 Phase 6.1 的重点不应只问“是否看背景”，还应区分：

- 鸟体判别区域不足
- 姿态/轮廓相似
- plumage 颜色相似
- 背景或栖息环境相似
- 鸟体与背景边界处的混合证据

## 分析框架

正确预测样本重点看：

```text
CAM 是否覆盖 head / beak / wing / plumage 等鸟类判别区域？
```

错误预测样本重点看：

```text
CAM 是否落在背景、树枝、水面、天空、姿态轮廓或非判别性身体区域？
```

如果正确预测主要关注 head / beak / plumage，而错误预测更常关注 background，则说明模型存在 background shortcut。

如果正确和错误样本都关注鸟体，但错误集中在 wing / plumage 等局部相似区域，则说明失败来自细粒度类别本身的视觉相似性，而不是简单背景偏差。

如果 CE-only 更关注背景，而 CE + SupCon 更关注鸟体判别区域，则说明 representation learning 可能改善了模型的视觉证据来源。

## 当前阶段结论

本实验已经完成 Grad-CAM 生成流程，证明 Phase 6 的解释性分析 pipeline 可用。

目前可以确认：

```text
三组模型均已在 validation split 上生成 12 个高置信正确样本和 12 个高置信错误样本的 Grad-CAM。
```

但还不能给出最终的 Head / Beak / Wing / Plumage / Background 统计结论，因为人工标注尚未完成。

最终结论应回答：

```text
模型的预测证据是否与鸟类细粒度判别区域一致？
```

后续人工标注完成后，可能形成以下结论：

1. 模型主要关注鸟体局部判别区域，错误来自类别间细粒度相似性。
2. 模型在错误样本上明显依赖背景 shortcut，需要引入 foreground crop、bbox、attention 或 stronger augmentation。
3. SupCon joint training 使 CAM 从背景转向鸟体，说明 representation learning 不仅改善指标，也改善视觉证据。
4. SupCon 改善 retrieval，但 CAM 没有明显变化，说明 retrieval geometry 的改善不一定直接体现在分类证据区域上。

## 后续方向

如果发现背景 shortcut 明显，下一步优先做：

- bbox crop / foreground crop
- stronger background augmentation
- random erasing / cutmix
- attention-guided training

如果发现模型关注鸟体但仍错分，下一步优先做：

- 更高分辨率输入
- part-aware representation
- local feature aggregation
- fine-grained attribute supervision
