# 学习使命：VisionSearch-FG

## 为什么学习

通过完成一个科研级细粒度视觉识别与图像检索项目，系统掌握 PyTorch 工程、计算机视觉方法、论文阅读、实验设计和结果分析能力。这个项目不是为了第一时间追求 SOTA，而是为了学会如何从可靠 baseline 出发，逐步形成可复现、可解释、可比较的研究流程。

## 成功标准

- 能训练并评估 ResNet-18 细粒度分类 baseline。
- 能从分类模型中提取 visual embedding，并用于图像检索。
- 能在轻薄本硬件约束下比较 CNN 与轻量现代 backbone。
- 能使用 Grad-CAM、attention visualization、t-SNE 分析模型行为。
- 能写出包含 Motivation、Hypothesis、Method、Metric、Analysis 的实验记录。

## 约束

- 设备是轻薄本，优先使用迁移学习、预训练模型和轻量模型。
- baseline 稳定前，不做大规模训练，也不从零训练大型 Transformer。
- 写代码前先解释：模块为什么存在、解决什么问题、位于 pipeline 哪里、输入输出是什么、tensor shape 如何变化。
- teach 技能维护的学习文档统一使用中文。

## 暂不做的事

- 不把第一目标设为追求最高榜单精度。
- 不从零训练大型视觉 Transformer。
- 不在 baseline 可复现之前叠加复杂研究技巧。
