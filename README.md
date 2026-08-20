# VisionSearch-FG

基于 PyTorch 的细粒度视觉表征学习与图像检索系统。

本项目目标是构建一个轻量、可解释、可实验复现的细粒度视觉理解系统，围绕 CUB-200-2011 数据集完成：

1. 细粒度图像分类
2. 视觉表征学习
3. 深度图像检索
4. 可解释性分析

## 研究策略

项目采用：

```text
Baseline -> Improvement -> Analysis
```

不直接追求 SOTA，而是先建立可靠 baseline，再逐步加入 representation learning、retrieval 和 explainability 分析。

## 阶段路线

| 阶段 | 目标 | 主要方法 |
| --- | --- | --- |
| Phase 0 | 环境与仓库初始化 | 项目结构、依赖、实验规范 |
| Phase 1 | 细粒度分类 baseline | ResNet-18 + transfer learning |
| Phase 2 | 视觉表征学习 | feature embedding extraction |
| Phase 3 | 图像检索 | FAISS / nearest neighbor retrieval |
| Phase 4 | 现代 backbone 对比 | ResNet-18 vs Swin-Tiny |
| Phase 5 | 表征改进 | classification loss + supervised contrastive loss |
| Phase 6 | 可解释性分析 | Grad-CAM, attention visualization, t-SNE |

## 项目结构

```text
VisionSearch-FG/
├── configs/                 # 实验配置：数据路径、模型、训练超参数
├── data/                    # 数据集目录，只保留占位文件，不提交真实数据
├── docs/                    # 论文阅读、实验记录、阶段总结
├── notebooks/               # 探索性分析与可视化
├── outputs/                 # 训练日志、权重、embedding、图像结果
├── scripts/                 # 命令行入口，例如训练、评估、检索
├── src/visionsearch_fg/     # 项目核心 Python 包
└── tests/                   # 单元测试与轻量级行为检查
```

## 环境配置

当前本机检测到 Python 版本：

```text
Python 3.11.9
```

建议先使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果安装 PyTorch 时需要区分 CPU / CUDA 版本，请以后以 PyTorch 官网给出的命令为准。轻薄本优先使用 CPU 或低显存友好的配置。

## Phase 0 Day 1

今天只完成四件事：

1. 初始化 Git 仓库
2. 建立科研项目目录
3. 准备最小依赖说明
4. 明确每个模块为什么存在

下一步会进入 Phase 1：实现 ResNet-18 + CUB-200-2011 分类 baseline。

## Phase 1 冒烟测试

在没有真实 CUB 数据集时，可以先运行伪造数据集冒烟测试：

```powershell
python scripts\smoke_test_cub_dataset.py
```

验证 ResNet-18 的输入、embedding、logits 形状：

```powershell
python scripts\smoke_test_resnet18.py
```

下载并解压 CUB-200-2011 后，确认目录类似：

```text
data/raw/CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── classes.txt
```

然后检查真实数据集：

```powershell
python scripts\inspect_cub_dataset.py --split train
```

## Phase 1 训练闭环

CPU 冒烟训练：

```powershell
python scripts\train_baseline.py --epochs 1 --batch-size 2 --pretrained false --freeze-backbone --max-train-batches 2 --max-val-batches 2 --device cpu
```

这个命令只用于验证训练机制，不代表正式实验结果。
