# 项目环境说明

## 目标

为 VisionSearch-FG 建立一个独立 `.venv`，避免混用系统 Python、conda base 和项目依赖。

## Python 版本

项目环境使用：

```text
Python 3.11
```

原因：

- PyTorch、torchvision、FAISS、Grad-CAM 在 Python 3.11 上有更稳定的 Windows wheel。
- 当前项目代码已经在 Python 3.11 上完成过数据读取和训练闭环验证。
- 避免使用 base 环境里的 Python 3.13，减少依赖兼容风险。

## 依赖选择逻辑

| 项目阶段 | 需要的能力 | 主要包 |
| --- | --- | --- |
| Phase 1 分类 baseline | ResNet-18、迁移学习、训练循环 | `torch`, `torchvision` |
| Phase 2 表征学习 | embedding、特征空间分析 | `torch`, `numpy`, `scikit-learn` |
| Phase 3 图像检索 | 向量检索、nearest neighbor | `faiss-cpu` |
| Phase 4 现代 backbone | ViT / Swin-Tiny | `timm` |
| Phase 5 对比学习 | loss 与实验分析 | `torch`, `numpy`, `pandas` |
| Phase 6 可解释性 | Grad-CAM、t-SNE、可视化 | `grad-cam`, `opencv-python`, `matplotlib`, `scikit-learn` |

## 为什么不用最新版全家桶

科研项目更看重可复现和稳定。这里没有直接追最新 `torch`，而是选择当前项目已经验证过的 CPU 组合：

```text
torch==2.11.0+cpu
torchvision==0.26.0+cpu
```

`numpy` 固定在 `1.26.4`，是为了减少 OpenCV、FAISS、Grad-CAM 等包之间的 ABI 兼容问题。

## 创建环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 验证环境

```powershell
python -m pytest tests
python scripts\inspect_cub_dataset.py --split train
python scripts\train_baseline.py --epochs 1 --batch-size 2 --pretrained false --freeze-backbone --max-train-batches 2 --max-val-batches 2 --device cpu
```

项目已将 pytest 临时目录固定到 `.pytest_tmp`，并关闭 pytest cache。这样可以避免 Windows 上不同 Python/conda/沙箱用户混用时出现临时目录权限问题。
