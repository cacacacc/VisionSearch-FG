# Phase 1 Day 1：CUB 数据集与 ResNet baseline 外壳

## 目标

今天先搭建最小 baseline 基础设施：解析 CUB 元数据、生成图像 tensor，并定义后续可以训练的 ResNet-18 分类模型。

## What：是什么

PyTorch 的 `Dataset` 是一个“索引到样本”的映射对象。给它一个整数下标，它返回一个训练样本。

在 CUB 细粒度分类任务中，一个样本包含：

```text
图像、类别标签、元数据
```

经过 transform 和 DataLoader 组 batch 后，主要 tensor 形状是：

```text
单张图像 image: [3, 224, 224]
一批图像 batch images: [B, 3, 224, 224]
类别标签 labels: [B]
分类输出 logits: [B, 200]
视觉表征 embedding: [B, 512]
```

## Why：为什么需要

细粒度视觉识别非常依赖干净的数据处理。如果类别标签、训练/测试划分或图像预处理出错，后面再复杂的模型改进都会失去意义，因为实验比较到的可能只是数据错误带来的噪声。

## Where：在 pipeline 的位置

数据集模块位于整个 pipeline 的最前面：

```text
CUB 元数据文件
-> CUB200Dataset
-> DataLoader
-> ResNet-18
-> logits / embedding
```

## How：如何实现

CUB 提供了一组文本格式的元数据文件：

| 文件 | 作用 |
| --- | --- |
| `images.txt` | 将 image id 映射到相对图片路径 |
| `image_class_labels.txt` | 将 image id 映射到类别 id |
| `train_test_split.txt` | 将 image id 映射到训练集/测试集划分 |
| `classes.txt` | 将类别 id 映射到类别名称 |

数据集读取时会把 CUB 的 1-based 类别编号转换成 0-based 标签。原因是 PyTorch 的 `CrossEntropyLoss` 要求 target label 的范围是 `0` 到 `num_classes - 1`。

## Trade-off：取舍

第一版数据集类刻意保持简单。虽然 CUB 提供 bounding box 和 part annotation，但这一阶段暂时不用。这样可以让第一个 baseline 专注于分类主线，也方便后续在可解释性阶段再讨论“模型是否真的看到了鸟的关键区域”。

## Research Value：科研价值

这是项目里的第一个可复现边界。只要数据解析稳定，后面比较 ResNet、Swin-Tiny、对比学习损失和检索指标时，就不需要反复怀疑训练/测试划分是否被意外改变。

## 下一步

当 fake data 或真实 CUB 文件夹上的 smoke test 通过后，下一课进入真正的训练循环：

```text
batch -> forward -> loss -> backward -> optimizer.step -> validation metric
```
