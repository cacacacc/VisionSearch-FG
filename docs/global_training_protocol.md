# Global Training Protocol：全局训练协议

## 目标

本协议定义 VisionSearch-FG 从 Phase 1 到 Phase 6 的统一数据划分、训练预算、验证流程和最终测试规则。

项目的实验价值不只来自模型精度，还来自不同实验之间是否可比较。所有正式实验必须尽量控制变量：固定数据划分、固定评价协议、清楚记录训练预算，并只在最终阶段使用 official test set。

## 1. Dataset Protocol

数据集使用：

```text
CUB-200-2011
```

数据规模：

```text
Total Images: 11,788
Classes: 200

Official Train: 5,994
Official Test:  5,794
```

CUB-200-2011 包含 bounding box、part location 和 attributes 等标注，但本项目第一版 baseline 只使用：

```text
image + class label
```

原因：不额外使用 bbox、part、attribute，可以避免给某些 baseline 带来额外 annotation 优势。

参考资料：

- [CUB-200-2011 技术报告](https://authors.library.caltech.edu/records/cvm3y-5hh21/files/CUB_200_2011.pdf)
- [Caltech 数据集页面](https://www.vision.caltech.edu/datasets/)

## 2. Train / Validation / Test 划分

不要把全部 11,788 张图片重新随机拆成 80/10/10。这样会破坏 CUB 官方 test protocol。

正式实验采用：

```text
Official CUB Dataset
|
+-- Official Train: 5,994
|       |
|       +-- Training:   about 4,795 (80%)
|       |
|       +-- Validation: about 1,199 (20%)
|
+-- Official Test: 5,794
```

`Official Train -> Training / Validation` 使用固定随机种子和 stratified split，保证 200 个类别在 training 和 validation 中都存在。

固定：

```text
random_seed = 42
```

并保存划分结果：

```text
data/processed/splits/cub_train_ids_seed42.txt
data/processed/splits/cub_val_ids_seed42.txt
data/processed/splits/cub_test_ids.txt
```

## 3. 三部分严格分工

Training set：

```text
用于 loss.backward() 和 optimizer.step()
```

Validation set：

```text
用于选择 learning rate、lambda、embedding dimension、augmentation、backbone、best checkpoint
```

Test set：

```text
不参与模型选择，只用于最终结果报告
```

禁止流程：

```text
不断看 Test Accuracy
        ↓
根据 Test 调参数
        ↓
Test 实际变成 Validation
        ↓
最终结果存在 Test Leakage
```

因此，在 Phase 1 到 Phase 5 的开发阶段，主要看 validation。official test 只在模型和协议固定后使用。

## 4. 固定 Split

整个项目从 Phase 1 到 Phase 6 使用完全相同的数据划分。

这些实验必须共享同一个 split：

```text
ResNet
Swin
CrossEntropy
SupCon
CrossEntropy + SupCon
Grad-CAM / Error Analysis
Retrieval
```

如果 ResNet 使用 Split A，Swin 使用 Split B，那么两者最终精度差异无法归因于模型本身。

## 5. Image Resolution

第一版统一：

```text
224 x 224
```

核心 tensor shape：

```text
[B, 3, 224, 224]
```

原因：

- ResNet-18 和 pretrained Swin-Tiny 都能方便使用 224 输入。
- 当前计算资源有限，先降低训练成本。
- 224 是一个合理的迁移学习起点。

后续可独立做 resolution ablation：

```text
224 x 224
vs
384 / 448
```

该实验研究 fine-grained recognition 是否受益于更高分辨率，但不应该在 baseline 阶段引入。

## 6. Data Augmentation

Training transform：

```text
Resize / RandomResizedCrop
        ↓
Random Horizontal Flip
        ↓
Mild Color Jitter
        ↓
ToTensor
        ↓
ImageNet Normalization
```

禁止默认使用：

```text
Vertical Flip
```

原因：倒置鸟类通常不符合真实数据分布。

Validation / Test transform：

```text
Resize
↓
Center Crop
↓
ToTensor
↓
ImageNet Normalization
```

Validation 和 test 不使用随机 augmentation，这样同一模型每次得到一致输入。

## 7. Batch Size

ResNet-18 第一版：

```text
batch_size = 32
```

如果显存或内存不足：

```text
batch_size = 16
```

Swin-Tiny：

```text
batch_size = 16
```

如果显存或内存不足：

```text
batch_size = 8
```

SupCon 需要特别注意：batch 中的其他样本会成为 positive / negative，因此 batch size 会影响 contrastive learning 本身。

建议：

```text
16-32 original images / batch
```

如果每张图生成两个 augmented views：

```text
16 original images
        ↓
2 views each
        ↓
32 actual image views
```

## 8. Epoch Protocol

不要简单规定所有模型都训练 100 epochs。CUB 数据量较小，且本项目主要使用 pretrained model。

推荐训练预算：

| 类型 | 用途 | Epoch |
| --- | --- | ---: |
| Debug Run | 验证代码能跑 | 1-2 |
| Pilot Experiment | 验证 pipeline 和初步趋势 | 5 |
| Formal Experiment | 正式分类实验 | 20-30 |
| SupCon / Joint Training | 表征学习正式实验 | 30-40 |

注意：

```text
5 epoch pilot 只能判断 pipeline 是否正常，不能作为论文式正式结果。
```

## 9. Phase 1：ResNet-18 Baseline

### Experiment 1.1：Frozen Backbone

```text
Dataset:
Training    about 4,795
Validation  about 1,199
Test        5,794

Input:
224 x 224

Batch:
32

Epoch:
10

Optimizer:
AdamW

Head LR:
1e-3

Loss:
CrossEntropy
```

冻结结构：

```text
ImageNet ResNet-18 backbone: frozen
Classifier head: trainable
```

因为只训练 classifier，通常收敛比完整 fine-tuning 更快。

### Experiment 1.2：Partial Fine-tuning

结构：

```text
Early ResNet layers: frozen
Later ResNet layers: trainable
Classifier: trainable
```

推荐：

```text
Epochs: 20-25
Batch: 32 / 16
Backbone LR: 1e-4
Classifier LR: 1e-3
Weight Decay: 1e-4
Early Stopping: 5 epochs
```

原则：

```text
LR_backbone < LR_classifier
```

原因：pretrained backbone 已经包含有用 visual representation，backbone 更新幅度不宜过大。

### Experiment 1.3：Full Fine-tuning

结构：

```text
ResNet-18 backbone: trainable
Classifier: trainable
```

推荐：

```text
Epoch: 25-30
Batch: 16-32
Backbone LR: 1e-4
Classifier LR: 5e-4 ~ 1e-3
Early Stopping: 5
```

比较表：

| Strategy | Max Epoch | Best Epoch | Val Acc | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 10 | | | |
| Partial FT | 25 | | | |
| Full FT | 30 | | | |

正式结果使用 best validation checkpoint，不默认使用最后一个 epoch。

### Experiment 1.4：Augmentation Ablation

建议比较：

```text
A. Basic
B. + Horizontal Flip
C. + RandomResizedCrop
D. + Crop + Flip + Mild ColorJitter
```

每组：

```text
Max Epoch: 20
```

固定：

```text
dataset split
backbone
optimizer
learning rate
batch size
```

唯一改变：

```text
augmentation
```

## 10. Phase 2：Visual Embedding Baseline

Phase 2 第一部分不需要重新训练。

直接使用 Phase 1 的最佳 checkpoint：

```text
Image
↓
Best ResNet-18
↓
Remove Classification Layer
↓
512-D Feature
```

因此：

```text
Training Epoch = 0
```

研究问题：

```text
单纯由 CrossEntropy 学出来的 feature 有多少 retrieval 能力？
```

这会建立 CE Retrieval Baseline。

## 11. Phase 2 Retrieval Validation Split

开发过程中不要立即使用 official test。Phase 2 的 CE Retrieval Baseline、metric ablation、normalization ablation、dimension ablation 和后续 SupCon / Swin retrieval 对照，默认都先在 validation split 上完成。

```text
Protocol target:
Validation = about 1,199 images

Current saved split:
data/processed/splits/cub_val_ids_seed42.txt = 1,200 images
```

将 validation 图片同时作为 query 和 gallery：

```text
Query Image
↓
Search remaining Validation Images
```

必须排除 query 自身：

```text
Gallery = Validation Images - Query Itself
```

否则会出现：

```text
query embedding
vs
self embedding

cosine similarity = 1
```

这会让 Top-1 永远先命中自己，得到假的高分。最终模型、checkpoint、metric、normalization、embedding dimension 和检索协议全部确定后，才在 official test 上运行完全相同 protocol：

```text
Official Test = 5,794 images
Query = Official Test Image
Gallery = Official Test Images - Query Itself
```

## 12. Phase 3：FAISS

FAISS 阶段：

```text
Training Epoch = 0
```

FAISS 不是神经网络训练，它负责：

```text
Embedding
↓
Index
↓
Nearest Neighbor Search
```

开发阶段：

```text
Validation Gallery: about 1,199
```

最终阶段：

```text
Official Test Gallery: 5,794
```

评价：

```text
Recall@1
Recall@5
Recall@10
mAP
Query Latency
Indexing Time
```

### Experiment 3.1：Brute Force vs FAISS Exact Search

研究问题：

```text
在保持检索结果相同的情况下，FAISS 能否降低向量搜索成本？
```

固定：

```text
Embedding
Validation split
Query / gallery protocol
Exclude query itself
L2 normalization
Cosine-equivalent inner product search
```

唯一改变：

```text
Search backend:
Brute Force NumPy
vs
FAISS IndexFlatIP
```

预期：

```text
Recall@K 和 mAP 完全一致
Ranking 完全一致或只在极少数浮点 tie 上不同
Indexing time / query latency 用于工程成本比较
```

### Experiment 3.2：Exact vs Approximate Retrieval

比较：

```text
Exact Search
vs
Approximate Search

Flat
IVF
HNSW
```

研究问题：

```text
能够牺牲多少 Retrieval Accuracy 换取多少搜索速度？
```

评价：

```text
Search Recall@1 / 5 / 10
Retrieval Recall@1 / 5 / 10
Query Time
Indexing Time
Index Memory
```

Search Recall 用 exact Flat top-K 作为 reference，衡量 ANN 是否找回同一批 nearest neighbors。Retrieval Recall 衡量 ANN 的近似误差是否真正影响类别检索任务。该实验建立典型 ANN trade-off：

```text
Accuracy ↔ Speed
```

## 13. Phase 4：Swin-Tiny

公平比较仍然使用：

```text
Training    about 4,795
Validation  about 1,199
Test        5,794
```

输入：

```text
224 x 224
```

推荐：

```text
Pretrained Swin-Tiny
Max Epoch: 25-30
Batch: 8-16
Optimizer: AdamW
Backbone LR: 5e-5 ~ 1e-4
Head LR: 5e-4 ~ 1e-3
Weight Decay: 1e-4
Early Stopping: 5
```

科研上的公平不是所有模型必须使用完全相同 learning rate。公平是：

```text
数据一致
评价协议一致
训练预算可比
每个模型使用合理配置
完整报告配置
```

Experiment 4.1 比较：

```text
ResNet-18
vs
Swin-Tiny
```

评价：

```text
Accuracy
Macro-F1
Top-5 Accuracy
Parameter Count
Training Time
```

这里不要求 ResNet 和 Swin 的 learning rate 完全相同。公平比较的核心是数据、split、resolution、augmentation、训练预算和 evaluation protocol 可比，同时每个模型使用合理训练配置。

Experiment 4.2 比较：

```text
ResNet Feature
vs
Swin Feature
```

执行完全相同 retrieval protocol：

```text
Validation query/gallery
Exclude query itself
L2-normalized cosine similarity
Recall@1 / Recall@5 / Recall@10 / mAP
```

研究问题：

```text
分类表现更好的 backbone 是否一定产生更好的 retrieval embedding？
```

如果出现 `Swin Accuracy > ResNet` 但 `Swin Recall@10 ≈ ResNet`，这不是失败结果，而是说明 classification objective 和 retrieval embedding geometry 不完全等价，后续 SupCon 或 metric learning 更有研究必要。

## 14. Phase 5：SupCon / Multi-task Learning

模型结构：

```text
                     +-> Classifier -> CE
Image -> Backbone -> h
                     +-> Projection Head -> z -> SupCon
```

总 loss：

```text
L = L_CE + lambda * L_SupCon
```

建议正式训练：

```text
Max Epoch: 30-40
Batch: 16 original images, if possible 32
Views: 2 augmentations / image
Backbone LR: 1e-4
Heads LR: 1e-3
Temperature: tau = 0.07 initially
Early Stopping: 5-7 epochs
```

### Experiment 5.1：Loss Comparison

统一：

```text
30 epochs
```

比较：

```text
CE
SupCon
CE + SupCon
```

不要直接比较：

```text
CE 训练 10 轮
SupCon 训练 50 轮
```

否则训练预算成为 confounding variable。

### Experiment 5.2：Lambda Ablation

测试：

```text
lambda = 0, 0.1, 0.25, 0.5, 1.0
```

每组：

```text
Max Epoch = 30
Early Stopping = 5
```

固定：

```text
Dataset
Split
Backbone
Augmentation
Batch
Optimizer
LR
Temperature
```

唯一改变：

```text
lambda
```

结果表：

| lambda | Best Epoch | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | | | | | | |
| 0.1 | | | | | | |
| 0.25 | | | | | | |
| 0.5 | | | | | | |
| 1.0 | | | | | | |

### Experiment 5.3：Projection Dimension

候选：

```text
64
128
256
512
```

结构：

```text
512 Backbone Feature
       ↓
Projection Head
       ↓
64 / 128 / 256 / 512
```

统一：

```text
Epoch = 30
lambda = Phase 5.2 selected best value
tau = fixed
```

评价：

```text
Accuracy
Recall@K
mAP
Storage
Search Time
```

## 15. Phase 6：Explainability

该阶段：

```text
Training Epoch = 0
```

直接使用已训练 checkpoint：

```text
CE ResNet
CE + SupCon ResNet
Swin
```

分析：

```text
Grad-CAM
Attention Visualization
T-SNE / UMAP
Error Analysis
```

T-SNE / UMAP 建议从 validation 或 test 中采样：

```text
10-20 classes
20-30 images per class
```

不要一开始把所有 5,794 个 test 点放进图里，否则图会很难读。

## 16. 最终推荐训练预算

| Phase | Experiment | Data | Epoch |
| --- | --- | ---: | ---: |
| P1 | Frozen ResNet | about 4,795 train | 10 |
| P1 | Partial FT | about 4,795 train | 20-25 |
| P1 | Full FT | about 4,795 train | 25-30 |
| P1 | Augmentation Ablation | about 4,795 train | 20/group |
| P2 | Feature Extraction | Val/Test | 0 |
| P3 | FAISS | Val/Test | 0 |
| P4 | Swin-Tiny | about 4,795 train | 25-30 |
| P5 | CE/SupCon/Joint | about 4,795 train | 30-40 |
| P5 | Lambda Ablation | about 4,795 train | 30/group |
| P5 | Embedding Dim Ablation | about 4,795 train | 30/group |
| P6 | Visualization | Val/Test | 0 |

这些数字不是理论规定，而是结合 CUB 数据规模、pretrained backbone、项目目标和计算预算制定的起始协议。正式结果使用 best validation checkpoint。

## 17. Random Seeds

如果算力允许，最终最重要的模型至少重复 3 个 seeds：

```text
seed = 42
seed = 123
seed = 2026
```

建议重复：

```text
ResNet-18 + CE
ResNet-18 + CE + SupCon
Swin-Tiny + Best Loss
```

报告：

```text
mean ± std
```

例如：

```text
Recall@10 = 84.7 ± 0.8
```

这比单独报告一个数更有科研说服力，因为它说明提升不是某个随机初始化带来的偶然结果。

## 18. 当前项目状态说明

Phase 1 Day 1 到 Day 5 的早期运行属于 bootstrap 实验，主要目标是跑通数据、训练、验证、checkpoint 和错误分析链路。

从正式对照实验开始，应遵守本协议：

```text
Official Train
        ↓
Stratified Train / Validation
        ↓
根据 Validation 选择模型和 checkpoint
        ↓
最终只在 Official Test 上评估一次
```

下一步工程任务是实现固定 train / validation / test split，并让训练脚本支持 `train`、`val`、`test` 三种评估语义。
