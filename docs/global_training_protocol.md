# 全局训练协议

## 协议目标

本协议定义 VisionSearch-FG 从 Phase 1 到 Phase 6 的统一数据划分、训练预算、验证流程和最终测试规则。

项目的科研价值不仅来自模型精度，也来自不同实验之间是否可比较。所有正式实验必须尽量控制变量：固定数据划分、固定评价协议、清楚记录训练预算，并且只在最终阶段使用 official test set。

最重要的原则：

```text
Test set 不用于调参。
所有模型选择、超参数选择和 ablation 决策都先基于 validation set。
最终协议固定后，才在 official test set 上做一次性评估。
```

## 1. 数据集协议

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

原因：不额外使用 bbox、part、attribute，可以避免给某些 baseline 带来额外 annotation 优势，从而保持不同方法之间的公平性。

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

固定随机种子：

```text
random_seed = 42
```

保存划分结果：

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
不断查看 Test Accuracy
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
Retrieval
Grad-CAM / 错误分析
```

如果 ResNet 使用 Split A，Swin 使用 Split B，那么两者最终精度差异无法归因于模型本身。

## 5. 图像分辨率

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
- 224 是迁移学习的合理起点。

后续可以独立做 resolution ablation：

```text
224 x 224
vs
384 / 448
```

该实验研究 fine-grained recognition 是否受益于更高分辨率，但不应在 baseline 阶段引入。

## 6. 数据增强

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

默认禁止：

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

Validation 和 test 不使用随机 augmentation，保证同一模型每次得到一致输入。

## 7. Batch Size

ResNet-18 第一版：

```text
batch_size = 32
```

显存或内存不足时：

```text
batch_size = 16
```

Swin-Tiny：

```text
batch_size = 16
```

显存或内存不足时：

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

正式结果使用：

```text
best validation checkpoint
```

而不是默认使用最后一个 epoch。

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

| 策略 | 最大 epoch | 最优 epoch | Val Acc | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 10 | | | |
| Partial FT | 25 | | | |
| Full FT | 30 | | | |

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

这会让 Top-1 永远先命中自己，得到假的高分。

最终模型、checkpoint、metric、normalization、embedding dimension 和检索协议全部确定后，才在 official test 上运行完全相同 protocol：

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
Index Memory
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

Search Recall 用 exact Flat top-K 作为 reference，衡量 ANN 是否找回同一批 nearest neighbors。Retrieval Recall 衡量 ANN 的近似误差是否真正影响类别检索任务。

该实验建立典型 ANN trade-off：

```text
Accuracy ↔ Speed
```

### Experiment 3.3：定性检索分析

不仅看数字，还要选择若干 query，展示：

```text
Query
Top-1
Top-5
Top-10
```

分析维度：

- 同类别结果
- 外观相似但类别不同
- 背景相似导致错误
- 姿态相似
- 局部特征相似

这是后续改进 representation 的重要证据。

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
实验文档包含训练指令和评估指令
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

否则训练预算会成为 confounding variable。

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

这张表是整个项目非常重要的 ablation study。

### Experiment 5.3：Projection Head Ablation

比较：

```text
Backbone -> SupCon
vs
Backbone -> Projection Head -> SupCon
```

研究问题：

```text
Projection Head 是否真正改善 Representation Learning？
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
lambda
```

唯一改变：

```text
SupCon loss 的输入：
h = backbone feature
vs
z = projection head output
```

正式 retrieval evaluation 一律使用 backbone feature `h`，而不是 projection feature `z`。这样才能判断 projection head 是否帮助 backbone 学到更好的通用 representation。

结果表：

| Projection Head | SupCon 输入 | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 无 | h | | | | | |
| 有 | z | | | | | |

该实验直接对应 SimCLR / SupCon 类论文中的设计思想。

### Experiment 5.4：Embedding Dimension Ablation

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

本实验中的 Embedding Dimension 指 retrieval evaluation 使用的 projection feature `z` 维度，而不是 backbone feature `h`。因此评估命令必须显式使用：

```text
--feature projection
```

统一：

```text
Epoch = 30
lambda = Phase 5.2 selected best value
tau = fixed
Projection Head = MLP
SupCon feature = z
```

评价：

```text
Accuracy
Recall@1 / Recall@5 / Recall@10
mAP
Storage
Search Time
```

结果表：

| Projection Dim | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Memory | Query Time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | | | | | | | |
| 128 | | | | | | | |
| 256 | | | | | | | |
| 512 | | | | | | | |

### Experiment 5.5：Temperature Ablation

候选：

```text
tau = 0.05 / 0.07 / 0.1 / 0.2
```

研究问题：

```text
SupCon temperature 如何影响 embedding distribution？
```

固定：

```text
lambda = 0.1
Projection Head = MLP
Projection Dim = Phase 5.4 selected main value
SupCon feature = z
```

评价：

```text
Accuracy
Recall@1 / Recall@5 / Recall@10
mAP
Train SupCon Loss
```

该实验优先级低于 lambda ablation、projection head ablation 和 embedding dimension ablation。只有在算力允许时完成。

结果表：

| Tau | Accuracy | Recall@1 | Recall@5 | Recall@10 | mAP | Train SupCon Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | | | | | | |
| 0.07 | | | | | | |
| 0.1 | | | | | | |
| 0.2 | | | | | | |

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
错误分析
```

### Experiment 6.1：Grad-CAM Analysis

研究问题：

```text
ResNet 在做细粒度预测时关注的是鸟类判别区域，还是背景 shortcut？
```

比较：

```text
正确预测
vs
错误预测
```

观察区域：

```text
Head
Beak
Wing
Plumage
Background
```

协议：

```text
Training Epoch = 0
Split = validation
Target layer = ResNet-18 layer4[-1]
Target class = predicted class
```

优先分析模型：

```text
CE-only ResNet
CE + SupCon tau=0.1
CE + SupCon tau=0.2
```

输出：

```text
Grad-CAM overlay images
HTML report
CSV manual annotation table
```

### Experiment 6.2：Attention Visualization

研究问题：

```text
CNN 与 Transformer 是否使用不同视觉区域完成细粒度识别？
```

Swin-Tiny 分析区域：

```text
局部细节
鸟主体
背景
```

协议：

```text
Training Epoch = 0
Model = Swin-Tiny best validation checkpoint
Split = validation
Target class = predicted class
```

当前 `torchvision.models.swin_t` 不直接返回 attention weights，因此第一版使用 gradient-weighted final-stage Swin feature map 作为 attention-style visualization。后续如需严格 attention rollout，可替换为返回 attention weights 的 Swin 实现。

输出：

```text
Swin attention-style overlay images
HTML report
CSV manual annotation table
```

最终与 Experiment 6.1 对照：

```text
ResNet Grad-CAM
vs
Swin attention-style visualization
```

### Experiment 6.3：T-SNE / UMAP Representation Visualization

研究问题：

```text
CE embedding 与 CE + SupCon embedding 是否呈现不同的类别聚类结构？
```

比较：

```text
CE Embedding
vs
CE + SupCon Embedding
```

预期：

```text
CE：
类别存在一定分离，但同类样本可能较分散。

CE + SupCon：
同类别样本应形成更紧密 cluster。
```

协议：

```text
Training Epoch = 0
Split = validation
Same sampled image_ids for all variants
Default = 20 classes x up to 6 images per class
```

注意：

```text
T-SNE / UMAP 是定性证据。
主要结论仍然依赖 Recall@K / mAP 等定量指标。
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
| P5 | Projection Head Ablation | about 4,795 train | 30/group |
| P5 | Embedding Dim Ablation | about 4,795 train | 30/group |
| P6 | Visualization | Val/Test | 0 |
| P7 | BBox Crop / Fusion / PCA | Val/Test or about 4,795 train | 0 or 25-30 |
| P8 | Part Alignment Evaluation | Val/Test | 0 |
| P8 | Local Token Pooling Retrieval | Val/Test | 0 |
| P8 | Part-aware Training | about 4,795 train | 25-40 |

这些数字不是理论规定，而是结合 CUB 数据规模、pretrained backbone、项目目标和计算预算制定的起始协议。

正式结果最终使用：

```text
best validation checkpoint
```

而不是机械地取最后一轮。

## 17. Phase 7-8：Foreground-aware 与 Part-aware 扩展

Phase 7 和 Phase 8 是 Phase 6 可解释性分析之后的结构化优化方向。

Phase 7 研究 foreground-aware input：

```text
BBox crop
Original + BBox feature fusion
Fusion PCA compression
```

其中 BBox crop 使用 CUB 官方 bounding box，属于 annotation-assisted setting，必须单独报告。

Phase 8 研究 part-aware / local feature learning：

```text
Part annotation alignment evaluation
Local token pooling
Local-global feature fusion
Part-supervised auxiliary study
```

其中 CUB part locations 优先用于 evaluation / supervision study，不应默认成为最终 test pipeline 的必要输入。若最终方法依赖人工 part coordinates，必须明确标注为 oracle 或 annotation-assisted setting。

建议顺序：

```text
1. 先量化 Grad-CAM / Swin attention 与 beak、eye、wing 等 part 的对齐程度。
2. 再做 0 epoch 的 local token pooling retrieval，验证局部 token 是否能改善检索。
3. 最后再训练 part-aware local-global 模型。
```

## 18. Random Seeds

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

这比单独报告一个数字更有科研说服力，因为它说明提升不是某个随机初始化带来的偶然结果。

## 19. 当前项目状态说明

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

当前最需要严格遵守的一条是：

```text
Test 5,794 张图片不用于调参。
所有实验决策首先根据约 1,199 张 validation 图片做出。
```

这样最后的 Test Accuracy、Recall@K 和 mAP 才真正具有可信度。
