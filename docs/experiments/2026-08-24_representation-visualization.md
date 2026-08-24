# 实验 6.3：T-SNE / UMAP Representation Visualization

## 实验动机

Phase 5 已经通过 Recall@K 和 mAP 比较了 CE 与 CE + SupCon 的 retrieval representation。Experiment 6.3 进一步使用 t-SNE / UMAP 对 embedding space 做二维可视化，帮助直观观察类别聚类结构。

注意：

```text
t-SNE / UMAP 是定性证据。
主要结论仍然依赖 Recall@K / mAP 等定量指标。
```

## 研究问题

```text
CE embedding 与 CE + SupCon embedding 是否呈现不同的类别聚类结构？
```

比较：

```text
CE Embedding
vs
CE + SupCon Embedding
```

## 实验预期

CE：

```text
类别存在一定分离，但同类样本可能较分散。
```

SupCon / CE + SupCon：

```text
同类别样本应形成更紧密 cluster。
类别之间的局部混叠应减少。
```

## 实验设置

本实验不训练新模型，只复用已经保存的 embedding。

```text
Training Epoch = 0
```

默认使用 validation split 上保存的 embedding：

| Variant | Embedding | 说明 |
| --- | --- | --- |
| CE | `outputs/embeddings/phase5_lambda_ablation/20260823_093502_phase5_resnet18_ce_supcon_lambda0/cosine/embeddings.npy` | `lambda=0`，等价于 CE-only embedding |
| CE + SupCon | `outputs/embeddings/phase5_temperature_ablation/20260823_222957_phase5_resnet18_ce_supcon_temp0_2/cosine/embeddings.npy` | Phase 5.5 检索最优，`tau=0.2`，projection embedding |

为了让两个图可比较，采样必须一致：

```text
同一批 image_ids
同一批 classes
同一 split
同一 random seed
```

当前 validation split 每类样本数量较少，因此默认设置为：

```text
20 classes
每类最多 6 images
```

如果后续使用 official test 或 all split，可以改为：

```text
10-20 classes
20-30 images per class
```

## 训练指令

本实验不需要训练。

```powershell
# 无需训练，直接使用已保存 embeddings.npy。
```

## 可视化指令

### A. t-SNE

```powershell
cd D:\code\VisionSearch-FG

.\.venv\Scripts\python.exe scripts\visualize_embedding_space.py `
  --variant CE `
    outputs\embeddings\phase5_lambda_ablation\20260823_093502_phase5_resnet18_ce_supcon_lambda0\cosine\embeddings.npy `
    outputs\embeddings\phase5_lambda_ablation\20260823_093502_phase5_resnet18_ce_supcon_lambda0\cosine\records.csv `
  --variant CE_SupCon_tau0_2 `
    outputs\embeddings\phase5_temperature_ablation\20260823_222957_phase5_resnet18_ce_supcon_temp0_2\cosine\embeddings.npy `
    outputs\embeddings\phase5_temperature_ablation\20260823_222957_phase5_resnet18_ce_supcon_temp0_2\cosine\records.csv `
  --num-classes 20 `
  --samples-per-class 6 `
  --method tsne `
  --perplexity 30 `
  --seed 42 `
  --output-dir outputs\explainability\representation_visualization_phase6\ce_vs_supcon_tsne
```

输出：

```text
outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/
  summary.json
  selected_records.csv
  ce_tsne.png
  ce_tsne.csv
  ce_supcon_tau0_2_tsne.png
  ce_supcon_tau0_2_tsne.csv
```

### B. UMAP，可选

UMAP 需要额外安装 `umap-learn`。当前 `requirements.txt` 未固定该依赖，因此默认先不跑。

如果后续安装了 `umap-learn`，可以运行：

```powershell
.\.venv\Scripts\python.exe scripts\visualize_embedding_space.py `
  --variant CE `
    outputs\embeddings\phase5_lambda_ablation\20260823_093502_phase5_resnet18_ce_supcon_lambda0\cosine\embeddings.npy `
    outputs\embeddings\phase5_lambda_ablation\20260823_093502_phase5_resnet18_ce_supcon_lambda0\cosine\records.csv `
  --variant CE_SupCon_tau0_2 `
    outputs\embeddings\phase5_temperature_ablation\20260823_222957_phase5_resnet18_ce_supcon_temp0_2\cosine\embeddings.npy `
    outputs\embeddings\phase5_temperature_ablation\20260823_222957_phase5_resnet18_ce_supcon_temp0_2\cosine\records.csv `
  --num-classes 20 `
  --samples-per-class 6 `
  --method umap `
  --seed 42 `
  --output-dir outputs\explainability\representation_visualization_phase6\ce_vs_supcon_umap
```

## 最终结果表

| Variant | Visualization | Samples | Classes | Qualitative Cluster Pattern | Quantitative Anchor |
| --- | --- | ---: | ---: | --- | --- |
| CE | t-SNE | 120 | 20 | 多数类别已有局部聚集，但仍存在类内分散和局部混叠 | Recall@K / mAP 见 Phase 5 |
| CE + SupCon tau=0.2 | t-SNE | 120 | 20 | 部分类别 cluster 更紧凑，但仍有离群点和相似类别混叠 | Recall@K / mAP 见 Phase 5.5 |

已生成文件：

| File | Path |
| --- | --- |
| Summary | `outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/summary.json` |
| Selected Records | `outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/selected_records.csv` |
| CE t-SNE Figure | `outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/ce_tsne.png` |
| CE t-SNE Coordinates | `outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/ce_tsne.csv` |
| CE + SupCon t-SNE Figure | `outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/ce_supcon_tau0_2_tsne.png` |
| CE + SupCon t-SNE Coordinates | `outputs/explainability/representation_visualization_phase6/ce_vs_supcon_tsne/ce_supcon_tau0_2_tsne.csv` |

运行确认：

```text
method = t-SNE
num_selected_samples = 120
num_classes = 20
samples_per_class = 6
seed = 42
```

## 结果分析

CE t-SNE 图中，多数类别已经形成一定程度的局部聚集，说明 CE embedding 本身具备可用的语义结构。这与 Phase 2 / Phase 5 中 CE baseline 已经具备较强 Recall@K 的结果一致。

但是 CE 图中仍然可以看到类内分散和局部混叠现象。一些类别点虽然大致相邻，但 cluster 边界并不总是清晰，说明 CE 主要优化分类边界，并不显式要求同类样本在 embedding space 中紧密聚集。

CE + SupCon tau=0.2 图中，部分类别呈现更紧凑的 cluster，这符合 SupCon 的预期：同类样本被拉近，异类样本被推远。该现象也与 Phase 5.5 中 `tau=0.2` 的 retrieval 指标最优相互呼应。

不过，CE + SupCon 图并不是所有类别都完美分离，仍然存在离群点和相似类别局部混叠。因此不能只凭 t-SNE 图宣称 SupCon 全面改善 representation。更准确的说法是：

```text
t-SNE 定性支持 CE + SupCon 可能改善局部聚类结构，
但最终证据仍然来自 Recall@K / mAP。
```

## 分析框架

观察 CE 图：

- 同类点是否能大致聚在一起
- 不同类别之间是否存在明显混叠
- 是否有离群点远离同类 cluster

观察 CE + SupCon 图：

- 同类点是否更紧凑
- 类间边界是否更清晰
- 容易混淆的细粒度类别是否仍然混在一起

对比时要谨慎：

```text
t-SNE 会改变全局距离关系。
不同图之间不能用二维距离绝对值做强结论。
```

更可靠的判断方式是：

```text
t-SNE / UMAP 用来提供视觉证据。
Recall@K / mAP 用来支持定量结论。
```

## 科研意义

如果 CE + SupCon 的图中同类 cluster 更紧密，同时 Recall@K / mAP 也更高，则可以说明 SupCon 确实改善了 embedding geometry。

如果图上看起来更紧凑，但 Recall@K / mAP 没有提升，则说明二维可视化可能产生误导，不能作为主要证据。

如果 Recall@K / mAP 提升明显，但 t-SNE 图差异不明显，则说明高维空间中的检索结构不一定能被二维投影完整表达。

本实验最终服务于一个核心问题：

```text
CE + SupCon 是否真的学到了更适合 retrieval 的 representation？
```
