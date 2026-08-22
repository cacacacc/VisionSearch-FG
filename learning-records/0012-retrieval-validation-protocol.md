# 0012：Retrieval 开发阶段只用 validation

## 背景

Phase 2 的 retrieval 实验会反复比较 metric、normalization、embedding dimension、SupCon、Swin 和其他 representation。为了避免 test leakage，开发阶段不能反复查看 official test 结果。

## 学到什么

Retrieval validation protocol 是：validation 图片同时作为 query 和 gallery，但每个 query 必须从自己的 gallery 中排除自身。当前项目保存的 validation split 为 `data/processed/splits/cub_val_ids_seed42.txt`，实际包含 1,200 张图片；最终 official test 为 5,794 张图片。

## 为什么重要

如果不排除 query 自己，query embedding 与自身 embedding 的 cosine similarity 等于 1，Top-1 会先命中自己，Recall@1 会被虚高。如果开发阶段反复使用 test，test 会被间接用于模型选择，最终结果会出现 test leakage。

## 后续行动

Phase 2 到 Phase 5 的 retrieval 开发实验都先使用 validation split。只有当模型、checkpoint、metric、normalization、embedding dimension 和检索协议全部固定后，才在 official test 上运行完全相同 protocol。
