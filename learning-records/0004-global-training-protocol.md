# 0004：全局训练协议决定实验可信度

## 背景

项目已经跑通 frozen baseline、错误样本分析和非冻结 backbone probe。接下来如果要比较 frozen、partial fine-tuning、full fine-tuning、Swin 和 SupCon，必须先固定全局训练协议。

## 学到什么

正式实验不能直接把 CUB 全部 11,788 张随机拆成 80/10/10。应该保留官方 test set，只把 official train 的 5,994 张按 stratified split 拆成 training 和 validation。

```text
Official Train -> Training / Validation
Official Test  -> Final Test Only
```

## 为什么重要

如果不断查看 test accuracy 并据此调学习率、epoch、augmentation 或模型结构，test set 就会变成 validation set，最终结果存在 test leakage。

## 后续行动

下一步应该实现固定 split 文件：

```text
data/processed/splits/cub_train_ids_seed42.txt
data/processed/splits/cub_val_ids_seed42.txt
data/processed/splits/cub_test_ids.txt
```

然后修改训练配置，使正式实验优先使用 training / validation，最终报告才使用 official test。
