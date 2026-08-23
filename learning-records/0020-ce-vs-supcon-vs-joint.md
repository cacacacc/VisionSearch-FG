# 0020：SupCon-only 的 Accuracy 需要 linear probe

## 背景

Experiment 5.1 要比较 CE only、SupCon only 和 CE + SupCon，并同时报告 Accuracy 与 Retrieval metrics。这里有一个容易混淆的问题：SupCon loss 只优化 embedding space，不直接训练分类头。

## 学到什么

CE only 和 CE + SupCon 都包含 CrossEntropy，因此 classifier head 的 validation accuracy 可以直接作为分类指标。SupCon only 不包含 CrossEntropy，训练过程中的 classifier head 没有被有效优化，所以它的 accuracy 不能作为正式分类结果。正确做法是训练完 SupCon encoder 后冻结 encoder，重置并训练一个 linear classifier，再把这个 linear probe 的 validation accuracy 作为 SupCon-only 的分类指标。

## 为什么重要

如果把 SupCon-only 随机分类头的 accuracy 写进表格，会错误低估 SupCon 表征的分类可线性分性。Linear probe 可以回答一个更合理的问题：这个 embedding 是否包含足够的类别信息，使一个简单线性分类器能够利用它。

## 后续行动

Phase 5.1 已新增 `scripts/train_contrastive.py` 和 `scripts/train_linear_probe.py`。正式实验顺序是先跑 CE only、SupCon only、CE + SupCon；SupCon only 跑完后追加 linear probe；最后对三组 best checkpoint 执行同一个 validation retrieval protocol。
