# 0015：Retrieval 需要定性证据

## 背景

Phase 2 和 Phase 3 已经建立了 CE retrieval baseline、metric ablation、normalization ablation、dimension ablation 和 FAISS search backend 对照。这些实验给出了数值结果，但还不能解释模型为什么成功或失败。

## 学到什么

Retrieval qualitative analysis 应该展示 Query、Top-1、Top-5 和 Top-10，并区分同类别结果、外观相似但类别不同、背景相似、姿态相似和局部特征相似。定性分析不是装饰，它是判断下一步 representation 改进方向的证据。

## 为什么重要

如果错误主要来自背景相似，那么改进方向应偏向 foreground、crop、attention 或更强 augmentation。如果错误主要来自细粒度局部特征混淆，那么更高分辨率、part-aware representation 或 SupCon 更有价值。如果同类样本常在 Top-5/Top-10 但 Top-1 错，说明 embedding 有语义结构，但类内紧致性和类间间隔还不够。

## 后续行动

运行 `scripts/generate_retrieval_qualitative_report.py` 生成 HTML 报告，人工查看每组案例并把关键失败模式写回正式实验文档。
