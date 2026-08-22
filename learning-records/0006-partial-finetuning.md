# 0006：Partial Fine-tuning 是 frozen 与 full 之间的折中

## 背景

项目已经支持 frozen 和 full fine-tuning。为了更公平地比较迁移学习策略，需要加入 partial fine-tuning。

## 学到什么

Partial fine-tuning 不是“随便解冻一些参数”。本项目第一版明确为：

```text
layer4 + classifier 可训练
其他 ResNet-18 backbone 层冻结
```

## 为什么重要

Frozen 训练成本最低，但适配能力弱；full fine-tuning 适配能力强，但成本高，也更容易破坏预训练特征。partial fine-tuning 在二者之间折中，常作为细粒度分类的合理起点。

## 后续行动

运行：

```powershell
.\.venv\Scripts\python.exe scripts\train_baseline.py --config configs\baseline_resnet18_partial_protocol.yaml --device cpu
```

跑完后把 `history.json` 发给我，用同一张实验表比较 frozen、partial 和 full。
