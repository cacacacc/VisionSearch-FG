# Weighted Local Retrieval Qualitative Human Notes

## Context

本记录对应当前最高 mAP / Recall@1 的检索设置：

```text
Swin BBox Crop + class-evidence weighted local pooling
target_class = predicted
tau = 1.0
```

对应报告：

```text
outputs/experiments/retrieval_qualitative_weighted_local/swin_bbox_evidence_weighted_tau1/retrieval_qualitative_report.html
```

本次人工分析覆盖 16 个案例，每组 4 个：`top1_correct`、`top1_wrong_top5_hit`、`top5_wrong_top10_hit` 和 `top10_failure`。因为该模型训练和评估时使用 BBox crop，本次人工标注主要参考 `contact_sheets_bbox` 中的裁剪后视图。

## Case Notes

### Top-1 正确

| Query | Class | 同类别结果 | 外观相似但类别不同 | 背景相似导致错误 | 姿态相似 | 局部特征混淆 |
| --- | --- | --- | --- | --- | --- | --- |
| #3 / id=41 | 001.Black_footed_Albatross | 同类样本集中在前 5，Top-1 正确 | 错误样本多为大型海鸟，长翼和褐灰色羽毛接近 | 多数样本仍有海面或水面背景 | 飞行和水面停栖姿态均参与相似度 | Sooty / Fulmar 与 query 在头颈和喙部细节不同，但仍被排入 Top-10 |
| #6 / id=80 | 002.Laysan_Albatross | 同类样本出现在 Top-1、Top-2、Top-3、Top-5 | 错误样本主要是其他长翼海鸟，黑白翅和白头区域接近 | 海面和蓝天背景仍有影响 | 多个错误样本为飞行姿态 | 头部白色区域和浅色长喙被部分利用，但不同海鸟的喙形差异没有完全区分 |
| #13 / id=142 | 003.Sooty_Albatross | 同类样本在 Top-1、Top-4、Top-6 | 错误样本仍集中在 albatross / fulmar 类海鸟 | 海面背景相似明显 | 飞行姿态主导多个错误邻居 | Sooty 的深色头颈和体色被利用，但与 Black-footed / Laysan 的头部和翼色差异仍被混淆 |
| #18 / id=181 | 004.Groove_billed_Ani | 多个同类样本进入前排和 Top-10 | 错误样本多为黑色或深色鸟体，长尾轮廓接近 | 枝叶背景参与相似度 | 多数样本为树枝站立姿态 | 喙部沟纹和头部轮廓仍不够稳定，Shiny Cowbird / Fish Crow 等深色鸟会靠前 |

### Top-1 错误，但 Top-5 有同类

| Query | Class | 同类别结果 | 外观相似但类别不同 | 背景相似导致错误 | 姿态相似 | 局部特征混淆 |
| --- | --- | --- | --- | --- | --- | --- |
| #0 / id=11 | 001.Black_footed_Albatross | 同类样本在第 3、7、9 位，Top-1 错误但 Top-5 命中 | 错误样本与 query 都是长翼海鸟，整体形态接近 | 海面/水面背景非常明显 | 多个错误样本为飞行姿态 | 喙和头颈颜色没有压过飞行剪影，Fulmar / Sooty / Laysan 被排到前面 |
| #7 / id=86 | 002.Laysan_Albatross | 同类样本在第 3 位 | 错误样本多为白腹、长翼、飞行海鸟 | 蓝天背景占主导 | 几乎全部为展开双翼飞行姿态 | 局部喙和头部区域在远景下很弱，Frigatebird / Jaeger / Tern 被误认为相似 |
| #12 / id=121 | 003.Sooty_Albatross | 同类样本在第 4 和第 6 位 | 错误样本多为大型海鸟或水鸟，体型和羽色接近 | 水面背景较强 | 飞行和水面停栖姿态都参与排序 | 头颈颜色、喙形和体色细节不足以把 Black-footed / Laysan / Gadwall 排到后面 |
| #20 / id=203 | 004.Groove_billed_Ani | 同类样本在第 2、4、6、8、9 位，类内召回较好 | 错误样本为深色鸟或长尾鸟，整体轮廓接近 | 枝叶和绿色背景参与 | 多数为树枝站立姿态 | Groove-billed Ani 的喙部沟纹、头部纹理和体型比例仍不足，Shiny Cowbird / Blue Grosbeak 会靠前 |

### Top-5 错误，但 Top-10 有同类

| Query | Class | 同类别结果 | 外观相似但类别不同 | 背景相似导致错误 | 姿态相似 | 局部特征混淆 |
| --- | --- | --- | --- | --- | --- | --- |
| #45 / id=406 | 008.Rhinoceros_Auklet | 同类样本只在第 7 位 | 错误样本多为灰白水鸟或黑白海鸟，短腿站立外观接近 | 岩石、水边或浅色背景有影响 | 多数为站立姿态 | 橙/黄喙被模型利用但过度泛化，喙形、头部白斑和体型关系区分不足 |
| #52 / id=471 | 009.Brewer_Blackbird | 同类样本在第 9、10 位 | 错误样本集中在 crow / cowbird / raven，黑色鸟体接近 | 背景不完全一致，主体颜色更主导 | 多数为站立姿态 | 黄色眼点、喙形、体型比例和羽毛光泽没有稳定区分 Brewer Blackbird 与其他黑色鸟 |
| #87 / id=812 | 015.Lazuli_Bunting | 同类样本在第 6、10 位 | 错误样本多为 Indigo Bunting / Blue Grosbeak，蓝色小鸟外观接近 | 绿色枝叶背景有一定影响 | 多数为站立姿态 | Lazuli 的翼部棕色斑块是关键线索，但前排排序仍主要被蓝色羽毛主导 |
| #97 / id=906 | 017.Cardinal | 同类样本在第 6、8 位 | 错误样本多为 Summer Tanager / Purple Finch，红色鸟体接近 | 枝条和浅色背景有影响 | 多数为站立姿态 | Cardinal 的冠羽、黑脸和粗短喙没有稳定压过整体红色，颜色 shortcut 仍明显 |

### Top-10 完全失败

| Query | Class | 同类别结果 | 外观相似但类别不同 | 背景相似导致错误 | 姿态相似 | 局部特征混淆 |
| --- | --- | --- | --- | --- | --- | --- |
| #2 / id=32 | 001.Black_footed_Albatross | Top-10 无同类 | Top-10 几乎都是长翼海鸟或大型海鸟，整体形态非常接近 | 海面/蓝天背景明显 | 飞行姿态强烈主导 | Black-footed 与 Sooty / Laysan / Fulmar 的头颈颜色、喙形和翼部细节未被稳定区分 |
| #15 / id=145 | 003.Sooty_Albatross | Top-10 无同类 | 错误样本包含 crow、gull、puffin、auklet 等深色或灰白水鸟 | 水面、岩石、岸边背景都有影响 | 多数为站立姿态 | 深色主体 shortcut 很强，Sooty 的大型海鸟体型和长喙特征没有被正确保留 |
| #43 / id=392 | 008.Rhinoceros_Auklet | Top-10 无同类 | 错误样本集中在 puffin、auklet、guillemot、gull 等黑白水鸟 | 岩石和水边环境相似 | 多数为站立或水边姿态 | 橙色/黄色喙和黑白头部被粗粒度匹配，但 Rhinoceros Auklet 的特殊喙形和头部结构没被捕捉 |
| #48 / id=426 | 009.Brewer_Blackbird | Top-10 无同类 | 错误样本几乎都是蓝黑色或黑色光泽鸟 | 背景差异较大，主体颜色更主导 | 多数为站立姿态 | 黄色眼睛和黑亮羽毛导致 Cape Glossy Starling / Boat-tailed Grackle 大量靠前，体型、尾形和喙形区分不足 |

## Summary

与 ResNet CE qualitative report 相比，`Swin BBox Evidence-weighted Local tau1.0` 的改善主要体现在：同类样本更容易进入前排，尤其在 Albatross 和 Groove-billed Ani 这些早期类别中，Top-10 内的同类数量明显增加。这个结果与定量指标一致：该方法把 BBox global 的 mAP 从 55.51% 提升到 59.62%，Recall@1 从 68.42% 提升到 71.25%。

但错误模式没有完全消失。跨案例看，模型仍然明显依赖整体颜色、体型、姿态和场景类型。海鸟类仍受长翼飞行姿态、海面/蓝天背景和大体型影响；黑色鸟类仍受黑色或蓝黑色羽毛、站立姿态和黄色眼点影响；红色鸟类仍受整体红色主导；蓝色小鸟仍受蓝色羽毛主导。

局部层面，最常见的失败仍然来自喙、头部、眼睛和翼部纹理没有被稳定用于排序。Evidence-weighted local pooling 能提升数值，说明 classifier evidence 确实比 token norm 更适合选择有判别性的局部信息；但它仍是 post-hoc pooling，没有显式约束模型必须学习 part-level discrimination。

## Implication

当前结果支持继续做 part-aware 方法，但方向应该更精确：

```text
1. 保留 evidence-weighted local pooling 作为当前最强 mAP / Recall@1 retrieval baseline。
2. 不再优先推进 naive token norm Top-K，因为它已经被 8.2 证明不够有效。
3. 下一步应分析 evidence-weighted token heatmap 是否更接近 CUB beak / eye / wing annotations。
4. 如果 alignment 仍然不强，再进入 part-guided crop upper bound 或 local-global training。
```

因此，8.3 的结论不是“局部问题已解决”，而是：类别证据加权能更好利用局部 token，但真正的 part-aware learning 仍有研究空间。
