# Compound-Centric 排名目标

## 动机

当前 HDCTI 使用逐 pair BCE，在随机正负样本上优化分类概率；实际应用却是给定 compound 后对所有候选 protein 排序。静态 Hctx-P 与 CHCR 改善了表示和反事实上下文鲁棒性，但没有直接约束同一 compound 的正靶点得分高于负候选。

因此，第三项候选创新转向训练目标，而不继续叠加同源图传播模块。候选方法是保留原 BCE 与 CHCR，同时增加 compound-centric pairwise 或 listwise ranking loss。

## 冻结 Headroom 审计

先使用冻结 CHCR checkpoint 检查 Strict inner-validation 中同一 compound 的正负候选排序：

```text
checkpoint: saved_model/2026-07-15 20-14-07/hdcti_model.ckpt
fold: 1
outer-test: 不计算
optimizer steps: 0
```

只有同一 compound 同时具有正例和负例时才进入分组审计。统计：

* 所有正负 pair 的 margin；
* macro/micro pairwise violation；
* BPR softplus loss；
* 首个正例排名、MRR 和 Top-1 miss；
* 按 Strict model-train C-P 正边 degree 分层；
* compound 级 bootstrap 95% 区间。

进入 validation-only 训练 Pilot 必须同时满足：

```text
有效 compound >= 1000
有效记录覆盖率 >= 40%
macro violation >= 5%，且 bootstrap 下限 >= 4%
Top-1 miss >= 8%，且 bootstrap 下限 >= 6%
至少两个样本数 >=100 的训练 degree 层违例率 >=5%
```

该门槛要求排序问题具有足够规模且跨 degree 存在。若未通过，不实现 ranking loss，也不搜索 margin、温度或 loss weight。

## 命令

```bash
python tools/audit_compound_ranking_headroom.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/compound_ranking_headroom/etcm_mention10_fold1
```

正结果只允许进入一个固定设置的单折 Pilot。最终方法仍需使用每个 compound 的固定完整 protein 候选集合评价 Precision@K、Recall@K、Hits@K 和 MRR。

## Headroom 审计结果

审计于 2026-07-16 完成，checkpoint 和编码器保持冻结，outer-test 未计算。

| 指标 | 结果 |
|---|---:|
| 有效 compound | 2,368 / 7,151 |
| 有效记录覆盖率 | 49.89% |
| Macro pairwise violation | 2.47%（95% CI 1.87%-3.04%） |
| Micro pairwise violation | 1.93% |
| Top-1 miss | 2.70%（95% CI 2.07%-3.38%） |
| Macro MRR | 0.986275 |

总体违例率和 Top-1 失误率均远低于预注册门槛，且没有样本数至少 100、违例率至少 5% 的 degree 层，因此判定为 `stop_compound_centric_ranking_loss_route`。不实现全局 pairwise/listwise loss，也不搜索 margin 或 loss weight。

错误具有明显的低度集中现象：

| Model-train C-P degree | Compounds | Violation | Top-1 miss |
|---|---:|---:|---:|
| 0 | 60 | 13.89% | 16.67% |
| 1 | 87 | 8.62% | 10.34% |
| 2-3 | 309 | 3.02% | 3.24% |
| 4-7 | 893 | 2.15% | 2.24% |
| 8-15 | 832 | 1.39% | 1.32% |

这表明随机边 inner-validation 的总体排序已经接近饱和，剩余问题主要属于未见/低度 compound，而非所有 compound 的统一排序目标。后续若继续模型创新，应先建立 compound cold-start 或 low-degree 专门划分，不应根据这 147 个低度节点事后开启 degree-specific ranking loss。

## 2026-08-06：Cold-start 全候选分支重新开放

上述 No-Go 只适用于随机折的 1:1 采样候选。新的 compound cold-start
inner-validation 全候选审计严格不读取 outer-test，并在四库第 1 折均发现模型
明显低于折内非神经启发式：MRR 差值为
`-0.248574/-0.094332/-0.158062/-0.075583`，Recall@20 差值为
`-0.259891/-0.066558/-0.193093/-0.109526`。因此旧结论不再阻止
cold-start 专用 ranking Pilot。

本次只开放一个冻结编码器、固定负例构成和固定 pairwise logistic 目标的 Pilot；
它不复活随机折全局 ranking loss，也不允许根据 outer-test 搜索 margin、温度或
loss weight。协议与 Gate 见
[Inner-Validation 全候选排名审计](INNER_FULL_CANDIDATE_RANKING_AUDIT.md)。
