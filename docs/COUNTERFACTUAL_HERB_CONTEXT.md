# 反事实药材上下文可行性审计

## 1. 研究问题

冻结的静态 Hctx-P 模块已经证明药材上下文能够改善 C-P 排序，但仅凭性能增益无法判断模型使用的是候选成分的真实药材归属，还是 H-C degree、热门药材或上下文向量分布等捷径。

本审计回答一个更窄的问题：

> 在模型参数完全冻结时，将候选成分的真实药材上下文替换为度数匹配且药材集合不相交的上下文，是否会稳定降低正样本得分和验证集 AUPR？

审计只用于决定是否值得实现 Counterfactual Herb Context Regularization（CHCR）训练 Pilot，不作为最终模型结果。

## 2. 事实与反事实

对验证 pair $(c,p)$，事实打分使用成分 $c$ 的真实静态药材上下文 $h_c$：

$$
s_f(c,p)=s_{base}(c,p)+(h_c\odot w_{HP})^Tz_p
$$

从另一个成分 $c^-$ 取得反事实上下文 $h_{c^-}$：

$$
s_{cf}(c,p)=s_{base}(c,p)+(h_{c^-}\odot w_{HP})^Tz_p
$$

候选 $c^-$ 必须满足：

1. $deg_{HC}(c^-)=deg_{HC}(c)$；
2. $H(c^-)\cap H(c)=\varnothing$；
3. $c^-\ne c$；
4. 仅使用固定 H-C 侧信息，不使用 outer-test C-P 标签。

同一 draw 内，同一 compound 的所有 protein pair 共用同一个反事实 donor，避免为特定靶点选择有利或不利的上下文。默认固定 `20` 个 draw，seed 为 `42026`。

## 3. 评价范围

审计默认且当前只允许使用：

```text
Strict outer fold 内部的 validation pairs
```

不执行优化器、不更新参数、不根据 outer-test 选择第二创新。输出包括：

* `report.json`：协议、checkpoint、H-C 哈希、判定与完整统计；
* `report.md`：可读摘要；
* `draw_metrics.tsv`：每次反事实 draw 的 AUPR；
* `pair_margins.tsv`：逐 pair 的事实/反事实 logit 和 donor；
* `subgroup_metrics.tsv`：H-C degree 与训练 C-P degree 分层结果。

## 4. 预注册判定标准

在查看实际结果前固定以下继续门槛：

| 条件 | 门槛 |
|---|---:|
| 可获得严格反事实的 validation record 覆盖率 | $\ge 90\%$ |
| 正样本 pair 的平均反事实 margin 胜率 | $\ge 60\%$ |
| 正样本平均 margin | $>0$ |
| Factual AUPR - mean counterfactual AUPR | $\ge 0.001$ |
| 可分析 degree strata 中平均 margin 为正的比例 | $\ge 75\%$ |
| 可分析 subgroup 最少正样本 pair | $30$ |

全部通过时输出：

```text
supports_CHCR_training_pilot
```

覆盖不足时输出 `inconclusive_counterfactual_coverage`；其他情况输出 `does_not_support_CHCR_training_pilot`。不在查看结果后修改上述门槛。

## 5. 运行命令

当前 ETCM2.0_core_mention10 fold 1 的冻结静态 HerbOnly checkpoint：

```bash
./tools/audit_counterfactual_herb_context.py \
  --config configs/HDCTI_etcm_mention10_herb_only_early_stop.conf \
  --checkpoint "saved_model/2026-07-15 12-56-37/hdcti_model.ckpt" \
  --fold 1 \
  --draws 20 \
  --counterfactual-seed 42026 \
  --output-dir results/counterfactual_context/etcm_mention10_fold1
```

仅检查协议、checkpoint 和反事实覆盖率：

```bash
./tools/audit_counterfactual_herb_context.py \
  --config configs/HDCTI_etcm_mention10_herb_only_early_stop.conf \
  --checkpoint "saved_model/2026-07-15 12-56-37/hdcti_model.ckpt" \
  --fold 1 \
  --dry-run
```

## 6. 解释边界

该设计是冻结模型上的机制扰动，不是严格的生物学因果实验。反事实 donor 只是满足结构约束的合成替换，不能称为确认错误的药材上下文。多个 pair 可能共享 compound，因此 pair 胜率之外必须同时查看 compound 聚合胜率、degree 分层一致性和 AUPR 变化。

若审计通过，下一步只实现一组预注册的 CHCR 单折训练 Pilot；若不通过，则停止反事实训练分支，不通过增加 draw、放宽匹配条件或改变门槛追逐正结果。

## 7. ETCM2.0_core_mention10 Fold 1 结果

审计于 2026-07-15 使用冻结的静态 HerbOnly checkpoint 完成。运行过程中没有训练步骤，也没有读取 outer-test 指标。严格反事实覆盖 validation records `14,070/14,148`（`99.45%`）；20 次 draw 共检查 `281,400` 个替换，自替换、H-C degree 不相等和药材集合重叠均为 `0`。

| 指标 | 结果 |
|---|---:|
| Factual validation AUPR | 0.975812 |
| Counterfactual validation AUPR | 0.926713 (±0.001993) |
| Factual - counterfactual AUPR | 0.049099 |
| Counterfactual AUPR 范围 | 0.922327–0.930327 |
| 正样本 pair 胜率 | 0.827674 |
| 正样本 pair 胜率 95% Wilson CI | 0.818674–0.836318 |
| 正样本 compound 胜率 | 0.827509 |
| 正样本平均 logit margin | 2.831416 |
| 标准化平均 margin | 0.935127 |
| 正方向 degree strata | 9/9 (100%) |

负样本平均 margin 为 `-0.061740`，接近零且方向与正样本不同。这说明替换上下文主要破坏正样本的匹配信号，而不是简单地把所有 pair 的 logit 同方向平移。H-C degree 的四个可分析组和训练 C-P degree 的五个可分析组均保持正 margin；训练 C-P degree=0 组的正样本胜率仍为 `0.696429`，没有出现只在高连接节点上成立的完全反转。

预注册的五项条件全部通过，程序判定为：

```text
supports_CHCR_training_pilot
```

该结果支持“冻结 Hctx-P 确实依赖正确药材上下文”，并支持进入 CHCR 单折训练 Pilot；它本身不证明加入反事实损失一定改善预测性能。完整机器可读结果位于 `results/counterfactual_context/etcm_mention10_fold1/`，该目录受 `.gitignore` 管理，关键结果以本文档为长期记录。

## 8. CHCR 单折训练 Pilot

审计通过后冻结唯一一组训练设置：

```ini
counterfactual.context=True
counterfactual.match=exact_hc_degree_disjoint
counterfactual.weight=0.05
counterfactual.margin=0.2
counterfactual.draws=20
counterfactual.seed=42026
```

对训练正样本定义：

$$
L_{CHCR}=0.05\sum_{(c,p)\in B^+}
\max\left(0,0.2-s_{ctx}(c,p,h_c)+s_{ctx}(c,p,h_{c^-})\right)
$$

其中 $s_{ctx}$ 只包含 Hctx-P 上下文项，基础 Dot logit 不参与事实—反事实差值。每个 epoch 固定使用一个 donor draw，20 个 epoch 后循环；负样本和没有严格 donor 的 compound 不参与该损失。总损失为：

$$
L=L_{BCE}+L_{reg}+L_{CHCR}
$$

配置文件为 `configs/HDCTI_etcm_mention10_chcr_pilot.conf`。它复用静态 HerbOnly 的 fold 1、inner-validation、seed、最大 50 epoch 和早停设置，并设置 `evaluation.outer.test=False`。

静态 HerbOnly fold 1 的预注册参考 validation AUPR 为 `0.975782`。判定规则：

| CHCR validation AUPR | 决策 |
|---|---|
| $\ge 0.976782$ | 支持进入完整五折 |
| $0.974782$–$0.976781$ | 视为非劣但没有第二创新增益，停止扩展 |
| $<0.974782$ | Pilot 未通过，停止 CHCR |

不根据 Pilot 结果搜索 weight、margin、draw 数或 donor 规则。运行命令：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_chcr_pilot.conf
```

### 8.1 Pilot 结果

单折 Pilot 于 2026-07-15 完成：

| 项目 | 结果 |
|---|---:|
| 最佳 validation AUPR | 0.979722 |
| 最佳 epoch | 38 |
| 实际停止 epoch | 48 |
| 相对静态 Hctx-P 参考 | +0.003940 |
| 相对完整五折门槛 | +0.002940 |
| epoch 48 最后 batch CHCR loss | 4.623899 |
| epoch 48 最后 batch active positives | 663 |
| epoch 48 最后 batch mean margin | 2.257669 |
| 最终 Hctx-P weight mean abs | 1.717248 |
| 运行时间 | 434.475734 s |

checkpoint：

```text
saved_model/2026-07-15 20-14-07/hdcti_model.ckpt
```

Pilot AUPR 超过预注册的 `0.976782` 门槛，因此结果判定为 **通过**，允许进入完整五折。该结果来自用于方法筛选的 fold 1 inner-validation，不能作为最终泛化性能；不运行该 checkpoint 的 outer-test，也不根据结果调整 CHCR 超参数。

## 9. 完整五折协议

完整五折配置为 `configs/HDCTI_etcm_mention10_chcr_early_stop.conf`。除恢复全部五折和 outer-test 外，它与 Pilot 配置完全一致：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_chcr_early_stop.conf
```

最终比较对象是同一 Strict split、seed 和早停协议下已经完成的静态 HerbOnly 五折。需要报告五项 outer-test 指标的均值、fold 标准差、逐折方向及配对差值；不能将 Pilot validation AUPR 与五折 outer-test AUPR 混合比较。

### 9.1 完整五折结果

完整五折于 2026-07-15 完成。CHCR 与静态 Hctx-P 使用相同的 Strict split、训练 seed、内层验证早停、Dot decoder 和 outer-test 候选集合；唯一新增变量是冻结设置的反事实上下文排序约束。

| 指标 | 静态 Hctx-P | Hctx-P + CHCR | 配对均值增益 | 提升折数 |
|---|---:|---:|---:|---:|
| AUC | 0.977835(±0.000107) | 0.981949(±0.000453) | +0.004114 | 5/5 |
| AUPR | 0.973955(±0.000823) | 0.980136(±0.000775) | +0.006181 | 5/5 |
| Recall | 0.938856(±0.002077) | 0.940881(±0.003324) | +0.002024 | 4/5 |
| Precision | 0.925843(±0.004198) | 0.933951(±0.002054) | +0.008107 | 5/5 |
| F1-score | 0.932296(±0.001360) | 0.937397(±0.000858) | +0.005100 | 5/5 |

逐折配对差值如下：

| Fold | ΔAUC | ΔAUPR | ΔRecall | ΔPrecision | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.003958 | +0.006666 | -0.000170 | +0.005782 | +0.002813 |
| 2 | +0.004052 | +0.006220 | +0.004523 | +0.007271 | +0.005914 |
| 3 | +0.003935 | +0.006164 | +0.001300 | +0.009675 | +0.005559 |
| 4 | +0.004855 | +0.006848 | +0.003788 | +0.011690 | +0.007798 |
| 5 | +0.003769 | +0.005004 | +0.000679 | +0.006118 | +0.003417 |

Fold 5 的最佳 validation AUPR 为 `0.977961`（epoch 46），最后一个 batch 的 CHCR loss、有效正样本数和平均事实—反事实 margin 分别为 `4.189561/697/2.325990`，说明训练结束时约束仍在实际参与优化。Fold 5 checkpoint 为：

```text
saved_model/2026-07-15 20-55-35/hdcti_model.ckpt
```

五折总运行时间为 `2161.121490 s`，相对静态 Hctx-P 的 `1570.049160 s` 增加约 `37.65%`。后续跨数据集验证应同时报告排序增益和训练成本。

### 9.2 当前结论与边界

CHCR 的 AUPR 在 5/5 折同向提高，且同时改善 AUC、Precision 和 F1；Recall 仅在 fold 1 出现 `-0.000170` 的轻微回落。因此，完整五折结果支持将 CHCR 保留为静态 Hctx-P 之后的第二项模型创新，而不是只作为机制分析工具。

该结论仍限于 `ETCM2.0_core_mention10`、一套固定五折和一个训练 seed。五折标准差反映数据划分差异，不等同于独立随机初始化的稳定性；后续需要冻结当前实现，在其他数据集和多个训练 seed 上检验泛化。CHCR 结果支持“保持正确药材上下文排序有助于预测”，但不能单独证明因果效应或所有反事实 donor 都具有生物学真实性。

## 10. TCMSP 跨数据集验证

冻结 ETCM 阶段的 donor 规则、weight、margin、draws 和 seed 后，使用 `configs/HDCTI_tcmsp_chcr_early_stop.conf` 在 TCMSP 上进行完整五折验证。对照为相同 Strict split、内层验证早停、Dot decoder 和静态 Hctx-P 的 `configs/HDCTI_herb_only_early_stop.conf`。

| 指标 | 静态 Hctx-P | Hctx-P + CHCR | 配对均值增益 | 提升折数 |
|---|---:|---:|---:|---:|
| AUC | 0.987095(±0.001265) | 0.987519(±0.001232) | +0.000424 | 5/5 |
| AUPR | 0.984085(±0.001782) | 0.985090(±0.001718) | +0.001005 | 5/5 |
| Recall | 0.958290(±0.005606) | 0.961463(±0.003194) | +0.003173 | 4/5 |
| Precision | 0.951731(±0.001752) | 0.950467(±0.001419) | -0.001264 | 2/5 |
| F1-score | 0.954991(±0.002525) | 0.955932(±0.002142) | +0.000942 | 3/5 |

逐折配对差值：

| Fold | ΔAUC | ΔAUPR | ΔRecall | ΔPrecision | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.000035 | +0.000988 | +0.001693 | -0.003278 | -0.000813 |
| 2 | +0.000061 | +0.000337 | +0.002495 | +0.000208 | +0.001345 |
| 3 | +0.000508 | +0.000852 | +0.006595 | -0.001461 | +0.002534 |
| 4 | +0.000797 | +0.001661 | +0.008289 | -0.003553 | +0.002325 |
| 5 | +0.000718 | +0.001188 | -0.003209 | +0.001763 | -0.000683 |

Fold 5 最佳 validation AUPR 为 `0.983424`（epoch 22），checkpoint 为：

```text
saved_model/2026-07-15 22-14-28/hdcti_model.ckpt
```

五折运行时间为 `1602.395324 s`。静态 Hctx-P 对照用时 `1760.008411 s`，但两者早停 epoch 不同，因此不能据此声称 CHCR 单 epoch 更快。

TCMSP 的 AUC/AUPR 增益小于 ETCM mention10，但两项均在 5/5 折同向，支持 CHCR 排名作用具有初步跨数据集泛化。Precision 和 F1 的方向不稳定，说明固定 `0.5` 阈值下的分类增益较弱；当前不为此修改阈值或 CHCR 参数。该跨数据集实验仍是单训练 seed，训练初始化稳定性由下一节独立验证。

## 11. ETCM 多 seed 稳定性协议

多 seed 实验固定 ETCM mention10 的数据划分和验证集，只改变模型初始化与训练随机性：

```ini
split.seed=2026
split.dir=./dataset/ETCM2.0_core_mention10/splits/strict_seed_2026_k5
validation.seed=102026
counterfactual.seed=42026
random.seed=2026, 2027, 2028
```

`split.seed` 与 `random.seed` 已在代码中解耦。前者控制负样本和 outer folds，后者控制各 fold 的 Python、NumPy 与 TensorFlow 随机状态；因此改变训练 seed 不会创建或覆盖 split manifest。seed 2026 的静态 Hctx-P 与 CHCR 五折已经完成，新增运行配置为：

```text
configs/HDCTI_etcm_mention10_herb_only_seed2027.conf
configs/HDCTI_etcm_mention10_chcr_seed2027.conf
configs/HDCTI_etcm_mention10_herb_only_seed2028.conf
configs/HDCTI_etcm_mention10_chcr_seed2028.conf
```

可以逐个运行，也可以顺序执行：

```bash
./run_etcm_chcr_multiseed.sh
```

统计时先计算每个 seed 的五折均值，再计算 3 个 seed 均值的总体均值和 sample standard deviation。15 个 fold 不能视为 15 次独立重复。主指标为每个 seed 内配对的 AUPR 差值，AUC 为共同主排序指标，Recall/Precision/F1 为固定阈值补充指标。

预注册解释规则：

| 结果 | 解释 |
|---|---|
| 3/3 seed 的 AUPR 均值增益均为正 | 强稳定性支持 |
| 2/3 为正且三 seed 总体均值增益为正 | 有限稳定性支持，报告异质性 |
| 少于 2/3 为正或总体均值不为正 | 不支持稳定增益，CHCR 降为数据集依赖增强 |

无论结果如何，不调整 donor 规则、weight、margin、draws、早停参数或 decoder。

### 11.1 多 seed 结果

seed 2027/2028 的四组新增五折于 2026-07-15 至 2026-07-16 完成。每个 seed 内的配对均值差如下：

| Seed | ΔAUC | ΔAUPR | ΔRecall | ΔPrecision | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 2026 | +0.004114 | +0.006181 | +0.002024 | +0.008107 | +0.005100 |
| 2027 | +0.003956 | +0.005684 | +0.001990 | +0.008304 | +0.005189 |
| 2028 | +0.003637 | +0.005367 | +0.001346 | +0.007383 | +0.004401 |

下表先对每个 seed 的五折求均值，再对三个 seed 均值计算总体 mean 和 sample standard deviation：

| 指标 | 静态 Hctx-P | Hctx-P + CHCR | 配对增益 | 正增益 seed |
|---|---:|---:|---:|---:|
| AUC | 0.978091(±0.000385) | 0.981993(±0.000160) | +0.003903(±0.000243) | 3/3 |
| AUPR | 0.974329(±0.000517) | 0.980073(±0.000250) | +0.005744(±0.000410) | 3/3 |
| Recall | 0.940194(±0.001688) | 0.941981(±0.001314) | +0.001787(±0.000382) | 3/3 |
| Precision | 0.926226(±0.000726) | 0.934157(±0.000258) | +0.007931(±0.000485) | 3/3 |
| F1-score | 0.933150(±0.001191) | 0.938047(±0.000780) | +0.004897(±0.000431) | 3/3 |

AUC、AUPR、Precision 和 F1 在三个 seed 合计 15 个 fold 中均为 `15/15` 同向提升，Recall 为 `12/15`。因此结果满足预注册的“3/3 seed AUPR 均值增益均为正”，判定为 **强稳定性支持**。CHCR 在 ETCM mention10 上的增益不是 seed 2026 的偶然初始化结果。

seed 2028 CHCR 的五折结果为 AUC `0.982171(±0.001265)`、AUPR `0.980286(±0.001611)`、Recall `0.943436(±0.002012)`、Precision `0.934446(±0.004772)` 和 F1 `0.938912(±0.002201)`，运行时间 `2090.748247 s`。Fold 5 checkpoint 为：

```text
saved_model/2026-07-16 00-38-39/hdcti_model.ckpt
```

该结论解决了训练初始化稳定性问题，但仍不等于跨数据库全域稳定性证明。TCMSP 已提供 AUC/AUPR 的初步跨数据集支持；后续不再增加训练 seed 或调整 CHCR 超参数，转向案例解释与论文结果整理。
