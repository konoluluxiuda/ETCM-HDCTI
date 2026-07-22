# CHCR Donor 对照纯推理审计

## 1. 目的

CHCR 已在统一普通随机边四库实验中获得 AUPR 非负增益，但仅比较事实药材上下文与“同 H-C degree 且药材集合不相交”的反事实，仍可能受到两类审稿质疑：

1. 事实优势是否只是随机 donor 的 H-C degree 不同；
2. 在控制 degree 后，模型是否真正对药材集合重叠敏感。

本审计冻结未经过 CHCR 训练的静态 Hctx-P checkpoint，只使用 Strict inner-validation，不读取 outer-test，不执行优化器更新。

## 2. 三类 Donor

对每个候选 compound 固定生成 20 个 donor，同一 draw 内该 compound 的全部 protein pairs 共用相同 donor。

| 策略 | 排除自身 | H-C degree 相同 | 要求药材不相交 | 用途 |
|---|---|---|---|---|
| `random` | 是 | 否 | 否 | 宽松随机替换对照 |
| `exact_degree` | 是 | 是 | 否 | 排除 H-C degree 差异 |
| `exact_degree_disjoint` | 是 | 是 | 是 | 检验药材上下文特异性 |

主检验在所有可获得 `exact_degree` donor 的 validation records 上执行，以直接排除 H-C degree 差异。随后在三种策略共同可获得 donor 的 records 上比较 `random`、`exact_degree` 和 `exact_degree_disjoint`，作为更严格的不相交 donor 加强对照。两部分分别报告覆盖率，避免把 donor 可获得性差异混入结论。

## 3. 输出指标

每种策略报告：

```text
donor overlap fraction
donor degree-matched fraction
counterfactual AUPR
factual - counterfactual AUPR
positive mean logit margin
positive pair / compound win rate
pair win rate Wilson 95% CI
H-C degree 与训练 C-P degree 分层方向
```

同时保存逐 draw、逐 subgroup 和逐 pair 记录，便于检查结果是否由少量热门 compound 驱动。

## 4. 冻结判定

“控制 degree 后仍有上下文特异性证据”的主检验要求：

```text
exact_degree record 覆盖率 >= 90%
exact_degree 正样本 pair 胜率 >= 60%
exact_degree 正样本平均 margin > 0
exact_degree factual-CF AUPR >= 0.001
可分析 degree strata 中正 margin 比例 >= 75%
```

如果三策略共同 record 覆盖率也达到 90%，则以相同的胜率、margin、AUPR 降幅和分层一致性门槛检查 `exact_degree_disjoint`，作为不相交 donor 的确认性证据。若共同覆盖率不足，只能保留主检验结论，不能宣称已经完成严格不相交确认。

若共同子集上的 `exact_degree` donor 中实际发生药材重叠的比例至少为 10%，且严格 donor 的正样本 margin 和 AUPR 降幅还同时更大，则进一步记为存在 overlap-sensitive 证据。该比较只使用严格大于零的描述性门槛，不据此搜索 donor、draw 或数据集特定规则。若实际重叠低于 10%，只能判断是否存在超出 degree 的上下文信号，不能判断 overlap 的额外作用。

可能输出：

```text
supports_context_specificity_beyond_degree_and_overlap
supports_context_specificity_beyond_degree_disjoint_confirmed_overlap_inconclusive
supports_context_specificity_beyond_degree_disjoint_coverage_inconclusive
supports_context_specificity_beyond_degree_disjoint_not_confirmed
does_not_support_context_specificity_beyond_degree
```

## 5. ETCM Fold 1 命令

先检查协议、checkpoint 和共同 donor 覆盖率：

```bash
./tools/audit_chcr_donor_controls.py \
  --config configs/HDCTI_etcm_mention10_pair_stratified_herb_only_no_dense_full.conf \
  --checkpoint "saved_model/2026-07-17 17-32-22/hdcti_model.ckpt" \
  --fold 1 \
  --draws 20 \
  --counterfactual-seed 42026 \
  --output-dir results/chcr_donor_controls/etcm_mention10_fold1 \
  --dry-run
```

覆盖通过后删除 `--dry-run` 执行纯推理审计。输出：

```text
report.md
report.json
strategy_metrics.tsv
draw_metrics.tsv
subgroup_metrics.tsv
pair_margins.tsv
```

## 6. 解释边界

该审计回答的是冻结 Hctx-P 是否利用了超出 H-C degree 的药材集合信息。合成 donor 不代表生物学上确认错误的药材归属，因此不能表述为因果效应。若 ETCM fold 1 未通过冻结门槛，则停止该审计，不通过修改 seed、draw 或 overlap 规则追逐正结果；若通过，再使用完全相同设置扩展到其余数据库和 folds。

ETCM fold 1 的预检查中，`exact_degree` donor 实际药材重叠比例约为 2.25%。因此该数据上的 overlap 对照预期会被标记为证据不足；这不影响 degree 控制审计，但禁止据此声称已经排除药材集合重叠偏倚。

## 7. ETCM mention10 Fold 1 冻结结果

使用未经过 CHCR 训练的静态 Hctx-P checkpoint：

```text
saved_model/2026-07-17 17-32-22/hdcti_model.ckpt
```

在 20 个固定 draws、seed `42026` 下，主检验 `exact_degree` donor 覆盖率为 `14108/14148 = 99.72%`；三类 donor 的共同 validation record 覆盖率为 `14070/14148 = 99.45%`。

主检验结果为：

| Coverage | Factual-CF AUPR | 正样本平均 logit margin | Pair 胜率 |
|---:|---:|---:|---:|
| 0.997173 | 0.052019 | 2.977646 | 0.833499 |

共同子集上的加强对照结果为：

| Donor | Factual-CF AUPR | 正样本平均 logit margin | Pair 胜率 | Degree match | 实际 overlap |
|---|---:|---:|---:|---:|---:|
| Random | 0.053292 | 3.057069 | 0.834920 | 0.382059 | 0.009206 |
| Degree-matched, overlap allowed | 0.052035 | 2.981920 | 0.833925 | 1.000000 | 0.017919 |
| Degree-matched, disjoint | 0.052162 | 3.010630 | 0.832078 | 1.000000 | 0.000000 |

冻结判定为：

```text
supports_context_specificity_beyond_degree_disjoint_confirmed_overlap_inconclusive
```

该结果支持 Hctx-P 利用了超出 H-C degree 的药材上下文信息，并且在 ETCM fold 1 上通过了严格不相交 donor 的确认。由于允许重叠的同度数 donor 实际仅有 `1.79%` 发生重叠，当前结果不能进一步证明差异由药材集合重叠本身驱动。

完整机器可读结果位于：

```text
results/chcr_donor_controls/etcm_mention10_fold1
```

## 8. 四库冻结批处理

四库 20 个静态 Hctx-P checkpoint 已固化在：

```text
configs/chcr_donor_control_checkpoints.json
```

该 manifest 同时冻结配置 SHA-256、fold、checkpoint、20 个 draws、seed `42026` 和数据集级判定门槛。批处理工具只执行纯推理，并在已有 `report.json` 与冻结输入完全一致时复用结果：

```bash
python tools/run_chcr_donor_control_batch.py --dry-run
python tools/run_chcr_donor_control_batch.py --protocol-dry-run --fold 1
python tools/run_chcr_donor_control_batch.py
```

数据集级通过条件预先固定为：至少 `4/5` fold 通过主 `exact_degree` 检验、平均主检验 AUPR drop 至少 `0.001`、平均正样本 pair 胜率至少 `0.60`。四库均通过才得到总判定 `PASS`。

四库 20 折均已完成：

| 数据集 | 支持折 | 主检验 AUPR drop | 正样本 margin | Pair 胜率 | 数据集判定 |
|---|---:|---:|---:|---:|---|
| TCM-Suite | 5/5 | 0.026199(+-0.016788) | 1.273365(+-0.258768) | 0.831567(+-0.021814) | PASS |
| TCMSP | 5/5 | 0.005818(+-0.001648) | 0.790534(+-0.285209) | 0.747427(+-0.008655) | PASS |
| SymMap2.0 | 1/5 | 0.002776(+-0.001427) | 0.191904(+-0.109646) | 0.824798(+-0.068125) | NO-GO |
| ETCM2.0 mention10 | 5/5 | 0.039171(+-0.006829) | 2.212152(+-0.492210) | 0.821744(+-0.008798) | PASS |

四库总判定为 **NO-GO**。该结论不是因为总体 margin 或 pair 胜率不足，而是 SymMap2.0 的 H-C degree 分层一致性只有一折达到 `>=75%`：其余 folds 的主检验分层正 margin 比例为 `0.50` 或 `0.667`。因此不能声称 Hctx-P/CHCR 在四库中都稳定学习了超出 degree 的上下文语义。

证据边界如下：

1. TCM-Suite、TCMSP 和 ETCM2.0 mention10 支持 degree 控制后的上下文特异性。
2. TCMSP 和 ETCM2.0 的严格不相交 donor 覆盖充分，但允许重叠 donor 的实际 overlap 低于 10%，不能声称额外的 overlap-sensitive 机制。
3. TCM-Suite 的不相交共同覆盖率低于 90%，只支持主 degree-control 结论。
4. SymMap2.0 只能报告总体正向趋势，不能作为稳定机制证据；不得通过降低分层门槛或更换 seed 将其改写为通过。

完整汇总位于：

```text
results/chcr_donor_controls/four_dataset_static_hctxp/summary.md
results/chcr_donor_controls/four_dataset_static_hctxp/results.tsv
```

下一步不再修补 donor 规则或继续增加 seed，而是用现有逐 degree-stratum 输出完成 SymMap 失败模式表，并据此限制论文主张：CHCR 是在具备稳定药材上下文支持的数据环境中有效的训练约束，而不是四库无条件成立的普适机制。

## 9. Degree-Stratum 失败模式

使用四库 20 个冻结 `report.json` 的主 `exact_degree` 结果进行描述性聚合，不重新训练、不访问 outer-test，也不新增数据集特定阈值。分层方向一致性继续使用原冻结条件：至少 75% 的可分析 folds 具有正平均 margin。Pair 胜率 0.60 仅作为强度参考，不追加为新的分层通过条件。

| 数据集 | H-C degree 方向不一致 | 训练 C-P degree 方向不一致 | 解释 |
|---|---|---|---|
| TCM-Suite | 无 | `1-2`、`3-5` | 药材上下文方向稳定，但低 C-P 训练支持下收益不稳定 |
| TCMSP | 无 | 无 | 所有可分析区间方向稳定 |
| SymMap2.0 | `1` | `0`、`1-2` | 同时存在低药材上下文与低 C-P 支持失配 |
| ETCM2.0 mention10 | 无 | 无 | 所有可分析区间方向稳定 |

SymMap 的 `H-C degree=2-3` 和训练 `C-P degree=3-5` 虽达到 4/5 folds 正 margin，但 pair 加权胜率分别只有 `0.4879` 和 `0.5685`，属于方向多数为正但强度较弱的过渡区间。稳定区间集中在 `H-C degree>=4` 和训练 `C-P degree>=6`。

这一结果将原来的数据库级失败进一步定位为**训练支持度调节的上下文可靠性问题**：Hctx-P/CHCR 不是在所有实体上等强生效。论文可以将其作为适用边界和 SDIS/support-aware 设计动机，但不能在未验证新路由机制前声称已经自动解决该问题。

可复现工具与输出：

```bash
python tools/summarize_chcr_degree_strata.py
```

```text
results/chcr_donor_controls/four_dataset_static_hctxp/degree_strata_analysis/by_fold.tsv
results/chcr_donor_controls/four_dataset_static_hctxp/degree_strata_analysis/summary.tsv
results/chcr_donor_controls/four_dataset_static_hctxp/degree_strata_analysis/report.md
```
