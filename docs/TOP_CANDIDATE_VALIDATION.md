# Top 未标注候选外部核验

## 目的与结论

本次核验用于判断高分未标注 C-P pair 中是否存在实质比例的可信阳性，从而决定是否值得进入 PU Learning pilot。核验对象不是重新训练后的模型，也不用于选择 checkpoint。

结论：**当前证据不支持把 PU Learning 作为下一步优先方向。** 严格口径下可信阳性为 `0/30`；即使将一条未达到预设效力门槛的直接酶实验宽松计为阳性，也只有 `1/30`。两种口径的 Wilson 95% 置信区间上限都低于预先定义的 `20%`“实质比例”门槛。

## 固定样本

评价对象为 TCMSP Strict fold 1 的 HerbOnly + Dot + early stopping checkpoint：

```text
saved_model/2026-07-14 18-43-19/hdcti_model.ckpt
```

抽样流程：

1. 对外层测试中每个含正例的 compound，在过滤训练已知正边后取模型分数最高的未标注 pair；
2. 得到 `4,281` 个候选，按数值 compound ID 排序；
3. 使用 `floor(i*N/30)` 的系统抽样固定抽取 30 个，不按证据结果替换候选；
4. TCMSP 页面仅用于数字 ID 到实体名称、InChIKey 和 DrugBank target 的映射，不作为关系证据。

可复核标识：

| 项目 | 值 |
|---|---|
| `top_candidates.tsv` SHA-256 | `2c85e6e19f386baf67873b585c962880ab38fb26803de57022d2fd2300986bae` |
| `validation_sample.tsv` SHA-256 | `0503ff5ce9a3be4c15ca058091e61843316286990c1eac69c1620a11a2a05800` |
| 候选总体 | 每个测试 compound 的最高分未标注 pair，共 4,281 对 |
| 固定样本量 | 30 |
| 核验日期 | 2026-07-14 |

## 证据标准

严格可信阳性必须是与目标蛋白相符的直接结合或功能实验，并至少满足一项：

```text
pChEMBL >= 5
IC50 / EC50 / Ki / Kd / potency <= 10 µM
inhibition / activity >= 50%
```

仅有分子对接、网络药理预测、靶点表达变化、通路调控、药材/提取物活性或相关蛋白活性，不计为直接阳性。`ChEMBL` 无记录也不计为阴性，只表示该数据库未覆盖。

“实质比例”在最终靶点映射修正并重跑前固定为：

```text
可信阳性点估计 >= 20%
```

数据库筛查使用 [ChEMBL Data Web Services](https://www.ebi.ac.uk/chembl/api/data/docs)，文献检索使用 Europe PMC 的标题/摘要字段，并同时查询靶点基因名和常用别名。

## 核验结果

| 证据类别 | 数量 | 样本编号 | 说明 |
|---|---:|---|---|
| 严格可信阳性 | 0 | - | 无 pair 达到预设直接实验阈值 |
| 直接但低于阈值 | 1 | 21 | cyanidin-3-glucoside–COX-2 |
| 直接阴性 | 1 | 1 | MTL/mannitol–COX-2 |
| 靶点映射不够特异 | 3 | 2、4、10 | NCOA2 只能匹配到 PPARγ/NCOA2 蛋白互作复合物 |
| 仅间接/预测证据 | 1 | 18 | danshensu–HSP90 命中网络药理或复方机制研究，无直接 assay |
| 未找到 pair 级直接证据 | 24 | 其余样本 | Europe PMC 精确检索无命中，或命中内容不能支持该 pair |

ChEMBL 自动筛查得到：

```text
direct_negative: 1
target_mapping_not_specific: 3
compound_not_in_chembl: 17
no_direct_chembl_record: 9
credible_positive: 0
```

Europe PMC 固定检索中，27/30 个 pair 的标题/摘要精确组合为 0 命中；有命中的 3 个 pair 为样本 1、18 和 21，经人工核验后只有样本 21 存在直接但效力不足的实验信号。

## 关键候选复核

### 样本 21：cyanidin-3-glucoside–COX-2

开放全文实验报告 cyanidin 3-O-glucoside 的 COX-2 `IC50 = 7.21 ± 0.28 µg/mL`；按其分子量换算约为 `16 µM`，高于预设的 `10 µM` 门槛，因此归为“直接但低于阈值”，不计入严格阳性。[原始实验论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6222845/)

另外，多篇研究只显示 C3G 降低 COX-2 表达或通过 MAPK、NF-κB、Fyn 等上游通路产生作用，这些不能单独证明 C3G 与 COX-2 的直接有效结合。[示例研究](https://pubmed.ncbi.nlm.nih.gov/21501596/)

### 样本 1：MTL/mannitol–COX-2

ChEMBL 中有两条同一 assay 的记录，均为人 COX-2，结论是 `10 µM` 时抑制率低于 `50%`，未测得剂量响应曲线。因此该 pair 归为直接阴性，而不是数据库空白。[ChEMBL 活动记录](https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id=CHEMBL689&target_chembl_id=CHEMBL230&limit=1000)

### 样本 18：danshensu–HSP90

Europe PMC 的两条命中来自复方/网络药理或多成分机制研究，没有 danshensu 对 HSP90 的直接结合、抑制或功能 assay，因此只记为间接证据，不计阳性。

## 统计判断

严格口径：

```text
0 / 30 = 0.00%
Wilson 95% CI: 0.00% - 11.35%
```

最宽松敏感性分析，将样本 21 也计为阳性：

```text
1 / 30 = 3.33%
Wilson 95% CI: 0.59% - 16.67%
```

即使使用宽松口径，置信区间上限仍未达到 `20%`。因此这批 Top 未标注候选中没有发现“实质比例”的可信漏标阳性。

## PU 决策

当前停止 PU 分支，理由为：

1. 固定完整候选评价已经有 `Recall@50 = 0.905710` 和 `Hits@50 = 0.955618`；
2. 静态 mixed-negative pilot 已低于 Random 基线；
3. Top 未标注外部核验未发现实质比例的直接阳性；
4. 现阶段引入 nnPU、可靠负例选择或 PU 风险会增加方法复杂度，但缺少与之对应的错误机制证据。

这不证明 TCMSP 的全部未标注关系都是真负例。样本只覆盖一个数据集、一个 fold、一个 checkpoint，且自然产物在 ChEMBL 中覆盖不足。若未来在 ETCM2.0 案例研究或新增实验数据库中获得更多直接关系证据，可以重新开启 PU，但必须重新固定样本和判定门槛。

下一步优先进行主模型在外部数据集上的泛化验证和多 seed 稳定性评估，而不是继续设计 PU loss。

## 复现命令

```bash
./tools/prepare_top_candidate_validation.py \
  --input results/checkpoint_ranking/herb_only_fold1_2026-07-14/top_candidates.tsv \
  --sample-size 30 \
  --output-dir results/candidate_validation/tcmsp_herb_only_fold1_n30

./tools/audit_candidate_chembl.py \
  --sample results/candidate_validation/tcmsp_herb_only_fold1_n30/validation_sample.tsv \
  --output-dir results/candidate_validation/tcmsp_herb_only_fold1_n30

./tools/search_candidate_literature.py \
  --sample results/candidate_validation/tcmsp_herb_only_fold1_n30/validation_sample.tsv \
  --output-dir results/candidate_validation/tcmsp_herb_only_fold1_n30
```

生成结果位于被 `.gitignore` 排除的 `results/candidate_validation/`；本文记录固定哈希、统计结果和关键证据，核验脚本保存在仓库目录中，待最近修改统一提交时纳入版本控制。
