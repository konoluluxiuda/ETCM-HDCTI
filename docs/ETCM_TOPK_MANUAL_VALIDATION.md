# ETCM2.0 Top-K 独立证据核验

## 1. 当前状态

15 条待核验 E 级候选已在任何 BindingDB、ChEMBL 或 PubMed 检索开始前
冻结。每个 ETCM 案例保留 3 条候选，后续检索结果不能用于替换候选、调整
模型、改变 checkpoint 或修改排名。

```text
freeze_status=FROZEN_BEFORE_SEARCH
verification_status=IN_PROGRESS
candidate_count=15
query_count=45
reviewed_candidate_count=3
reviewed_query_count=9
selection_seed=2026
candidate_manifest_sha256=b6c68cb0d9e9cc55cd37d773ec099e0b8fc26c73c658609a79774cb78dd5fd01
```

## 2. Material Passport

| 字段 | 值 |
|---|---|
| Schema | ARS-9 |
| Artifact type | frozen evidence-verification worklist |
| Freeze status | `FROZEN_BEFORE_SEARCH` |
| Verification status | `IN_PROGRESS` |
| 数据范围 | ETCM2.0 mention10 fold 1 冻结 Top-K |
| 是否用于训练 | 否 |
| 是否用于模型/案例选择 | 否 |
| 是否改变候选排名 | 否 |

## 3. 选择协议

输入为：

```text
results/etcm_topk_cases/fold1/context/context_annotated_topk.tsv
```

候选必须同时满足：

1. 当前证据等级为 E；
2. Gene Symbol、Target Name 和 UniProt 完整；
3. 来自五个预先冻结案例之一；
4. 不使用 C-H-D-P 路径数、后续数据库命中或文献结果参与排序。

每个成分固定选择 3 条。成分内排序依次为：

1. 出现在三个冻结模型 Top-20 中的模型数降序；
2. reciprocal-rank 总和降序；
3. 最佳排名升序；
4. 平均排名升序；
5. seed 2026 的 SHA-256 tie-break。

这种等额分配避免 Quercetin 等热门成分占满人工核验清单。

## 4. 冻结候选

| 序 | 成分 | Gene | UniProt | 模型数 | 最佳排名 |
|---:|---|---|---|---:|---:|
| 1 | DEXPROPRANOLOL | CA12 | O43570 | 3 | 1 |
| 2 | DEXPROPRANOLOL | SIGMAR1 | Q99720 | 3 | 2 |
| 3 | DEXPROPRANOLOL | MCL1 | Q07820 | 3 | 2 |
| 4 | gallocatechin gallate | XBP1 | P17861 | 3 | 6 |
| 5 | gallocatechin gallate | CA2 | P00918 | 2 | 1 |
| 6 | gallocatechin gallate | CA1 | P00915 | 2 | 2 |
| 7 | (+)-Gallocatechin | MMP9 | P14780 | 3 | 2 |
| 8 | (+)-Gallocatechin | KDM1A | O60341 | 3 | 1 |
| 9 | (+)-Gallocatechin | CA4 | P22748 | 3 | 2 |
| 10 | Sulfuretin | PSMB5 | P28074 | 3 | 1 |
| 11 | Sulfuretin | FLT3 | P36888 | 3 | 3 |
| 12 | Sulfuretin | IGF1R | P08069 | 3 | 4 |
| 13 | Quercetin | NR0B1 | P51843 | 3 | 1 |
| 14 | Quercetin | PLAU | P00749 | 3 | 1 |
| 15 | Quercetin | OPRD1 | P41143 | 3 | 3 |

完整记录还包括 TCMIP ID、CAS、PubChem CID、Target Name、三模型排名、
页面路径计数和原始 Target 页面位置。

## 5. 证据等级

| 等级 | 判定 |
|---|---|
| B1 | 独立来源提供直接定量结合/活性证据，如 Kd、Ki、IC50、EC50 |
| B2 | 独立实验明确支持直接 C-P 作用，但定量信息不完整 |
| D | 只有通路、表达、疾病共现、分子对接或间接机制支持 |
| E | 未找到可核验的直接或间接支持 |
| Conflict | 可核验来源报告无活性或与候选关系冲突 |

数据库记录和文献必须保存 URL、记录号、DOI/PMID、物种、实验类型、活性值、
单位、检索日期和人工备注。分子对接不能升级为直接结合证据；未找到证据也
不能把候选写成真实负例。

## 6. 首批检索试运行

为避免将外消旋 propranolol 或相反对映体误认为 dexpropranolol，先完成
PubChem CID 21138 与 ChEMBL `CHEMBL275742` 的立体化学身份核对。二者
InChIKey 均为 `AQHHHDLHHXJYJD-CQSZACIVSA-N`。

| 序 | 候选 | 当前等级 | 核验结论 |
|---:|---|---|---|
| 1 | DEXPROPRANOLOL–CA12 | E | BindingDB、ChEMBL 和 PubMed 当前检索未找到精确 pair 支持 |
| 2 | DEXPROPRANOLOL–SIGMAR1 | B1 | ChEMBL human Sigma1 radioligand binding：Ki 1670 nM、IC50 3974 nM |
| 3 | DEXPROPRANOLOL–MCL1 | E | PubMed 命中均为外消旋 propranolol 的间接研究，未找到精确 pair 支持 |

SIGMAR1 记录的 ChEMBL molecule、target 和 UniProt 分别为
`CHEMBL275742`、`CHEMBL287` 和 `Q99720`，与冻结候选一致。CA12 和 MCL1
保留 E 级仅表示本轮规定来源中“尚未核验到支持”，不能解释为已经证明无作用。

版本化进度账本：

```text
configs/etcm_topk_evidence_progress.json
```

## 7. 文件与复现

跟踪配置：

```text
configs/etcm_topk_manual_validation.json
```

本地核验工作目录：

```text
results/etcm_topk_cases/manual_validation/
├── candidate_manifest.json
├── candidates.tsv
├── search_queries.tsv
├── evidence_review.tsv
└── summary.md
```

重新生成前必须明确删除或使用 `--overwrite`，避免无意覆盖冻结清单：

```bash
python tools/prepare_etcm_topk_manual_validation.py
```

## 8. 下一步

按照 `validation_order=4..15` 继续执行 BindingDB、ChEMBL 和 PubMed
检索并填写 `evidence_review.tsv`。全部 45 个数据库查询完成后，再按
B1/B2/D/E/Conflict 汇总 pair 级结论；核验期间不改候选清单。
