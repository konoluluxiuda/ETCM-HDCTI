# 跨库实体映射恢复

## 1. 目的

本阶段为 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0 建立可追溯的 compound/protein 对齐工作清单。它只整理本地证据，不联网、不下载属性，也不根据匿名数字 ID 猜测实体身份。

生成命令：

```bash
python tools/prepare_entity_alignment_manifests.py \
  --alignment 'SymMap2.0=results/symmap_official_alignment'
```

输出目录：

```text
results/entity_alignment_recovery/
  all_entities.csv
  report.json
  report.md
  <dataset>/compound_alignment.csv
  <dataset>/protein_alignment.csv
```

## 2. 清单字段

每个实体保留以下审查信息：

```text
dataset
entity_type
local_entity_id
cp_positive_degree
priority_rank
source_entity_id
source_identifier_namespace
canonical_identifier
canonical_name
molecular_formula
organism
mapping_method
mapping_source_file
mapping_confidence
review_status
review_note
```

`priority_rank` 按 C-P 正边度数从高到低生成，只用于安排映射审查顺序，不参与训练或评价。TCMSP 第二列连续编号保存在 `matrix_entity_id`，第一列官网查询键保存在 `source_entity_id`；两者均不能直接冒充 PubChem 或 UniProt ID。

## 3. 当前结果

清单共覆盖 `32,795` 个实际出现在 C-P 关系中的实体。

| 数据集 | 实体 | 数量 | Source-ready | Canonical | 当前状态 |
|---|---|---:|---:|---:|---|
| TCM-Suite | Compound | 1,187 | 0% | 0% | 缺少映射文件 |
| TCM-Suite | Protein | 7,258 | 0% | 0% | 缺少映射文件 |
| TCMSP | Compound | 6,929 | 100% | 0% | 本地 ID 可查询官方 molecule 页面 |
| TCMSP | Protein | 1,748 | 100% | 0% | 本地 ID 可查询官方 target 页面 |
| SymMap2.0 | Compound | 1,618 | 100% | 63.78% | 已与官方 V2 `Mol_id` 精确对齐；1,032 个有 PubChem CID |
| SymMap2.0 | Protein | 4,027 | 100% | 25.23% | 已与官方 V2 `Gene_id` 精确对齐；3,944 个有 Ensembl ID |
| ETCM2.0-mention10 | Compound | 9,519 | 100% | 0% | 可按 TCMIP ID、名称和分子式补全 |
| ETCM2.0-mention10 | Protein | 509 | 100% | 100% | 已有 UniProt accession |

这里的 `Canonical` 对化合物要求外部标准标识，例如 PubChem CID；TCMIP ID 只计为可追溯来源标识。蛋白当前按格式有效的 UniProt accession 计入。

## 4. 恢复顺序

### 4.1 TCMSP 官方页面属性补全

`compound_id_all.csv` 和 `protein_id_all.csv` 中夹带的原始字段名分别为 `molecule_ID` 和 `target_ID`；第二列是严格连续的预处理矩阵编号。本地 C-P 关系使用第一列 ID，且既有 Top 候选核验已经证明它们可作为 TCMSP 官方页面的 `qn`/`qt` 查询键，返回 `MOL...`、`TAR...`、名称、InChIKey、PubChem CID 或 DrugBank target 信息。

因此 TCMSP 的 `8,677` 个 C-P 实体从“映射阻塞”调整为“待外部补全”。清单中的第二列写入 `matrix_entity_id`，不再误记为来源实体 ID。

验收条件：

* 对全部 C-P 实体执行可恢复、限速的官方页面查询，并记录失败与重试状态；
* Compound 至少保留 Molecule ID、名称、InChIKey、PubChem CID 和分子式；
* Target 至少保留 Target ID、名称、DrugBank ID，并进一步解析到物种明确的 UniProt；
* 记录来源 URL、抓取日期、原始响应哈希和字段冲突，不能静默覆盖。

### 4.2 SymMap2.0 官方 V2 对齐

已从 [SymMap 官方下载页](https://www.symmap.org/download/)取得 V2 SMIT/SMTT 文件，并运行：

```bash
python tools/audit_symmap_official_alignment.py \
  --smit <SymMap-v2-SMIT.xlsx> \
  --smtt <SymMap-v2-SMTT.xlsx>
```

审计结果：

* 本地 C-P 使用的 `1,618` 个 compound 与官方 `Mol_id` 为 `1,618/1,618` 精确匹配；
* 本地 C-P 使用的 `4,027` 个 protein 与官方 `Gene_id` 为 `4,027/4,027` 精确匹配；
* 本地全量唯一 compound/protein ID 也分别达到 `25,681/25,681` 和 `18,192/18,192`；
* 未使用关系度数、名称相似度或数字后缀猜测，精确 ID 冲突数为 0；
* 官方文件 SHA-256 分别为 `2cadefea5a598d6d7ae39360a8f5ac9bbbcc4f9e62d8f9546a4eab8679b31dbf` 和 `9a1c03b22c3ea09fc417206e43e44a378f72e975805d66fcfabd72e38e3895f8`。

当前 V2 导出与论文数据的全量行数口径并不完全相同，因此导入器保留文件哈希、官方行号与匹配方法，不把官网现行版本静默当作论文快照。对 C-P 实际使用实体，已有属性覆盖为：PubChem `63.78%`、分子式 `40.73%`、UniProt `25.23%`、Ensembl `97.94%`。因此 SymMap 已从“映射阻塞”转为“待属性补全”，但尚未达到真实 SMILES/sequence 门槛。

### 4.3 TCM-Suite 预匿名化映射

TCM-Suite 仍缺少预匿名化实体对照。它不再阻塞最低三库 Pilot，但若要做四库统一多模态实验，仍需获取原数据库同版本导出或联系 HDCTI 作者索取对照表，且不得按关系度数近似匹配。

### 4.4 ETCM2.0 属性补全

ETCM2.0 可以并行补全 PubChem SMILES 和 UniProt sequence，但它只用于验证补全流程和案例研究。在其他库映射恢复前，不据此提前实现四库共享多模态模型。

## 5. Go/No-Go

完成映射后，使用：

```bash
python tools/audit_multidataset_attributes.py \
  --entity-mapping 'SymMap2.0=results/symmap_official_alignment' \
  --alignment-root <alignment-root>
```

只有至少三个数据集同时满足以下条件才进入共享多模态 Pilot：

* SMILES 覆盖率至少 70%；
* 已映射 SMILES 的分子式确认率至少 95%；
* 蛋白序列覆盖率至少 95%。

当前 TCMSP、SymMap2.0 和 ETCM2.0 已达到来源标识门槛，下一步是生成三库标准化 SMILES/sequence 文件并重新审计。真实属性覆盖未达标前，论文仍使用 Strict-HDCTI + Hctx-P + CHCR 作为冻结主线。
