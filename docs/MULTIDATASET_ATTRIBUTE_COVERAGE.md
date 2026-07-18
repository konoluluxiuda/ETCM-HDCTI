# 多数据集实体属性覆盖审计

## 1. 审计目的

本审计用于判断分子结构与蛋白序列多模态分支能否作为 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0 的共享主创新，而不是只在 ETCM2.0 上成立的附加实验。

审计只读取当前仓库中的 C-P 关系和实体映射，不访问网络、不下载属性、不训练模型。匿名数字 ID 只计为矩阵实体映射，不计为可检索的生物学标识；只有标准化 enrichment 文件中的真实 SMILES 和蛋白序列才计入模态覆盖。

运行命令：

```bash
python tools/audit_multidataset_attributes.py
```

本地机器可读结果位于：

```text
results/multidataset_attribute_coverage/report.json
results/multidataset_attribute_coverage/report.md
```

## 2. 当前结果

| 数据集 | C-P compound | C-P protein | 映射类型 | 化合物生物标识 | 蛋白生物标识 | 实际 SMILES | 实际序列 |
|---|---:|---:|---|---:|---:|---:|---:|
| TCM-Suite | 1,187 | 7,258 | 无实体映射 | 0% | 0% | 0% | 0% |
| TCMSP | 6,929 | 1,748 | 匿名数字 ID 对照 | 0% | 0% | 0% | 0% |
| SymMap2.0 | 1,618 | 4,027 | 匿名数字 ID 列表 | 0% | 0% | 0% | 0% |
| ETCM2.0_core_mention10 | 9,519 | 509 | 生物学元数据 | 100% | 100% | 0% | 0% |

总体判定：

```text
blocked_cross_dataset_entity_alignment
```

TCMSP 的 `*_id_all.csv` 只是一个数字 ID 到另一个数字 ID 的映射，没有化合物名称、PubChem/TCMSP compound identifier、基因符号或 UniProt accession。SymMap2.0 的 `*_id_all.txt` 只有匿名数字 ID，TCM-Suite 当前没有实体映射文件。ETCM2.0 虽然具备化合物名称、TCMIP ID、分子式和 UniProt accession，但仓库中仍没有实际 SMILES 与蛋白序列文件。

因此，当前直接实现多模态模型只会得到 ETCM2.0 专用扩展，不能作为四数据集共享主模型，也不能在其他三个基准数据集上进行公平消融。

### 2.1 上游发布文件初查

HDCTI 论文的数据可用性声明只指向[作者 GitHub 仓库](https://github.com/tong87-bio/HDCTI)。该仓库当前发布的数据文件与本地清单一致：TCM-Suite 目录只有关系和样本文件，TCMSP 与 SymMap2.0 的映射仍是匿名数字 ID，没有另行发布名称、PubChem、UniProt 或 SMILES/sequence 对照表。因此，重新下载 HDCTI 仓库不能补齐实体映射。

后续溯源对象应是 TCM-Suite、TCMSP 和 SymMap2.0 各自数据库的原始版本、导出文件或作者预处理前对照表，而不是 HDCTI 仓库中已经匿名化的关系矩阵。若上游版本与论文抓取时间不一致，必须记录版本和日期，不能仅凭关系度数强行对齐。

## 3. 共享主模型门槛

在实现多模态主模型前固定以下门槛：

* 单数据集 SMILES 覆盖率至少 70%；
* 已映射 SMILES 的分子式确认率至少 95%；
* 单数据集蛋白序列覆盖率至少 95%；
* 至少三个数据集同时满足上述门槛。

标准化补全文件接口为：

```text
<alignment-root>/<dataset>/compound_attributes.csv
  entity_id,canonical_smiles,pubchem_cid,formula_match

<alignment-root>/<dataset>/protein_attributes.csv
  entity_id,uniprot_accession,sequence
```

补全后使用：

```bash
python tools/audit_multidataset_attributes.py \
  --alignment-root <alignment-root>
```

只有总体判定变为 `supports_cross_dataset_multimodal_pilot`，才允许把多模态作为共享模型创新。

## 4. 后续路线更新

本审计最初建议的“结构支持度自适应双专家冷启动框架”已经完成四库一折 Pilot，并因 macro AUPR 为负、SymMap2.0 明显退化而终止。后续的超边注意力、HILGA 和侧超图角色迁移也没有通过四库冻结闸门。因此，不能再把支持度路由或另一种匿名拓扑注意力写成本文档的下一步。

当前下一阶段改为“跨库实体映射恢复”，顺序固定为：

```text
原始数据库或作者发布文件溯源
        ↓
本地匿名 ID 与上游实体 ID 的可追溯对齐
        ↓
化合物名称/分子式与蛋白 accession 冲突审查
        ↓
PubChem/UniProt 属性补全
        ↓
重新运行本审计
```

对齐表至少记录：

```text
dataset,entity_type,local_entity_id,source_entity_id,
canonical_identifier,mapping_method,source_reference,
mapping_confidence,review_status
```

约束如下：

1. 不使用匿名数字 ID 直接查询 PubChem 或 UniProt。
2. 不把名称模糊匹配结果自动视为确定映射；一对多和多对一冲突必须保留并审查。
3. 化合物需要使用名称、来源 ID 和分子式共同校验；蛋白优先使用 UniProt accession，并记录物种。
4. 在至少三个数据集达到本文第 3 节门槛前，不实现共享多模态分支。
5. 若无法恢复 TCMSP 和 SymMap2.0 的上游标识，多模态继续限定为 ETCM2.0 扩展，不包装为跨数据库主创新。

当前模型主线仍冻结为 Strict-HDCTI + Hctx-P + CHCR；实体映射恢复是决定能否开启下一项信息增量型创新的数据前置工作，不是新的模型贡献。
