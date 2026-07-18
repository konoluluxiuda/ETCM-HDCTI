# 多数据集实体属性覆盖审计

## 1. 审计目的

本审计用于判断分子结构与蛋白序列多模态分支能否作为 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0 的共享主创新，而不是只在 ETCM2.0 上成立的附加实验。

初始全量审计只读取当前仓库中的 C-P 关系和实体映射，不访问网络、不下载属性、不训练模型。匿名数字 ID 只计为矩阵实体映射，不计为可检索的生物学标识；只有标准化 enrichment 文件中的真实 SMILES 和蛋白序列才计入模态覆盖。2026-07-18 另增加 TCMSP/ETCM 联网小样本审计，用于在全量补全前作 Go/No-Go 决策。

运行命令：

```bash
python tools/audit_multidataset_attributes.py \
  --entity-mapping 'SymMap2.0=results/symmap_official_alignment' \
  --alignment-root results/multidataset_attributes
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
| TCMSP | 6,929 | 1,748 | TCMSP 页面查询 ID | 100% | 100% | 0% | 0% |
| SymMap2.0 | 1,618 | 4,027 | 官方 V2 精确对齐 | 100% | 100% | 65.27% | 97.91% |
| ETCM2.0_core_mention10 | 9,519 | 509 | 生物学元数据 | 100% | 100% | 0% | 0% |

结合第 2.2 节小样本可行性审计后的总体判定：

```text
stop_shared_multimodal_enrichment
```

TCMSP 的 `*_id_all.csv` 中第一列 ID 可作为官方网站 molecule/target 页面的查询键，第二列是预处理生成的连续矩阵编号；既有候选核验已从页面返回 Molecule ID、Target ID、InChIKey、PubChem CID 和 DrugBank target 信息，因此 TCMSP 记为“可外部补全”。SymMap2.0 已使用官方 V2 SMIT/SMTT 文件完成 100% 精确 ID 对齐，并完成第一轮 PubChem/UniProt 属性补全。ETCM2.0 具备化合物名称、TCMIP ID、分子式和 UniProt accession。三个库均具备继续补全的来源标识，但当前只有 SymMap 已生成批量标准属性文件，且其 SMILES 尚未达到 70%；TCM-Suite 仍没有实体映射。

因此，当前已经满足“三库共享 Pilot”的映射前提，但未满足实际属性覆盖前提。SymMap2.0 的蛋白序列覆盖率为 97.91%，可核验 SMILES 的分子式确认率为 590/619（95.32%），两项已通过门槛；SMILES 覆盖率为 65.27%。第 2.2 节进一步确认 TCMSP 蛋白和 ETCM 成分自动覆盖率明显不足，继续补齐 SymMap 的 77 个成分也无法单独挽救三库共享多模态路线。

### 2.1 上游发布文件初查

HDCTI 论文的数据可用性声明只指向[作者 GitHub 仓库](https://github.com/tong87-bio/HDCTI)。该仓库当前发布的数据文件与本地清单一致，重新下载仓库本身不能补齐属性。进一步从 [SymMap 官方下载页](https://www.symmap.org/download/)取得 V2 SMIT/SMTT 后，确认本地 C-P 实体 ID 可与官方 `Mol_id/Gene_id` 精确对齐；TCMSP 第一列 ID 也可用于官网页面查询。当前上游映射缺口只剩 TCM-Suite。

上游映射审查证明 TCMSP、SymMap 和 ETCM 都存在可查询标识，但“可查询”不等于“可高置信补全”。小样本审计完成后，三库全量补全和 SymMap 102 条人工复核队列均暂停；已有缓存和工作清单保留，供后续单库扩展或案例研究使用。

### 2.2 TCMSP/ETCM 自动覆盖率小样本审计

运行命令：

```bash
python tools/audit_tcmsp_etcm_attribute_feasibility.py --sample-size 80
```

抽样方法为按 C-P degree 排序后的确定性系统抽样，每个数据集、每类实体抽取 80 个。TCMSP 成分使用官网 InChIKey 与 PubChem InChIKey 精确一致作为身份依据；TCMSP 蛋白只接受唯一的人类 UniProt 名称匹配；ETCM 成分要求名称检索后只有一个候选通过源分子式核验；ETCM 蛋白按已有 UniProt accession 直接解析。区间为 Wilson 95% CI。

| 数据集 | 指标 | 成功/样本 | 覆盖率 | 95% CI | 门槛 | 判定 |
|---|---|---:|---:|---:|---:|---|
| TCMSP | 可信 SMILES | 62/80 | 77.50% | 67.21%-85.27% | 70% | 尚不确定 |
| TCMSP | 身份核验 | 62/80 | 77.50% | 67.21%-85.27% | 95% | No-Go |
| TCMSP | 蛋白序列 | 42/80 | 52.50% | 41.70%-63.08% | 95% | No-Go |
| ETCM2.0-mention10 | 可信 SMILES | 44/80 | 55.00% | 44.12%-65.42% | 70% | No-Go |
| ETCM2.0-mention10 | 身份核验 | 44/50 | 88.00% | 76.20%-94.38% | 95% | No-Go |
| ETCM2.0-mention10 | 蛋白序列 | 80/80 | 100.00% | 95.42%-100.00% | 95% | Go |

失败状态未发现批量接口错误：TCMSP 蛋白包含 29 个无唯一匹配和 9 个多候选；ETCM 成分包含 30 个 PubChem 名称检索失败、3 个分子式冲突和 3 个多候选通过分子式。由此可将不足归因于实体标识和命名质量，而不是临时网络故障。

机器可读结果位于：

```text
results/tcmsp_etcm_attribute_feasibility/report.json
results/tcmsp_etcm_attribute_feasibility/report.md
results/tcmsp_etcm_attribute_feasibility/*_sample.csv
```

## 3. 共享主模型门槛

在实现多模态主模型前固定以下门槛：

* 单数据集 SMILES 覆盖率至少 70%；
* 可核验 SMILES 的分子式确认率至少 95%，缺失源分子式的记录不进入分母；
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

只有总体判定变为 `supports_cross_dataset_multimodal_pilot`，才允许把多模态作为共享模型创新。当前判定已变为 `stop_shared_multimodal_enrichment`：不实施三库共享多模态模型，也不为达到门槛而继续人工修补 SymMap。ETCM 与 SymMap 的高覆盖蛋白序列可以保留为单模态、单库扩展候选，但不能据此声称实现跨库统一多模态。

## 4. 后续路线更新

本审计最初建议的“结构支持度自适应双专家冷启动框架”已经完成四库一折 Pilot，并因 macro AUPR 为负、SymMap2.0 明显退化而终止。后续的超边注意力、HILGA 和侧超图角色迁移也没有通过四库冻结闸门。因此，不能再把支持度路由或另一种匿名拓扑注意力写成本文档的下一步。

跨库实体映射恢复的最低三库目标已经完成，小样本审计也已完成。当前顺序更新为：

```text
冻结现有映射、样本结果和查询缓存
        ↓
停止三库共享多模态全量补全
        ↓
主线返回 Strict-HDCTI + Hctx-P + CHCR
        ↓
属性仅用于 ETCM/SymMap 单库案例或后续工作
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
4. 当前样本证据已触发 No-Go，不实现共享多模态分支，也不通过扩大人工审查量绕过预注册门槛。
5. TCM-Suite 映射不阻塞最低三库 Pilot，但在恢复前不得声称实现了四库统一多模态模型。

当前模型主线仍冻结为 Strict-HDCTI + Hctx-P + CHCR；三库真实属性覆盖 Go/No-Go 已完成并关闭共享多模态路线。下一步应整理主模型四库结果与 ETCM 实体映射下的案例解释，而不是继续进行高成本全量属性补全。
