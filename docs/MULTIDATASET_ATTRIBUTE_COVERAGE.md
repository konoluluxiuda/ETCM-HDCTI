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

## 4. 对下一创新的约束

当前第三创新必须优先使用四个数据集共同拥有的 H-C、C-P 和 P-D 结构。更合适的候选是“结构支持度自适应双专家冷启动框架”：
```text
协同结构专家：原始 HDCTI / Dot 分数
药材上下文专家：Hctx-P + CHCR
支持度路由：根据训练 C-P degree 与 H-C 覆盖选择专家权重
伪冷启动训练：按 compound 整体隐藏 C-P 边，训练上下文专家
```

该路线可以在四个数据集上使用同一输入关系和同一实现，并直接面向 compound C-P cold-start 问题。结构可行性审计已经完成：四库 H-C 支撑率均不低于 98.64%，平衡后的每折正边均超过 7,500 条，因而都可以进入统一 compound cold-start Pilot。CHCR 在 SymMap2.0 仅覆盖 81.58% compound 和 33.40% C-P 正边，后续必须选择性启用，不能作为路由框架成立的前提。详见 [MULTIDATASET_COLD_START_FEASIBILITY.md](MULTIDATASET_COLD_START_FEASIBILITY.md)。

下一步先在四库统一的一折 cold-start 协议下比较 NoContext 与 Hctx-P。若没有稳定的跨数据集互补空间，则停止该路线，不实现路由网络。

多模态仍可作为 ETCM2.0 扩展或后续工作，但在其他基准数据库实体映射补齐前，不进入当前论文的共享主贡献。
