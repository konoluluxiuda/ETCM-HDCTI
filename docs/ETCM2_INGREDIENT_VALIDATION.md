# ETCM2.0 Ingredient 外部验证证据库

## 1. 定位

`dataset/ETCM2.0/etcm_ingredients` 中的成分页不参与模型训练、超参数选择、早停或模型选择。本项目将其加工为独立的外部证据库，用于：

1. 冻结 checkpoint 后评价未见确认靶点的排名；
2. 核验 Top-K 预测是否得到 ETCM2.0 确认关系支持；
3. 为 Hctx-P、CHCR 等药材上下文方法提供独立药材来源；
4. 在案例研究中提供名称、CAS、PubChem、活性和参考文献。

禁止将生成目录中的确认靶点或潜在靶点重新写入训练使用的 `C_P.txt`、`ONE_indices.txt`、交叉验证 split 或任何标签依赖图。

## 2. 原始数据

原始目录：

```text
dataset/ETCM2.0/etcm_ingredients
```

全量审查结果：

| 项目 | 数量 |
|---|---:|
| JSON 文件 | 38,281 |
| 成功页面 | 38,272 |
| 无数据页面 | 9 |
| 成功率 | 99.9765% |
| 原始体积 | 138,578,840,653 bytes（129.08 GiB） |

9 个失败页面均返回 `No Data`，列表保存在：

```text
dataset/ETCM2.0_validation/audit/failed_pages.tsv
```

选择性解析异常数为 0。

## 3. 处理方法

构建脚本：

```text
tools/build_etcm2_validation_evidence.py
```

脚本采用逐页选择性解析：

- 解析 `Basic Information`；
- 解析 `Herbs`；
- 分开解析 `Confirmed Targets` 与 `Potential Targets`；
- 跳过疾病、方剂和中成药的完整明细，仅从 `complete_fetch_summary` 保留计数；
- 使用 TCMIP ID 对齐 compound，使用大写 Gene Symbol 对齐 protein；
- 分别与五个 ETCM 数据目录的 `C_P.txt` 比较；
- 输出训练重叠、训练未见确认边和 OOV 三类关系；
- 为全部生成文件写入 SHA-256 manifest。

这样避免复制 1.21 亿条富集疾病、4,559 万条方剂和 906 万条中成药记录。最终验证证据库约 130 MB，不是第二份 130 GB 原始数据。

重建命令：

```bash
python tools/build_etcm2_validation_evidence.py \
  --input dataset/ETCM2.0/etcm_ingredients \
  --output dataset/ETCM2.0_validation \
  --progress-every 500
```

重新生成已有目录时显式增加：

```bash
--overwrite
```

## 4. 输出结构

```text
dataset/ETCM2.0_validation/
├── manifest.json
├── statistics.json
├── statistics.md
├── audit/
│   ├── failed_pages.tsv
│   └── parse_issues.tsv
├── entities/
│   ├── ingredient.tsv
│   └── ingredient_alias.tsv
├── relations/
│   ├── ingredient_herb.tsv
│   ├── confirmed_target_pairs.tsv
│   ├── confirmed_target_evidence.tsv
│   ├── potential_target_pairs.tsv
│   └── potential_target_evidence.tsv
└── validation/
    ├── ETCM2.0_processed/
    ├── ETCM2.0_core/
    ├── ETCM2.0_core_mention10/
    ├── ETCM2.0_core_cpdeg3/
    └── ETCM2.0_core_cpdeg5/
```

每个 `validation/<dataset>/` 目录包含：

```text
training_overlap.tsv
unseen_confirmed.tsv
out_of_vocabulary.tsv
```

## 5. 基础实体统计

| 字段 | 数量 | 覆盖率 |
|---|---:|---:|
| 英文名称 | 38,272 | 100.00% |
| 分子式 | 38,272 | 100.00% |
| 分子量 | 38,272 | 100.00% |
| 2D 结构链接 | 38,272 | 100.00% |
| 别名 | 19,417 | 50.73% |
| PubChem CID | 19,405 | 50.70% |
| CAS | 16,926 | 44.23% |

实体映射结果：

| 状态 | 数量 |
|---|---:|
| 成功映射到 `ETCM2.0_processed` | 38,238 |
| 未映射的新成分 | 34 |

页面本身没有直接给出 SMILES 或 InChIKey。PubChem CID 可用于后续案例研究中的结构属性补全，但不应在冻结模型后反向进入训练。

## 6. 关系统计

### 6.1 页面原始关系

| 关系 | 有数据页面 | 原始行数 |
|---|---:|---:|
| 药材 | 38,272 | 67,377 |
| 确认靶点 | 1,810 | 13,254 |
| 潜在靶点 | 22,558 | 186,642 |
| 富集疾病 | 19,263 | 120,906,268 |
| 中成药 | 29,779 | 9,060,970 |
| 方剂 | 27,547 | 45,594,788 |

所有 `complete_fetch_summary` 均满足 `expected == actual`。其中 `status=skipped` 表示原页面已完整、不需要续抓，不表示关系缺失。

### 6.2 紧凑关系

| 关系 | 原始证据行 | 去重关系 |
|---|---:|---:|
| Confirmed Target | 13,254 | 4,753 |
| Potential Target | 186,642 | 181,121 |
| Ingredient-Herb | 67,377 | 67,375 |

确认靶点的 13,254 条证据全部具有 Activity 和 Reference，且没有 `Similar Score`。潜在靶点证据全部具有 `Similar Score`，必须与确认关系分开使用。

确认关系映射状态：

| 状态 | 唯一关系 |
|---|---:|
| compound 和 protein 均成功映射 | 4,739 |
| protein 未映射 | 13 |
| compound 未映射 | 1 |

## 7. 与当前 ETCM 训练标签的关系

当前 `ETCM2.0_processed/C_P.txt` 来自药材页的 `Component Target`。原始构建数据的 385,362 条 Component Target 记录全部带有 `Similar Score`，范围为 `0.8-1.0`，说明当前训练标签主要是相似性迁移得到的候选关系，而不是全部实验确认关系。

成分页的确认关系与训练数据比较如下：

| 数据集 | C-P 训练边 | 训练重叠 | 未见确认边 | OOV |
|---|---:|---:|---:|---:|
| `ETCM2.0_processed` | 180,589 | 2,624 | 2,115 | 14 |
| `ETCM2.0_core` | 109,747 | 1,606 | 953 | 2,194 |
| `ETCM2.0_core_mention10` | 88,431 | 1,284 | 685 | 2,784 |
| `ETCM2.0_core_cpdeg3` | 99,450 | 1,382 | 746 | 2,625 |
| `ETCM2.0_core_cpdeg5` | 86,555 | 1,151 | 623 | 2,979 |

解释：

- `training_overlap` 只能用于数据库一致性检查，不能证明模型发现了未见关系；
- `unseen_confirmed` 是 compound 和 protein 均在模型实体空间中、但 pair 未进入该数据集 C-P 训练边的确认关系；
- `out_of_vocabulary` 至少有一个实体不在对应模型实体空间中，不能把它按模型预测失败处理。

主 ETCM 实验若采用 `ETCM2.0_core_mention10`，当前可直接用于纯推理外部评价的是 685 条 `unseen_confirmed` 关系。

## 8. 外部验证协议

验证必须在模型、checkpoint、候选集合和评价规则冻结后执行：

1. 加载已保存 checkpoint，不重新训练；
2. 对每个可评分 compound 的统一 protein 候选全集打分；
3. 过滤该模型训练中的已知 C-P 边；
4. 保留 `unseen_confirmed.tsv` 中的确认靶点；
5. 计算确认靶点排名。

建议报告：

```text
Hits@1/5/10/20
Recall@5/10/20
MRR
Mean Percentile Rank
Enrichment Factor@K
```

未记录关系属于 unlabeled，不应直接作为可靠生物学负样本。普通 AUC 只有在明确、冻结且经过审计的候选负例协议下才能作为补充指标。

## 9. 案例解释

对模型 Top-K 预测可按以下顺序查询证据：

1. `confirmed_target_pairs.tsv`：是否存在 ETCM 确认关系；
2. `confirmed_target_evidence.tsv`：具体 Activity 与 DOI/PubMed/BindingDB；
3. `ingredient_herb.tsv`：预测成分的独立药材来源；
4. `potential_target_pairs.tsv`：仅作为 ETCM 推断网络一致性参考；
5. `ingredient.tsv`：名称、CAS、PubChem、QED 和药代属性。

Hctx-P 或 CHCR 的解释应比较模型高权重药材与 `ingredient_herb.tsv` 的独立来源是否一致。潜在靶点不能与模型预测互相充当独立验证，因为二者可能共享相似性迁移信号。

## 10. 可复现性与版本边界

`manifest.json` 明确记录：

```text
purpose=external_validation_and_explanation_only
training_use_prohibited=true
```

同时包含：

- 原始文件数和总体大小；
- 基于文件名与大小的 source inventory SHA-256；
- 五个被审计数据集的绝对路径；
- 每个生成文件的 SHA-256；
- 每个数据集的三类确认关系计数。

大型 `dataset/` 目录继续由 `.gitignore` 排除。Git 仅跟踪构建脚本、测试和本文档。
