# ETCM2.0 Top-K 案例解释

## 1. 当前状态

ETCM2.0 ingredient 外部证据库已经完成。案例研究采用冻结 checkpoint
纯推理，不重新训练、不调参，也不把 ingredient Confirmed/Potential Target
写回模型。

当前已完成：

1. 冻结 fold 1 的 Strict-HDCTI、Hctx-P 和 Hctx-P+CHCR checkpoint；
2. 校验三组配置、Strict split 和 checkpoint SHA-256；
3. 在查看模型分数前，从 332 个可评分且具有未见确认关系的 compound 中，
   按实际 model-train C-P degree 分层选择 5 个案例；
4. 完成纯推理脚本 dry-run；
5. 在 selection manifest 不变的前提下完成三组 checkpoint 的正式纯推理；
6. 在排名冻结后接入原始 Herbs、Targets、Diseases 页面，生成带来源边界的
   post-hoc 上下文与机制路径注释。

## 2. 冻结 checkpoint

清单：

```text
configs/etcm_topk_case_checkpoints.json
```

当前使用 fold 1：

| 模型 | Checkpoint |
|---|---|
| Strict-HDCTI | `saved_model/2026-07-22 16-46-39/hdcti_model.ckpt` |
| Hctx-P | `saved_model/2026-07-17 17-32-22/hdcti_model.ckpt` |
| Hctx-P+CHCR | `saved_model/2026-07-17 17-42-26/hdcti_model.ckpt` |

三组模型复用相同 pair-stratified Strict split、fold 1、inner-validation
划分和 mention10 实体空间。候选集为 509 个 protein，排名前过滤完整
`ETCM2.0_core_mention10/C_P.txt` 中的 88,431 条已知关系。

## 3. 案例预选

生成命令：

```bash
/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
  tools/prepare_etcm_topk_cases.py
```

冻结输出：

```text
configs/etcm_topk_case_selection.json
results/etcm_topk_cases/selection/case_selection_manifest.json
results/etcm_topk_cases/selection/eligible_candidates.tsv
results/etcm_topk_cases/selection/selected_cases.tsv
```

选择规则不读取 checkpoint 分数：

1. compound 和 confirmed target 均位于 mention10 模型空间；
2. 至少存在一条训练数据未见的 Confirmed Target；
3. ingredient 身份信息和独立 Herb 来源可用；
4. 按 fold 1 model-train C-P degree 做 rank-based tertiles；
5. 低、中、高支持层按 `2/1/2` 选择；
6. 层内依次优先 CAS/PubChem、未见确认靶点数、确认记录数、Herb 来源数和
   mention count，最后使用固定 seed 的 SHA-256 打破平局。

冻结案例：

| 层级 | 成分 | Model-train C-P degree | 未见确认靶点 |
|---|---|---:|---:|
| Low | DEXPROPRANOLOL | 3 | 8 |
| Low | gallocatechin gallate | 3 | 7 |
| Medium | (+)-Gallocatechin | 4 | 7 |
| High | Sulfuretin | 15 | 7 |
| High | Quercetin | 30 | 11 |

上述选择只能在重新制定并记录案例协议时改变，不能根据 Top-K 命中结果替换
“表现不好”的案例。

## 4. 纯推理检查

不导入 TensorFlow、不恢复权重的检查：

```bash
/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
  tools/evaluate_etcm_topk_cases.py \
  --dry-run
```

该命令验证：

- selection manifest 状态为 `frozen_before_checkpoint_scoring`；
- 三组 config、checkpoint、split 和证据库 SHA-256 未变化；
- 三组模型使用同一 model-train、validation 和 outer-test；
- 509 个候选 protein 与 88,431 条过滤关系可用；
- optimizer steps 和 checkpoint updates 均为 0。

## 5. 正式推理

本次 selection manifest 在打分前生成并冻结 SHA-256，推理报告记录该哈希。
后续正式复算应先确认该文件已提交 Git。案例工具不需要通过训练入口，命令为：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
  tools/evaluate_etcm_topk_cases.py
```

脚本依次恢复三个 checkpoint。它会构建模型计算图，但不会执行 optimizer
或保存权重。

## 6. 输出

```text
results/etcm_topk_cases/fold1/
├── report.json
├── checkpoint_manifest.json
├── topk_predictions.tsv
├── evidence_annotated_topk.tsv
├── confirmed_target_ranks.tsv
├── herb_context_explanations.tsv
├── case_metrics.tsv
├── aggregate_case_metrics.tsv
├── case_summary.md
└── context/
    ├── manifest.json
    ├── context_annotated_topk.tsv
    ├── mechanism_paths.tsv
    ├── herb_entities.tsv
    ├── target_entities.tsv
    ├── disease_entities.tsv
    ├── case_context_summary.tsv
    └── summary.md
```

`evidence_annotated_topk.tsv` 的证据等级：

| 等级 | 含义 |
|---|---|
| A | mention10 训练中未见的 ETCM Confirmed Target，带 Activity/Reference |
| C | ETCM Potential Target，只表示推断网络一致性 |
| E | 当前 ETCM target 证据未覆盖，属于 unlabeled |

B 级独立数据库或文献证据需要在 Top-K 冻结后人工核验。原始
Herb/Disease 页面形成的 C-H-D-P 路径记为 D 级算法/数据库机制假设；
既有 H-D 来源审查表明它不能升级为独立验证证据。

## 7. 当前结果

五个案例共有 40 条 `unseen_confirmed` 关系。Top-20 中的 A 级确认关系数：

| 模型 | A 级 Top-20 命中 | Macro MRR | Macro Recall@10 | Macro Recall@20 |
|---|---:|---:|---:|---:|
| Strict-HDCTI | 16/40 | 0.287794 | 0.231169 | 0.342857 |
| Hctx-P | 23/40 | 0.353420 | 0.226623 | 0.538961 |
| Hctx-P+CHCR | 28/40 | 0.733333 | 0.387662 | 0.675000 |

按成分观察：

- DEXPROPRANOLOL 的首个确认靶点由 Strict rank 23 提升到 Hctx-P rank 3，
  CHCR rank 1；
- `(+)-Gallocatechin` 由 Strict rank 19 提升到 Hctx-P/CHCR rank 1；
- gallocatechin gallate 的 Hctx-P 首位确认关系略有下降，但 CHCR 提升到
  rank 1；
- Sulfuretin 的改善较小；
- Quercetin 的 Strict 结果已经很强，Hctx-P 首位排名下降，CHCR 恢复到
  rank 2，Top-20 仍找回全部 11 个确认靶点。

三组模型的 300 条 Top-20 记录中，A 级分别为 16、23 和 28 条；其余均为
E 级，本次没有 Potential Target 落入 Top-20。该结果说明药材上下文在部分
低、中支持案例上显著前移外部确认靶点，但并非所有成分都单调改善。

上述数字只来自 5 个在查看分数前冻结的证据丰富案例，不能替代 332 个
compound、685 条确认关系的全量外部排名评价，也不能用于反向选择模型。

## 8. 原始多实体页面注释

生成命令：

```bash
python tools/annotate_etcm_topk_context.py
```

该脚本不加载 TensorFlow、不恢复 checkpoint、不重新选择案例，也不改变
`evidence_annotated_topk.tsv` 中的分数和排名。它复用已经从原始页面抽取且
解析异常为 0 的关系：

| 关系/信息 | 原始来源 | 当前用途 | 是否独立验证 |
|---|---|---|---|
| H-C | Herb 页 ingredient/network | 解释成分的药材来源 | 否，属于模型侧信息 |
| P-D | Disease 页 target | 解释靶点疾病语义 | 否，属于模型侧信息 |
| H-D | Disease 页 herb | 构造 C-H-D-P 路径假设 | 否，与完整 C-P 网络高度耦合 |
| Target identity | Target 详情页及其他页面 Target 表 | 核对 Gene/UniProt/名称 | 不验证 C-P |

当前 300 条 Top-20 记录包含 167 个唯一成分-蛋白候选，其中 68 个至少存在
一条 C-H-D-P 页面路径，共形成 7,073 条去重路径。26/26 个涉及药材和
375/375 个涉及疾病可回链到原始 JSON；101 个涉及靶点中 22 个具有独立
Target 详情页，其余靶点身份来自 Herb/Disease 页的 Target 表和加工后的
实体映射。

按案例统计：

| 成分 | Herb 数 | 路径支持的 Top-K 唯一候选 | 路径数 |
|---|---:|---:|---:|
| DEXPROPRANOLOL | 2 | 1 | 1 |
| gallocatechin gallate | 1 | 27 | 4,423 |
| (+)-Gallocatechin | 19 | 24 | 2,527 |
| Sulfuretin | 3 | 0 | 0 |
| Quercetin | 1 | 16 | 122 |

路径数量受疾病集合宽泛程度和实体度数强烈影响，不能当作预测置信度或多个
独立证据计数。`mechanism_paths.tsv` 保留每条路径的 Herb、Disease、
Target 原始页面位置以及各关系是否参与模型侧信息，供后续筛选少量路径作图。

## 9. 解释边界

- Top-K 案例用于解释，不用于重新选择模型；
- Potential Target 与当前训练图可能共享相似性迁移信号，不是独立真值；
- 没有 ETCM target 证据不等于确认负例；
- `rank_gain_vs_strict = Strict rank - 当前模型 rank`，正值表示前移；
- Hctx-P/CHCR 的上下文权重不是生物学因果证据；
- 原始 Herb/Disease 页面关系可增强可追溯性，但不能重复包装为外部验证；
- SDIS 属于 compound cold-start 专用机制，不进入本次普通 pair-unseen
  案例主表。

## 10. 下一步

当前唯一优先任务是完成冻结 E 级候选的独立证据核验，不再训练模型。

执行顺序：

1. 从三个模型的 Top-20 中合并 E 级候选，按 `compound-protein` 去重；
2. 在不读取后续检索结果的前提下，优先选择跨模型重复出现、排名靠前且实体
   身份明确的 10–20 个 pair，生成冻结核验清单和 SHA-256 manifest；
3. 使用 BindingDB、ChEMBL、PubMed 逐项检索，记录检索日期、查询式、数据库
   记录号、DOI/PMID、关系类型和人工判定；
4. 将存在直接结合或活性证据的候选升级为 B 级；只有通路或共现证据的候选
   保持 D/E 级，不能写成确认 C-P；
5. 从 `context/mechanism_paths.tsv` 选择 2–3 个来源完整、疾病集合不过度宽泛且
   能与 A/B 级关系对应的路径制作机制图；
6. 汇总案例表、核验表和路径图后进入论文 Results/Case study 写作。

验收文件计划：

```text
configs/etcm_topk_manual_validation.json
results/etcm_topk_cases/manual_validation/candidate_manifest.json
results/etcm_topk_cases/manual_validation/evidence_review.tsv
docs/ETCM_TOPK_MANUAL_VALIDATION.md
```

边界：不替换未命中的冻结案例，不根据人工核验结果调参、改 checkpoint、
改变候选排名或重新选择方法。
