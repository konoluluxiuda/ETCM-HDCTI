# 全候选靶点排名 Gate

## 1. 审计动机

已有 compound cold-start 主结果在每个测试正例旁采样一个未观测 pair，适合与
历史 AUC/AUPR 协议比较，但不能证明模型能够从完整靶点库中找回真实靶点。
非神经对照进一步显示，折内蛋白流行度和 H-C 相似性即可在该采样协议上取得很高
AUPR。因此，当前论文主张必须增加不重新训练的全候选排名 Gate。

该 Gate 固定使用：

```text
四个数据库
compound cold-start seed 52026
原有五折 assignments
已保存的 20 个 Hctx-P + SDIS + SCHPT checkpoint
所有 C-P 或 P-D 中出现的蛋白候选
过滤外层训练正边，保留外层测试正边
其余 pair 仅标记为 unlabeled
```

模型和启发式均调用 `util.checkpoint_ranking.evaluate_fixed_candidate_ranking`，
报告 compound-macro MRR、Precision@K、Recall@K、Hits@K 和 Enrichment@K。

## 2. 非神经对照

三个对照只读取当前折训练 C-P 正边和固定 H-C：

1. `GlobalPrior`：折内蛋白流行度；
2. `HerbPrototype-EB`：带经验贝叶斯平滑的药材—靶点原型；
3. `HC-Jaccard-LP`：从 H-C Jaccard 相似训练成分传播 C-P 标签。

四库五折启发式排名已经完成：

| 数据库 | 最佳 MRR | 方法 | 最佳 Recall@20 | 方法 |
|---|---:|---|---:|---|
| TCM-Suite | 0.291811 | HC-Jaccard-LP | 0.211155 | HC-Jaccard-LP |
| TCMSP | 0.652160 | HC-Jaccard-LP | 0.653517 | HC-Jaccard-LP |
| SymMap2.0 | 0.486224 | HC-Jaccard-LP | 0.383769 | HC-Jaccard-LP |
| ETCM2.0-mention10 | 0.657789 | HerbPrototype-EB | 0.654998 | HerbPrototype-EB |

## 3. 四库首折模型试跑

20 个 checkpoint 均已核验完整。为避免在方向性失败后继续消耗资源，先只恢复
每库第 1 折。恢复后的采样 AUPR 与原 SCHPT 逐折记录一致，候选数也与启发式
完全相同，因此不存在 checkpoint、fold 或实体全集错配。

| 数据库 | 候选蛋白数 | 测试 compound | Ours MRR | 同折最佳启发式 MRR | 差值 | Ours Recall@20 | 同折最佳启发式 Recall@20 | 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 7,258 | 236 | 0.049144 | 0.306778 | -0.257635 | 0.013899 | 0.225820 | -0.211921 |
| TCMSP | 1,748 | 1,384 | 0.539896 | 0.654713 | -0.114817 | 0.568948 | 0.651503 | -0.082556 |
| SymMap2.0 | 18,192 | 324 | 0.297357 | 0.476281 | -0.178924 | 0.202976 | 0.420865 | -0.217889 |
| ETCM2.0-mention10 | 509 | 1,904 | 0.600697 | 0.672832 | -0.072135 | 0.553732 | 0.657382 | -0.103650 |

四库首折均为负方向，说明此前较高的 sampled AUPR 不能直接外推为完整候选靶点
检索能力。当前 Gate 状态为 `INCOMPLETE`，但在修改训练目标前没有必要机械完成
剩余 16 个 checkpoint 推理。

## 4. 支持度诊断

| 数据库 | 模型 Top-20 零训练支持靶点比例 | 测试阳性靶点平均训练 degree | 模型 Top-20 平均训练 degree |
|---|---:|---:|---:|
| TCM-Suite | 25.53% | 16.84 | 6.81 |
| TCMSP | 0.01% | 632.56 | 567.27 |
| SymMap2.0 | 35.19% | 53.50 | 110.22 |
| ETCM2.0-mention10 | 0.02% | 942.88 | 1,026.55 |

TCM-Suite 和 SymMap 的问题部分来自模型把大量当前折无 C-P 支持靶点排入前列；
TCMSP 和 ETCM 几乎不存在该现象，却仍低于简单启发式。因此更一般的原因是：
当前 1:1 pair BCE 只学习局部二分类边界，没有直接优化同一 compound 下所有候选
protein 的相对顺序。

## 5. 决策

1. 旧 sampled AUC/AUPR 表保留为与 HDCTI 原论文兼容的历史协议结果，但不作为
   “完整候选检索优越性”的证据。
2. 暂停剩余 16 个 checkpoint 全候选推理，不根据 outer-test 结果调参数。
3. 下一步先在 inner-validation 上实现 compound-centric 排名目标审计：固定每个
   compound 的正例、度数匹配未观测候选和高分困难候选，比较 BCE 与 pairwise
   ranking 的 Recall@20/MRR。
4. 只有 inner-validation Pilot 同时超过当前模型和三种启发式，才生成新 checkpoint
   并重新执行四库五折全候选 Gate。
5. 若 Pilot 失败，则当前模型只能定位为“采样 pair 分类模型”，期刊主张和标题必须
   相应收窄。

随后完成的 inner-validation 全候选审计在完全不评分 outer-test 的条件下得到同向
结论，四库 MRR 差值为 `-0.248574/-0.094332/-0.158062/-0.075583`，
Headroom Gate 为 `PASS`。因此第 3 项现已满足，只允许执行一次预注册的冻结编码器
ranking-head Pilot。详细记录见
[Inner-Validation 全候选排名审计](INNER_FULL_CANDIDATE_RANKING_AUDIT.md)。

## 6. 复现命令

完整 dry-run：

```bash
PYTHON_BIN=/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
./run_full_candidate_ranking_gate.sh --dry-run
```

启发式四库五折：

```bash
PYTHON_BIN=/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
./run_full_candidate_ranking_gate.sh --heuristics-only
```

单库单折纯推理：

```bash
PYTHON_BIN=/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
./run_full_candidate_ranking_gate.sh --ours-only --dataset tcmsuite --fold 1
```

当前结果：

```text
results/full_candidate_ranking_gate/heuristics/summary.json
results/full_candidate_ranking_gate/summary.json
results/full_candidate_ranking_gate/summary.md
```
