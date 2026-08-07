# Inner-Validation 全候选排名 Headroom 审计

## 1. 目的

旧 Compound-Centric 审计只比较随机折 inner-validation 中每个 compound 的
1:1 正负 pair，得到较高 MRR，因而停止 ranking loss。新的全候选 outer 审计
证明该结论不能外推到 compound cold-start 检索任务。本审计重新使用最终
SCHPT checkpoint，但只评价 inner-validation，禁止读取 outer-test 指标。

## 2. 冻结协议

```text
split: compound_cold_start
seed: 52026
fold: 每库固定第 1 折
checkpoint: 对应 SCHPT 最终第 1 折 checkpoint
training/optimizer steps: 0
candidate proteins: C-P 或 P-D 中出现的全部 protein
filter: model-train positives
target: inner-validation positives
outer-test scored: false
```

比较三种仅使用 model-train C-P 与固定 H-C 的启发式：GlobalPrior、
HerbPrototype-EB 和 HC-Jaccard-LP。进入一次固定 ranking-loss Pilot 的门槛为：
至少三个数据库上，Ours 相对该指标最佳启发式的 MRR 或 Recall@20 落后至少
`0.02`。该门槛只判断是否存在 headroom，不构成性能提升证据。

## 3. 结果

| 数据库 | Ours MRR | 最佳启发式 MRR | 差值 | Ours Recall@20 | 最佳启发式 Recall@20 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 0.045879 | 0.294453 | -0.248574 | 0.024673 | 0.284564 | -0.259891 |
| TCMSP | 0.524737 | 0.619069 | -0.094332 | 0.571568 | 0.638126 | -0.066558 |
| SymMap2.0 | 0.361654 | 0.519716 | -0.158062 | 0.199801 | 0.392894 | -0.193093 |
| ETCM2.0-mention10 | 0.572435 | 0.648017 | -0.075583 | 0.539382 | 0.648908 | -0.109526 |

四库均满足 headroom 条件，Gate 为 `PASS`。所有报告中的
`outer_test_scored=false`，模型与启发式使用相同候选数和过滤规则。

## 4. 决策

允许实现一次固定的 validation-only ranking Pilot。Pilot 不重新搜索图编码器，
而冻结已有表示并训练轻量 cold-ranking head，以判断排序目标是否能改善完整候选
检索：

```text
输入：Hctx-P 逐维交互、SCHPT LOCO residual、折内 protein support prior
训练单位：同一 compound 的 (positive, negative) pair
负例：50% degree-matched + 50% 当前冷启动分数 hard negative
目标：pairwise logistic softplus(s_neg - s_pos)
编码器：冻结
outer-test：禁止评分
```

固定 Pilot 不搜索 margin、temperature、隐藏层或负例比例。进入新 checkpoint
训练的门槛为：

1. 四库首折中至少 3/4 的 inner-validation MRR 高于原 checkpoint；
2. macro MRR 与 macro Recall@20 均提高至少 `0.02`；
3. 至少 2/4 数据库超过其最佳非神经启发式；
4. 任一数据库 MRR 下降不得超过 `0.02`。

未通过即停止该 ranking-head 方案，不调整上述固定设置。

## 5. 复现

```bash
PYTHON_BIN=/home/zry/.conda/envs/HDCTI_tfnew/bin/python \
./run_inner_full_candidate_ranking_audit.sh
```

机器可读结果：

```text
results/inner_full_candidate_ranking_audit/summary.json
results/inner_full_candidate_ranking_audit/heuristics/summary.json
```
