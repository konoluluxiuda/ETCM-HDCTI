# SCHPT 第三算法候选预注册

## 1. 目的

`SCHPT`（Support-Calibrated Herb Prototype Transfer，支持度校准药材原型迁移）用于替换 HDCTI 的成分侧 C-P PageRank。蛋白侧 P-D PageRank、静态 Hctx-P 与 SDIS 保持不变。

该候选来自 LOCO 双超图迁移审计：四库中 `Compound-cold / Target-warm` 的 H-C→C-P 迁移信号稳定，但完整双向替换因覆盖不足和 cold-cold 退化被判定为 No-Go。

## 2. 固定方法

每折只使用当前训练折的 C-P 正边。药材 `h` 对靶点 `p` 的统计只包含具有训练 C-P 支撑的药材成员。评价候选 `(c,p)` 时，动态删除 `c` 的支撑状态及 `(c,p)` 正边贡献：

```text
posterior(h,p|-c) = (count(h,p|-c) + kappa * prevalence(p))
                    / (supported_members(h|-c) + kappa)

residual(c,p) = mean_h[posterior(h,p|-c) - prevalence(p)]
```

固定 `kappa=1`。没有其他受支持药材成员时，该项严格回退为 `0`。最终只增加一个从 `0` 初始化的标量参数：

```text
logit(c,p) += alpha * residual(c,p)
```

候选启用时不再执行成分侧 C-P PageRank，但保留蛋白侧 P-D PageRank。

## 3. 隔离实验

数据集：`ETCM2.0_core_mention10`。

协议：compound cold-start、严格折内构图、仅第 1 折内层验证、全节点稠密注意力关闭、外层测试关闭。使用此前未参与方法判断的固定 seed `52026`。

对照与候选除 SCHPT 开关及其所替换的成分 PageRank 外完全一致：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_schpt_baseline_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_schpt_pilot.conf
```

推荐使用自动校验配置哈希并生成 Gate 汇总的统一入口：

```bash
./run_schpt_pilot.sh
```

冻结配置及 SHA-256 记录在 `configs/schpt_pilot_manifest.json`。

## 4. 预注册 Gate

只有同时满足下列条件才进入四库确认：

1. 候选内层验证 AUPR 相对配对基线提升至少 `0.003`；
2. 验证 pair 的原型证据覆盖率至少 `30%`；
3. 保存元数据中的 `abs(learned_scale) > 1e-6`；
4. 训练与验证不读取外层测试结果；
5. 单元测试证明 LOCO 去自身贡献和无证据回退成立。

若 Gate 失败，不围绕 `kappa`、seed 或阈值做搜索。仅允许检查实现错误或证据覆盖失败原因，然后冻结为负结果。

## 5. 当前边界

首次配对 Pilot 已于 2026-08-05 完成并通过 Gate：

| 指标 | 基线 | SCHPT | 差值/数值 |
|---|---:|---:|---:|
| Validation AUPR | 0.906082 | 0.912476 | +0.006394 |
| 原型证据覆盖率 | - | - | 99.42% |
| 原型残差平均绝对值 | - | - | 0.078485 |
| learned scale | - | - | 1.670859 |

验证正例与负例的平均原型残差分别为 `+0.112150` 与 `-0.003324`。这说明原型信号方向正确，且模型没有把从 `0` 初始化的分支留在未使用状态。训练在 epoch 10 的最佳 inner-validation checkpoint 早停，outer test 未用于选择。

当前证据升级为 **E1：ETCM 新 seed 单折 inner-validation PASS**，仍不是已成立论文贡献。

## 6. 四库 Gate 1

下一阶段冻结四库同 seed、同 fold-1 inner-validation 配对实验：

```bash
./run_schpt_gate1.sh
```

四库 Gate 在运行前固定为：

1. 四库平均 Validation-AUPR 增量至少 `+0.003`；
2. 至少 `3/4` 数据集增量为正；
3. 任一数据集增量不得低于 `-0.003`；
4. 每库证据覆盖率至少 `30%`，learned scale 非零；
5. 全部实验保持 `evaluation.outer.test=False`。

配置和 SHA-256 见 `configs/schpt_gate1_manifest.json`。只有 Gate 1 通过，才构建四库五折确认配置。

首次四库运行在 `SymMap2.0 candidate` 初始化阶段以状态 `137` 终止。Linux OOM
记录显示被杀进程常驻内存约 `14.0 GiB` 且当时 swap 已耗尽，原因是实现曾将完整
C-P 成员矩阵稠密化；这不是 GPU 显存不足，也不是方法 Gate 结果。现已将训练正边
成员查询改为排序 `int64` 边键和二分查找，空间从
`O(|C||P|)` 降为 `O(|E_CP|)`，并增加大实体空间的稀疏内存回归测试。公式、配置、
seed 和 Gate 均未改变。已完成日志通过下面的断点续跑命令复用：

```bash
./run_schpt_gate1.sh --resume results/batch_runs/schpt_gate1_20260805_141202
```

修复后已从同一目录断点续跑并生成四库汇总。Gate 1 结果为 **PASS**：

| 数据集 | Baseline AUPR | SCHPT AUPR | Delta | Coverage | Scale |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.688849 | 0.705196 | +0.016347 | 0.9888 | 0.468535 |
| TCMSP | 0.957631 | 0.960557 | +0.002926 | 0.9998 | 0.479657 |
| SymMap2.0 | 0.801589 | 0.826556 | +0.024967 | 0.9868 | 0.552081 |
| ETCM2.0-mention10 | 0.906068 | 0.913596 | +0.007528 | 0.9942 | 1.671621 |

四库平均 Validation-AUPR 增量为 `+0.012942`，`4/4` 数据集为正，覆盖率和
非零 scale 条件全部通过。证据等级升级为 **B：四库单折预注册 Gate PASS**；
下一阶段允许构建冻结五折确认配置，但在五折完成前仍不作为最终主结果。

## 7. 四库五折确认

正式确认配置已经冻结，继续使用 compound cold-start、seed `52026`、相同 split
目录、inner-validation AUPR 早停、Dot、Hctx-P、SDIS、P-D PageRank 和
`attention.max.nodes=0`。与 Gate 1 相比只做两项协议升级：移除
`evaluation.fold.limit=1`，并设置 `evaluation.outer.test=True`。Baseline 与
SCHPT 每库除原型迁移开关及其替代的 compound PageRank 外完全一致。

运行命令：

```bash
./run_schpt_full.sh
```

中断后复用已完成任务：

```bash
./run_schpt_full.sh --resume results/batch_runs/schpt_full_<timestamp>
```

运行前冻结的五折 Gate 为：

1. 四库平均 outer AUPR 增量至少 `+0.003`；
2. 至少 `3/4` 数据集 outer AUPR 增量为正；
3. 任一数据集 outer AUPR 增量不得低于 `-0.005`；
4. 20 个配对 fold 中至少 12 个 AUPR 增量为正；
5. 每个候选 fold 的原型覆盖率至少 `30%`，learned scale 非零。

配置、SHA-256 和判据见 `configs/schpt_full_manifest.json`。五折 Gate 通过后
证据升级为 A；失败则冻结为 No-Go，不搜索 prior、seed、阈值或数据库专用配置。

四库五折确认已完成并 **PASS**：

| 数据集 | Baseline AUPR | SCHPT AUPR | Delta | 正向 folds |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.717960 | 0.721718 | +0.003758 | 2/5 |
| TCMSP | 0.939557 | 0.957323 | +0.017766 | 5/5 |
| SymMap2.0 | 0.807284 | 0.837607 | +0.030323 | 5/5 |
| ETCM2.0-mention10 | 0.901777 | 0.913655 | +0.011878 | 5/5 |

四库平均 outer AUPR 增量为 `+0.015931`，`4/4` 数据集和 `17/20` 配对 folds
为正；全部 fold 的覆盖率和 scale 条件通过。SCHPT 因而升级为 **A：四库完整
五折、匹配配置、冻结判据通过**，可以作为第三个算法贡献进入最终方法与消融表。
边界是 TCM-Suite 仅 `2/5` folds 为正，且最差一折下降 `-0.005690`；因此只主张
跨数据库总体有效，不主张所有划分均单调提升。完整结果位于
`results/batch_runs/schpt_full_20260805_152323/summary.md`。
