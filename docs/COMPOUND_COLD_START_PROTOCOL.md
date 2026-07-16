# 成分 C-P 冷启动评估协议

## 1. 目的

随机 pair 五折中，同一成分通常同时出现在训练集和测试集。现有排名审计显示，模型错误主要集中在训练 C-P degree 为 0 或 1 的成分，而随机折整体已接近饱和。因此新增 `compound_cold_start` 协议，用于检验模型仅依靠 H-C 侧信息能否为没有训练 C-P 标签的成分预测靶点。

该协议不会替代原有 `pair_stratified` 主表。二者回答的问题不同：

| 协议 | 测试问题 |
|---|---|
| `pair_stratified` | 已知成分和靶点之间缺失边的补全能力 |
| `compound_cold_start` | 没有训练 C-P 标签的成分能否依靠 H-C 上下文完成预测 |

## 2. 外层五折

配置：

```ini
experiment.protocol=strict
split.strategy=compound_cold_start
split.seed=2026
evaluation.setup=-cv 5
```

划分规则：

1. 以 compound ID 为分组单位，同一成分的全部 C-P 正样本进入同一测试 fold。
2. 为每个成分固定采样与其正样本数相同的未观测 C-P pair，正负样本随成分一起进入同一 fold。
3. 优先使用数据集已有 ZERO 文件中的同成分候选；不足部分从该数据集已知 protein 全集中确定性补采。
4. 按每个成分的正边数进行确定性贪心负载均衡，使五折正样本规模尽量接近。
5. 每个外层 fold 均检查训练 compound 与测试 compound 交集为 0，且每个测试 compound 的正负样本数相等。

划分输出：

```text
dataset/ETCM2.0_core_mention10/splits/
└── strict_compound_cold_start_seed_2026_k5/
    ├── manifest.json
    ├── fold_assignments.tsv
    └── test_fold_0.txt ... test_fold_4.txt
```

manifest 记录 `split_strategy`、划分算法、源文件哈希、负样本来源与补采数量、每折成分数和 assignments 哈希。旧版未包含 `split_strategy` 的 version 3 manifest 只按 `pair_stratified` 解释，不会被冷启动配置误用。

## 3. 内层验证与早停

启用早停时，外层训练部分不能再进行随机 pair 划分。当前实现会自动使用同一个 `split.strategy`：

```text
outer-train compounds
├── inner-train compounds
└── validation compounds
```

inner-train 和 validation 的 compound 交集为 0。验证 compound 的全部 C-P pair 被整体保留，用于选择最佳 validation AUPR；validation 与外层 test 标签都不会进入训练 C-P PageRank、CHCR 或其他标签依赖统计。

## 4. 协议边界

这里的冷启动是“C-P 标签冷启动”，不是完全未知实体的归纳学习：

* 测试 compound 不具有任何训练 C-P 边；
* 测试 compound 仍存在于固定 H-C 侧图中，并通过药材上下文获得表示；
* P-D 仍作为固定、与测试 C-P 标签无关的侧信息；
* H-D 默认关闭；
* 因此结果应表述为 `compound C-P cold-start` 或“基于 H-C 侧信息的成分冷启动”，不能表述为完全无侧信息的新实体预测。

## 5. Pilot 对照

先只运行首折，并保持数据、seed、早停和 Dot decoder 完全一致：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_no_context_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_herb_only_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_chcr_pilot.conf
```

比较顺序：

1. `HerbOnly - NoContext`：判断静态 Hctx-P 是否真正改善无 C-P 标签成分。
2. `CHCR - HerbOnly`：判断反事实约束是否在冷启动场景提供额外增益。
3. 以 AUPR 为主指标，同时报告 AUC、Recall、Precision 和 F1。

首折只用于 Go/No-Go。只有至少一个创新模型相对 NoContext 明确提升，才运行完整五折；若三者差异很小或创新模型下降，则停止该分支，不做 seed 扩展。

## 6. 代码验证

```bash
/home/zry/.conda/envs/HDCTI_tfnew/bin/python -m unittest tests.test_strict_protocol -v
```

测试覆盖：确定性复用、成分互斥、逐成分正负匹配、ZERO 不足时的笛卡尔补采、manifest 策略冲突拒绝，以及成分互斥的内层验证。

## 7. ETCM2.0 mention10 实际划分审查

2026-07-16 已生成 seed 2026 的五折 manifest，但尚未运行 GPU 训练：

| 项目 | 数值 |
|---|---:|
| compound 总数 | 9,519 |
| H-C 支撑 compound | 9,519（100%） |
| 正样本 | 88,431 |
| 固定未观测样本 | 88,431 |
| ZERO 不足的 compound | 3,817 |
| 确定性补采 pair | 27,212 |
| fold 1 测试 compound | 1,904 |
| fold 1 测试正例 / 未观测例 | 17,687 / 17,687 |
| fold 1 外层 train/test compound 交集 | 0 |
| fold 1 inner-train / validation compound | 6,853 / 762 |
| fold 1 inner-train / validation records | 127,186 / 14,302 |
| assignments SHA-256 | `a4bfe2d50cedb0b9d4fd3e83b0b33c9b56a20c9fd44a395bdca4472e66338c56` |

其余四折测试正例分别为 17,686，测试 compound 为 1,903--1,904。所有 fold 都保持逐成分 1:1 正负匹配。
