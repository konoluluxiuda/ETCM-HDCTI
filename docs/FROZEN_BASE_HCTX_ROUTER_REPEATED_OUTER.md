# 冻结上下文专家重复 Outer 复验

## 1. 目的

V3 已在四库 `c0p0` 独立 outer unit 上全部通过，但该结果仍属于四库单 unit
证据。本阶段不再修改模型，而是检验同一冻结方法能否在新的实体支持划分中重复。

目标是将“支持状态感知的冻结上下文专家”从 B 级候选证据提升为四库多 unit
证据，而不是继续搜索结构、超参数或数据库特定路由。

## 2. 确认性单元

历史 `c0p0` 只进入最终五单元描述性汇总。新的确认性 Gate 仅使用此前未运行的：

```text
c1p1
c2p2
c3p3
c4p4
```

四个数据库合计 16 个新 outer units。每个单元都必须执行：

1. 使用该单元 outer-training 数据训练 NoContext base；
2. 只在该单元 inner-training/validation 数据训练独立 Hctx-P linear head；
3. 在查看 outer 指标前冻结 base、head、最佳 epoch、路由和文件哈希；
4. 对 outer 四种状态进行零训练纯推理评价。

## 3. 固定路由

```text
WW = frozen base + frozen Hctx-P head
CW = frozen Hctx-P head
WC = frozen base
CC = frozen base
```

WC/CC 必须与匹配 NoContext base 精确一致。该约束用于证明方法不会为了提升
compound-cold 状态而破坏其没有适用上下文专家的状态。

## 4. 预注册主终点与 Gate

主终点为每个 outer unit 相对匹配 NoContext base 的 Macro-AUPR 差值。

确认性 Gate 在读取 16 个新 outer 结果前固定为：

1. 16 个新单元总体平均 Macro-AUPR 差值不低于 `+0.005`；
2. 至少 12/16 个新单元 Macro-AUPR 差值为正；
3. 每个数据库四个新单元的平均 Macro-AUPR 差值不为负；
4. 任一数据库的状态平均 AUPR 下降不得超过 `0.020`；
5. WC/CC 必须精确保留；
6. outer unit 禁止训练、早停、阈值选择和参数选择。

新单元是主要确认性分析。完成后另报告包含历史 `c0p0` 的五单元均值、标准差
和逐单元差值，但不得用历史单元改变上述 Gate。

## 5. 冻结配置

预注册文件：

```text
configs/frozen_base_hctx_router_repeated_outer_plan.json
```

其中固定了数据源哈希、四个新单元、base/head 训练设置、路由和 Gate。

准备命令：

```bash
python tools/prepare_frozen_base_hctx_router_repeated_outer.py --dry-run
python tools/prepare_frozen_base_hctx_router_repeated_outer.py
```

准备阶段只生成四状态数据工件、NoContext 配置和对应哈希清单，不启动训练，
也不读取任何新 outer 指标。

## 6. 执行与恢复

先检查 16 个单元和固定协议，不启动训练：

```bash
./run_frozen_base_hctx_router_repeated_outer.sh --dry-run
```

默认使用 CPU 完成三阶段确认实验：

```bash
./run_frozen_base_hctx_router_repeated_outer.sh
```

如需使用 GPU，必须显式指定：

```bash
./run_frozen_base_hctx_router_repeated_outer.sh --device gpu
```

执行器将工件写入
`results/batch_runs/frozen_base_hctx_router_repeated_outer_<timestamp>/`，并按以下
顺序运行：

```text
base：训练与评价 16 个匹配 NoContext base
head：冻结 base，训练 16 个 Hctx-P head，并一次性冻结全部 head 哈希
outer：对 16 个新 outer units 做零训练纯推理并执行预注册 Gate
```

长任务中断后，可使用同一运行目录分阶段恢复：

```bash
./run_frozen_base_hctx_router_repeated_outer.sh \
  --run-dir results/batch_runs/<run-dir> --stage base
./run_frozen_base_hctx_router_repeated_outer.sh \
  --run-dir results/batch_runs/<run-dir> --stage head
./run_frozen_base_hctx_router_repeated_outer.sh \
  --run-dir results/batch_runs/<run-dir> --stage outer
```

恢复时会重新核验配置、支持划分、checkpoint、报告和 head 哈希。内层 Gate 仅作
诊断，不能据此删除失败单元；只有全部 16 个 head 都被冻结后，才允许进入 outer
阶段。

## 7. 结果解释边界

确认实验已于 2026-08-04 完成，结果目录为：

```text
results/batch_runs/frozen_base_hctx_router_repeated_outer_20260804_132807
```

16 个新 outer units 全部获得正 Macro-AUPR 增量，预注册 Gate 全部通过：

| 数据集 | Units | NoContext Macro-AUPR | V3 Macro-AUPR | 差值 | WW | CW | WC | CC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 4 | 0.565482 (±0.003834) | 0.614290 (±0.004159) | +0.048809 (±0.005374) | +0.018187 | +0.177048 | +0.000000 | +0.000000 |
| TCMSP | 4 | 0.554194 (±0.010327) | 0.692147 (±0.005126) | +0.137953 (±0.010952) | +0.007010 | +0.544804 | +0.000000 | +0.000000 |
| SymMap2.0 | 4 | 0.550975 (±0.005555) | 0.652725 (±0.008770) | +0.101750 (±0.004007) | +0.022779 | +0.384220 | +0.000000 | +0.000000 |
| ETCM2.0-mention10 | 4 | 0.553521 (±0.011886) | 0.686551 (±0.017287) | +0.133030 (±0.007360) | +0.008994 | +0.523126 | +0.000000 | +0.000000 |

总体平均 Macro-AUPR 增量为 `+0.105385`，正向单元为 `16/16`。收益集中于
方法预先指定负责的 CW 状态，WW 为小幅正增益；WC 和 CC 与匹配 NoContext
base 精确一致。全部 head 均在读取新 outer 指标前冻结，outer 阶段优化步数为
`0`，未执行阈值、epoch、路由或参数选择。

关键汇总哈希：

```text
summary.json  87442a0596690ced09523fa7004dbd1d05802af921c0e1d343b6b189b93dfc09
summary.md    30b8e2d87f41b3781a75fb1e40827a7a7198590a0d28adfb3b75d46909d77257
16 reports   b8b966cc1e54123f40d8c733db1915956243bfebfb04e6b5a9fe374f52bb014e
```

据此，可以主张 V3 对 CW 状态的改善能够跨数据库、跨支持划分重复，且不会
损害冻结的 WC/CC 状态。该结果将 V3 从四库单 unit 的 B 级候选证据提升为
四库多 unit 确认性证据。仍不能声称解决 target-cold 或 double-cold，也不能
把同一数据库产生的四个相关 unit 当作完全独立的生物学队列。

预注册约束在结果产生后继续有效：不得依据结果删除 unit、修改 head 超参数或
重新定义 Gate；后续只能进行描述性汇总和论文材料整理。

历史 `c0p0` 与新 `c1p1-c4p4` 的五单元描述性汇总使用：

```bash
python tools/summarize_frozen_base_hctx_router_five_units.py \
  --historical-summary \
  results/batch_runs/frozen_base_hctx_router_outer_20260804_124039/summary.json \
  --repeated-summary \
  results/batch_runs/frozen_base_hctx_router_repeated_outer_20260804_132807/summary.json \
  --output-dir \
  results/batch_runs/frozen_base_hctx_router_repeated_outer_20260804_132807
```

该汇总不得把历史 `c0p0` 重新纳入确认性 Gate。

五单元描述性汇总结果为：

| 数据集 | Units | NoContext Macro-AUPR | V3 Macro-AUPR | 差值 | Positive |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 5 | 0.566756 (±0.004272) | 0.614979 (±0.003967) | +0.048223 (±0.004947) | 5/5 |
| TCMSP | 5 | 0.561596 (±0.017449) | 0.695667 (±0.008400) | +0.134071 (±0.012500) | 5/5 |
| SymMap2.0 | 5 | 0.552728 (±0.006081) | 0.653935 (±0.008209) | +0.101206 (±0.003745) | 5/5 |
| ETCM2.0-mention10 | 5 | 0.553840 (±0.010650) | 0.686311 (±0.015470) | +0.132470 (±0.006678) | 5/5 |

20 个描述性单元全部为正，总体 Macro-AUPR 增量为
`+0.103993 (±0.035613)`。五单元汇总文件哈希为：

```text
five_unit_summary.json  8d76d7f237d21bf8199b8b5012dc3d161019a9bab5a3ae2b725f0bf2592327a1
five_unit_summary.md    fab6ec8259588d122c0cb48cb277ffeb7318318b853cc10f63152e950969e156
```
