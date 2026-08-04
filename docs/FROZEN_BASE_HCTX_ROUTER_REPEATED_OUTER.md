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

## 6. 结果解释边界

若 Gate 通过，可以主张 V3 对 CW 状态的改善能够跨数据库、跨支持划分重复，
且不会损害冻结的 WC/CC 状态。仍不能声称解决 target-cold 或 double-cold，
也不能把四个相关 unit 当作完全独立的生物学队列。

若 Gate 失败，应报告失败数据库和状态，不允许依据结果删除 unit、修改 head
超参数或重新定义 Gate。
