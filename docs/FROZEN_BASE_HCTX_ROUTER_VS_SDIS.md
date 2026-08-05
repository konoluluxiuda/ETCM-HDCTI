# 冻结上下文专家与 SDIS 的匹配比较

## 目的

V3 已相对匹配 NoContext 在四库 16 个新 outer units 上通过确认性 Gate，但该
结果尚未量化其相对既有 `Hctx-P + SDIS` 的独立增量。本阶段只回答：在相同
`c1p1-c4p4` 支持单元上，冻结 base 与隔离 head 是否优于联合训练的 SDIS。

## 冻结比较

两者使用完全相同的训练 pair、inner-validation、outer records、support
assignment、seed、50 epoch 上限、早停、Dot decoder 和
`attention.max.nodes=0`。

```text
Comparator: jointly trained Hctx-P + SDIS
Candidate:  frozen NoContext base + isolated Hctx-P head (V3)
```

SDIS 的 checkpoint 只根据 inner-validation 四状态 Macro-AUPR 选择。16 个
SDIS checkpoint 全部训练并冻结哈希后，才允许读取 untouched outer records。
V3 直接复用已冻结的确认性 outer 报告，不重新训练或选择参数。

## 决策门槛

预注册文件为：

```text
configs/frozen_base_hctx_router_vs_sdis_plan.json
```

同时报告两级结论：

1. 非劣：总体 V3-SDIS Macro-AUPR 不低于 `-0.005`，每库不低于 `-0.01`，
   且任一数据库状态均值不下降超过 `0.02`；
2. 优效：总体增量至少 `+0.005`，至少 `12/16` 单元为正，且每库均值不低于 0。

优效通过时 V3 才替代 SDIS 成为最终支持状态方法；仅非劣通过时只能强调冻结
保护性质；非劣失败时保留 SDIS，V3 降级为支持状态分析。

## 运行

```bash
python tools/prepare_frozen_base_hctx_router_vs_sdis.py
./run_frozen_base_hctx_router_vs_sdis.sh --dry-run
./run_frozen_base_hctx_router_vs_sdis.sh --device gpu
```

中断恢复：

```bash
./run_frozen_base_hctx_router_vs_sdis.sh \
  --run-dir results/batch_runs/<run-dir> --stage train --device gpu
./run_frozen_base_hctx_router_vs_sdis.sh \
  --run-dir results/batch_runs/<run-dir> --stage outer --device gpu
```

## 冻结结果

正式匹配比较已于 2026-08-04 完成：

```text
results/batch_runs/frozen_base_router_vs_sdis_20260804_171145
```

结果文件 `summary.json` 的 SHA-256 为
`cebd6d2437f274b3e181c36da40e2e0d9cbcc8ad00db245c71ccbbccf86710ad`。
16 个 SDIS checkpoint 在读取 outer records 前全部冻结；训练 manifest 的
SHA-256 为
`3d2f6fd4faf9d4e2ae9e98d563710f8f79d85dc870ae79028f34bfdd3cba3db8`。

| Dataset | SDIS Macro-AUPR | V3 Macro-AUPR | V3-SDIS | Positive units |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.601000 | 0.614290 | +0.013291 | 3/4 |
| TCMSP | 0.668938 | 0.692147 | +0.023209 | 3/4 |
| SymMap2.0 | 0.655420 | 0.652725 | -0.002696 | 1/4 |
| ETCM2.0-mention10 | 0.674064 | 0.686551 | +0.012486 | 3/4 |
| Overall | - | - | +0.011573 | 10/16 |

V3 的 CW 状态在 16/16 units 上均优于 SDIS，各库平均增量为
`+0.024842/+0.186545/+0.120828/+0.093898`。但 V3 的 WW 在多数单元下降，
CC 在 SymMap2.0 和 TCMSP 分别平均下降 `-0.097234/-0.038822`；ETCM 的 WC
也平均下降 `-0.031503`。因此：

```text
Noninferiority: FAIL
Superiority: FAIL
Frozen decision: retain Hctx-P + SDIS
```

总体 Macro-AUPR 提高不能覆盖预注册的状态退化。V3 不替代 SDIS，也不进入最终
`Ours-full`；它只保留为“冻结基座的冷成分—暖靶点专家”机制分析。不得根据该
outer 结果继续修改路由状态、损失权重或数据库特定阈值。
