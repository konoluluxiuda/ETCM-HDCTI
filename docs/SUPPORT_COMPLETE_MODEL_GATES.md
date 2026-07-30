# 支持状态完备模型 Gate

## 1. 目的

协议层已经完成，模型模块必须在预先指定的支持状态下做同单元配对，避免把多个
冷启动问题混在一次实验中。

第一阶段仅使用：

```text
dataset: TCM-Suite
outer unit: target fold 0 / double C0-P0
random.seed: 2026
validation.seed: 102026
validation.ratio: 0.1
metric: inner-validation AUPR
outer test: disabled
dense full attention: disabled
```

## 2. Gate 1A：Target-cold C-Dctx

对照：

```text
configs/HDCTI_tcmsuite_target_cold_no_context_early_stop_unit_pilot.conf
```

候选：

```text
configs/HDCTI_tcmsuite_target_cold_cdctx_early_stop_unit_pilot.conf
```

唯一模型变量：

```text
context.interaction: False -> True
context.compound_disease: False -> True
```

其余训练、划分、seed、早停和模型参数完全相同。

NoContext 冻结参考：

```text
best validation AUPR: 0.829882
best epoch: 12
stopped epoch: 22
```

预先判定：

- `delta AUPR >= +0.005`：通过低成本 Gate，进入四库 target-cold 单元确认；
- `0 < delta AUPR < +0.005`：边界结果，只允许再检查一个预注册 seed；
- `delta AUPR <= 0`：停止该项，不调整验证划分或读取外层测试挽救结果；
- 任意 NaN、已知正边误采负例或支持状态泄漏：协议失败，不解释性能。

该 Gate 只能判断 `C-Dctx` 是否值得继续，不能单独支持论文创新结论。

实际结果（2026-07-30，CPU）：

```text
NoContext best validation AUPR: 0.829882 (epoch 12)
C-Dctx best validation AUPR:    0.606159 (epoch 50)
delta AUPR:                    -0.223723
C-Dctx weight mean abs:         2.817455
```

判定：**未通过**。该项学到了幅度较大的上下文权重，但排序性能显著下降，说明
当前 C-Dctx 形式在 Target-cold 单元中引入了错误的疾病上下文偏置，而不是提供
有效的冷靶点迁移信号。按预先判定规则停止该项，不追加 seed、不读取外层测试，
也不根据该结果调整验证划分或超参数。

## 3. Gate 1B：Double-cold Hctx-Dctx

在 Gate 1A 完成后执行。对照使用：

```text
configs/HDCTI_tcmsuite_double_cold_no_context_early_stop_unit_pilot.conf
```

冻结参考：

```text
best validation AUPR: 0.553659
best epoch: 12
stopped epoch: 22
```

候选配置只启用 `context.herb_disease=True`，其余项与对照一致。判定阈值沿用
Gate 1A。该配置在 Gate 1A 结果记录前不启动，避免同时试多个模块后选择性报告。

候选：

```text
configs/HDCTI_tcmsuite_double_cold_hctx_dctx_early_stop_unit_pilot.conf
```

实际结果（2026-07-30，CPU）：

```text
NoContext best validation AUPR: 0.553659 (epoch 12)
Hctx-Dctx best validation AUPR: 0.578218 (epoch 44)
delta AUPR:                      +0.024559
Hctx-Dctx weight mean abs:        2.233022
```

判定：**通过低成本 Gate**。该结果证明 Hctx-Dctx 值得进入下一阶段，但证据仍
限于 TCM-Suite 的一个 Double-cold 单元，不能表述为四库稳定收益或最终贡献。

## 4. Gate 后决策

两个缺失分支没有同时通过：

| 支持状态 | 候选分支 | Delta AUPR | Gate |
|---|---|---:|---|
| warm compound / cold target | C-Dctx | -0.223723 | Stop |
| cold compound / cold target | Hctx-Dctx | +0.024559 | Go |

因此暂不实现“每个状态各有一个新增上下文项”的四状态路由，也不把 C-Dctx
替换成临时调参版本。下一步先对两个已保存 checkpoint 做内层验证纯推理分解：

```text
base-only
context-only
base + context
```

目的：

1. 判断 C-Dctx 的失败是上下文项本身排序反向，还是与 base 叠加冲突；
2. 判断 Hctx-Dctx 的正增益能否在关闭不可靠 ID base 后保留；
3. 只有 Hctx-Dctx context-only 仍优于对应 NoContext 或达到预注册可用门槛时，
   才设计由训练 C-P 支持度决定的统一推理路由。

该分解不重新训练、不评价外层测试，也不修改现有 checkpoint。
