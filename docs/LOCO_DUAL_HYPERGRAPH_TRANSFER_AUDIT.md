# LOCO 双超图原型迁移可行性审计

- 审计类型：四库冻结 support-complete 单元内的 inner-validation。
- 模型训练、checkpoint 恢复和外层测试读取次数均为 `0`。
- 统计量仅由 inner-training 正例、H-C 和 P-D 构造。
- compound/protein 上下文相似矩阵均删除对角线，禁止自身标签回流。
- 固定通道：`H→P`、`D→C`、双侧迁移及三者等权融合；未搜索权重。

## 预注册判定

**NO-GO**

| 检查 | 结果 |
|---|---|
| `cross_dataset_expected_macro` | `True` |
| `passing_dataset_count` | `True` |
| `passing_cold_state_cell_count` | `True` |
| `expected_channel_coverage` | `False` |

跨库 expected-channel Macro-AUPR：`0.738732`；通过数据集：`4/4`；通过 cold-state 单元：`11/12`。

## Expected-channel 结果

| 数据集 | WW Fusion | CW H→P | WC D→C | CC Dual | Macro |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.804398 | 0.675779 | 0.736603 | 0.631803 | 0.712146 |
| TCMSP | 0.919468 | 0.919598 | 0.506194 | 0.510663 | 0.713981 |
| SymMap2.0 | 0.922791 | 0.821828 | 0.874657 | 0.670607 | 0.822471 |
| ETCM2.0-mention10 | 0.901506 | 0.870656 | 0.559163 | 0.493988 | 0.706328 |

## 覆盖率与方向

| 数据集 | 状态 | 通道 | 非零覆盖率 | 正负均值差 |
|---|---|---|---:|---:|
| TCM-Suite | warm_warm | fixed_fusion | 0.555733 | +0.003889 |
| TCM-Suite | cold_warm | herb_to_target | 0.274465 | +0.002179 |
| TCM-Suite | warm_cold | disease_to_compound | 0.339892 | +0.009182 |
| TCM-Suite | cold_cold | dual_transfer | 0.342311 | +0.001504 |
| TCMSP | warm_warm | fixed_fusion | 0.556011 | +0.010773 |
| TCMSP | cold_warm | herb_to_target | 0.549898 | +0.034666 |
| TCMSP | warm_cold | disease_to_compound | 0.006404 | +0.000270 |
| TCMSP | cold_cold | dual_transfer | 0.018409 | +0.000126 |
| SymMap2.0 | warm_warm | fixed_fusion | 0.988608 | +0.005667 |
| SymMap2.0 | cold_warm | herb_to_target | 0.876134 | +0.004982 |
| SymMap2.0 | warm_cold | disease_to_compound | 0.780131 | +0.012274 |
| SymMap2.0 | cold_cold | dual_transfer | 0.904980 | +0.000262 |
| ETCM2.0-mention10 | warm_warm | fixed_fusion | 0.997296 | +0.014794 |
| ETCM2.0-mention10 | cold_warm | herb_to_target | 0.487640 | +0.042613 |
| ETCM2.0-mention10 | warm_cold | disease_to_compound | 0.941124 | +0.000045 |
| ETCM2.0-mention10 | cold_cold | dual_transfer | 0.992347 | -0.000054 |

## 解释边界

- `PASS` 只表示存在值得进入 validation-only Pilot 的统计信号，不表示新模块已经成为论文创新。
- `NO-GO` 时不得通过查看外层测试、搜索融合权重、Top-K、温度或数据库专用规则挽救该候选。
- 即使通过，下一阶段也必须比较 `保留 PageRank`、`删除 compound PageRank` 和 `删除 compound PageRank + LOCO-DHPT`，才能证明替换成立。

