# 四状态支持感知统一路由

## 1. 研究问题

随机边划分主要评估训练中已经出现过的成分和靶点，而冷启动划分会产生
训练 C-P 支持度为零的实体。统一模型不能继续让同一个评分分支在四种支持
状态下无条件工作：

| 状态 | 成分训练 C-P 支持 | 靶点训练 C-P 支持 |
|---|---:|---:|
| WW | 有 | 有 |
| CW | 无 | 有 |
| WC | 有 | 无 |
| CC | 无 | 无 |

支持状态完全由当前训练 C-P 图决定，不由数据集名称、测试结果或人工选择
配置决定。

## 2. 确定性路由

设基础成分—靶点分数为 `s_base`，药材上下文—靶点分数为 `s_HP`，
药材上下文—疾病上下文分数为 `s_HD`。统一 checkpoint 使用以下固定路由：

| 状态 | 最终 logit |
|---|---|
| WW | `s_base + s_HP` |
| CW | `s_HP` |
| WC | `s_base` |
| CC | `s_HD` |

额外要求：

- `s_HP` 仅在成分具有 H-C 侧信息时启用；
- `s_HD` 仅在成分具有 H-C 且靶点具有 P-D 侧信息时启用；
- 不训练可根据测试结果改变的软门控；
- `attention.max.nodes=0`，避免与原稠密全节点注意力混合。

## 3. 分支隔离训练

共享四状态协议的训练 pair 全部属于 WW，因此 CC 分支不能从主 BCE 获得
梯度。V1 让 `base + Hctx-P` 联合更新编码器，虽然提高 Macro-AUPR，却损害
只依赖 base 的 WC 状态。V2 将三个分支解耦：

```text
L_base = BCE(<C(c), P(p)>, y_cp)
L_HP = BCE(<stop_gradient(Hctx(c) * P(p)), w_HP>, y_cp)
L_HD = BCE(<stop_gradient(Hctx(c) * Dctx(p)), w_HD>, y_cp)
```

两个侧信息损失分别只更新 `w_HP` 和 `w_HD`：

- 不更新 H-C/P-D 编码器；
- base 编码器梯度与 NoContext 对照保持一致；
- 不使用 CW、WC 或 CC 验证标签；
- 推理时仍由一个 checkpoint 按四状态组合三个分支。

因此它是训练期的隔离侧信息头，不是将测试冷启动标签偷偷加入训练。

## 4. 配对实验

控制配置：

```text
configs/HDCTI_tcmsuite_four_state_no_context_unit_pilot.conf
```

V1 联合训练配置：

```text
configs/HDCTI_tcmsuite_four_state_support_routing_unit_pilot.conf
```

V2 隔离训练候选：

```text
configs/HDCTI_tcmsuite_four_state_isolated_routing_unit_pilot.conf
```

两者共享：

- 同一个四状态 manifest；
- seed `2026`；
- 相同训练 pair、验证 pair、优化器、batch size 和 50 epoch 上限；
- 相同早停规则；
- 相同 `attention.max.nodes=0`；
- 外层测试关闭。

V2 相对控制的唯一方法差异是启用两个隔离侧信息头和确定性四状态路由。

## 5. 预注册 Gate

本轮只使用 TCM-Suite 判断 V2 是否值得扩展到四库。判定标准保持为：

1. 四状态等权 Macro-AUPR 相对 NoContext 提升至少 `0.005`；
2. CC AUPR 不低于 NoContext；
3. 任一状态 AUPR 不得下降超过 `0.020`；
4. 训练日志确认主训练 pair 全部为 WW；
5. 冷冷辅助特征保持停止梯度。

若未通过，不调辅助损失权重追逐该验证集，记录失败并停止扩展。若通过，
再冻结配置并做四库验证。

## 6. 冒烟结果

2 epoch CPU 冒烟测试已通过：

| 模型 | WW | CW | WC | CC | Macro-AUPR |
|---|---:|---:|---:|---:|---:|
| NoContext | 0.539463 | 0.414484 | 0.811799 | 0.432002 | 0.549437 |
| 四状态路由 | 0.519978 | 0.520063 | 0.824344 | 0.624447 | 0.622208 |

该结果只证明训练、路由、四状态验证和 checkpoint 恢复能够贯通，不能替代
50 epoch 配对 Gate。

## 7. V1 正式 Gate 结果

V1 与 NoContext 均在 epoch 18 达到最佳 Macro-AUPR：

| 模型 | WW | CW | WC | CC | Macro-AUPR |
|---|---:|---:|---:|---:|---:|
| NoContext | 0.730593 | 0.422264 | 0.820482 | 0.481914 | 0.613813 |
| V1 联合路由 | 0.812647 | 0.625389 | 0.760289 | 0.606210 | 0.701134 |
| 差值 | +0.082054 | +0.203125 | -0.060193 | +0.124296 | +0.087320 |

V1 因 WC 下降超过 `0.020` 而未通过预注册 Gate。该现象与联合 Hctx-P
梯度扰动 base 编码器一致，因此进入结构隔离的 V2；这不是调整阈值或损失
权重后的结果追逐。

## 8. V2 正式 Gate 结果

V2 同样在 epoch 18 达到最佳 Macro-AUPR。结果通过保存 checkpoint 的
纯推理脚本重新计算，未执行训练或优化器步骤：

| 模型 | WW | CW | WC | CC | Macro-AUPR |
|---|---:|---:|---:|---:|---:|
| NoContext | 0.730593 | 0.422264 | 0.820482 | 0.481914 | 0.613813 |
| V2 隔离路由 | 0.754114 | 0.607243 | 0.820482 | 0.660896 | 0.710684 |
| 差值 | +0.023521 | +0.184979 | +0.000000 | +0.178981 | +0.096870 |

预注册 Gate 全部通过：

- Macro-AUPR 提升不低于 `0.005`：通过；
- CC 不低于 NoContext：通过；
- 任一状态下降不超过 `0.020`：通过；
- WC 与 NoContext 精确一致，支持 base 梯度隔离成立；
- 训练状态计数为 WW `38492`，其余三类均为 `0`；
- 单元测试确认两个侧信息头均不向编码器回传梯度。

当前结论仅限 TCM-Suite 的固定四状态单元。下一步应冻结 V2，不再在该单元
调参，并扩展到 TCMSP、SymMap2.0 和 ETCM2.0_core_mention10。

纯推理报告位于：

```text
results/four_state_routing_gate/tcmsuite_no_context/report.md
results/four_state_routing_gate/tcmsuite_isolated_routing/report.md
```

## 9. 验证与审计文件

```text
tests/test_model_components.py
tests/test_support_state_routing.py
configs/HDCTI_tcmsuite_four_state_support_routing_unit_smoke.conf
configs/HDCTI_tcmsuite_four_state_isolated_routing_unit_smoke.conf
tools/evaluate_four_state_checkpoint.py
```

训练完成后，每个 checkpoint 目录写出：

```text
support_state_routing.json
```

其中记录固定路由、训练状态计数、验证状态计数、辅助损失设置和两个上下文
交互头的参数统计。

## 10. 三库冻结扩展 Gate

TCMSP、SymMap2.0 和 ETCM2.0_core_mention10 已建立与 TCM-Suite 完全
同构的 NoContext/V2 配对配置。除数据路径和 `model.variant` 外，随机种子、
四状态构造、内层验证、早停、负样本、编码维度、优化参数和 Gate 阈值均保持
冻结。

配置文件：

```text
configs/HDCTI_tcmsp_four_state_no_context_unit_pilot.conf
configs/HDCTI_tcmsp_four_state_isolated_routing_unit_pilot.conf
configs/HDCTI_symmap_four_state_no_context_unit_pilot.conf
configs/HDCTI_symmap_four_state_isolated_routing_unit_pilot.conf
configs/HDCTI_etcm_mention10_four_state_no_context_unit_pilot.conf
configs/HDCTI_etcm_mention10_four_state_isolated_routing_unit_pilot.conf
```

运行入口：

```bash
./run_four_state_routing_cross_dataset_gate.sh --dry-run
./run_four_state_routing_cross_dataset_gate.sh
```

批处理会为每个模型自动执行以下步骤：

1. 训练并恢复最佳 Macro-AUPR checkpoint；
2. 从日志提取 checkpoint 路径；
3. 使用 `tools/evaluate_four_state_checkpoint.py` 进行纯推理复核；
4. 计算 V2 相对同库 NoContext 的 WW/CW/WC/CC 和 Macro-AUPR 差值；
5. 使用 `tools/summarize_four_state_routing_gate.py` 与已有 TCM-Suite
   报告合并。

四库总判定采用失败关闭原则：四个数据集必须分别通过既定 Gate，才允许将
V2 表述为跨数据库稳定的统一支持状态方法。任一数据集失败时，不调整该库
专属权重或阈值；应保留失败结果，并重新评估该模块的论文定位。

本节执行已完成，结果输出位于：

```text
results/batch_runs/four_state_routing_gate_20260730_164947/
```

## 11. 四库冻结 Gate 结果

四个数据集均使用冻结配置和固定四状态 assignment，checkpoint 通过纯推理
重新评价，评价过程的训练和优化器步数均为 `0`。四库结果如下：

| 数据集 | NoContext Macro-AUPR | V2 Macro-AUPR | 差值 | WW 差值 | CW 差值 | WC 差值 | CC 差值 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite | 0.613813 | 0.710684 | +0.096870 | +0.023521 | +0.184979 | +0.000000 | +0.178981 | PASS |
| TCMSP | 0.591945 | 0.698696 | +0.106750 | -0.014616 | +0.604066 | +0.015785 | -0.178234 | FAIL |
| SymMap2.0 | 0.614439 | 0.790405 | +0.175966 | +0.072927 | +0.490147 | -0.000094 | +0.140884 | PASS |
| ETCM2.0-mention10 | 0.592802 | 0.729850 | +0.137048 | +0.003917 | +0.556934 | +0.185249 | -0.197907 | FAIL |

四库平均 Macro-AUPR 差值为 `+0.129159`，但四库总 Gate 为 `FAIL`。
不能用宏平均增益掩盖状态级退化：

- CW 在四库均显著受益，是当前路由最稳定的有效部分；
- WC 基本保持或提升，说明隔离侧信息头避免了 V1 的 base 梯度污染；
- CC 在 TCM-Suite 和 SymMap2.0 提升，但在 TCMSP 和 ETCM2.0 分别下降
  `0.178234` 和 `0.197907`；
- 因此 V2 不能表述为跨数据库稳定的统一四状态方法，也不能直接作为论文
  最终模型。

该结论是预注册失败关闭规则的正常执行结果。后续不得降低 Gate 阈值、删除
失败数据集或为 TCMSP/ETCM2.0 设置专属路由权重来将本轮改写为成功。

## 12. CC 残差纯推理审计

为判断 CC 失败是否仅由“用 Hctx-Dctx 替换 base”造成，在四个已保存 V2
checkpoint 上执行了纯推理审计：

```text
s_CC = s_base + alpha * s_Hctx-Dctx
alpha in {0, 0.25, 0.5, 1.0}
```

该网格在审计前声明，不训练、不更新优化器，也不用于选择最终超参数。结果
如下：

| 数据集 | Hctx-Dctx only | Base only | Base + 0.25 H-D | Base + 0.5 H-D | Base + 1.0 H-D |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.660896 | 0.481914 | 0.685398 | 0.690325 | 0.690877 |
| TCMSP | 0.342292 | 0.498403 | 0.374614 | 0.340080 | 0.336033 |
| SymMap2.0 | 0.731840 | 0.566281 | 0.677255 | 0.710731 | 0.727279 |
| ETCM2.0-mention10 | 0.486532 | 0.444831 | 0.441292 | 0.439116 | 0.436738 |

表中数值为 CC AUPR。诊断表明：

1. TCM-Suite 和 SymMap2.0 的 Hctx-Dctx 信号可迁移到 CC；
2. TCMSP 中任何正向残差都会比候选 checkpoint 的 base-only 更差；
3. ETCM2.0 中候选 checkpoint 的 base-only 已低于 NoContext CC 基线，
   加入 Hctx-Dctx 后继续下降；
4. 因此“将替换改成残差并调 alpha”不能解决四库 Gate，当前方向判定为
   `No-Go`。

Hctx-Dctx 特征诊断进一步支持该结论：

| 数据集 | 训练非零率 | 训练特征 AUPR | CC 非零率 | CC 特征 AUPR |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.391 | 0.641 | 0.382 | 0.661 |
| TCMSP | 0.426 | 0.798 | 0.504 | 0.342 |
| SymMap2.0 | 0.916 | 0.719 | 0.919 | 0.732 |
| ETCM2.0-mention10 | 1.000 | 0.504 | 1.000 | 0.487 |

TCMSP 的训练特征 AUPR 较高，但 CC 下降到 `0.342`，属于明显的状态迁移
失配；ETCM2.0 的特征虽然全覆盖，但接近随机判别，说明“有上下文”不等于
“上下文可用于 CC 预测”。

完整审计报告位于：

```text
results/batch_runs/four_state_routing_gate_20260730_164947/cc_residual_audit/
```

## 13. 当前冻结结论与下一候选

本轮冻结以下结论：

- 保留 `Hctx-P`，因为其对应的 CW 改善在四库一致；
- 放弃将当前 `Hctx-Dctx` 作为 CC 主分支或正向残差；
- V2 四状态路由保留为失败消融，不进入最终主方法；
- 不再进行 alpha、阈值或数据集专属权重搜索。

下一候选应采用“两阶段冻结 base 的支持条件路由”：

```text
阶段 1：按 NoContext Macro-AUPR 训练并冻结 base encoder
阶段 2：仅训练隔离的 Hctx-P 侧信息头

WW: base + Hctx-P
CW: Hctx-P
WC: base
CC: base
```

该候选只保留已获得四库一致证据的 Hctx-P，并通过冻结 base 保证 WC/CC
不会因候选模型的早停 epoch 改变。它必须先在失败库 TCMSP 和
ETCM2.0-mention10 上做低成本 unit pilot；只有两库都不再发生状态级退化，
才值得扩展到四库完整 Gate。

## 14. V3 冻结 Base Pilot 实现

V3 已按上述边界实现为复合 checkpoint，而不是再次修改 `HDCTI.py`：

```text
NoContext TensorFlow checkpoint（全程只读）
              +
独立 Hctx-P 线性头（唯一可训练参数）
```

训练器恢复已经通过纯推理核验的 NoContext 最佳 checkpoint，从同一
four-state unit 的 inner-training WW pair 提取：

```text
x_cp = Hctx(c) * P(p)
s_HP = x_cp^T w_HP
```

只用 WW 标签优化 `w_HP`。base embedding、超图编码器、PageRank、原始
decoder 和 checkpoint 文件均不更新。内层早停仍使用固定四状态
Macro-AUPR，但 WC/CC 分数直接继承 NoContext 报告并执行精确相等断言。

冻结清单：

```text
configs/frozen_base_hctx_router_pilot.json
```

该清单预先固定：

- TCMSP 与 ETCM2.0-mention10 的 baseline config、checkpoint 和哈希；
- four-state assignment 哈希；
- head 的 seed、epoch、batch size、学习率、L2 和早停规则；
- 原有 Gate 阈值；
- `WC/CC` 必须与 NoContext 完全相同。

运行入口：

```bash
./run_frozen_base_hctx_router_pilot.sh --dry-run
./run_frozen_base_hctx_router_pilot.sh
```

训练与评价实现：

```text
tools/train_frozen_base_hctx_router.py
tools/summarize_frozen_base_hctx_router.py
```

每个报告必须同时满足：

1. base 模型优化步数为 `0`；
2. base checkpoint 所有文件训练前后哈希一致；
3. WC 与 CC 指标和 NoContext 报告精确一致；
4. Macro-AUPR 增量不低于 `0.005`；
5. 任一状态 AUPR 不下降超过 `0.020`；
6. 未使用 outer test。

本轮只运行两个先前失败的数据集。TCMSP 与 ETCM2.0-mention10 必须同时
通过，才能把 V3 扩展到 TCM-Suite 和 SymMap2.0；任一失败都不得通过调整
数据集专属学习率、路由或阈值进行补救。

## 15. V3 失败库 Pilot 结果与四库扩展

失败库 pilot 已于 2026-08-01 完成。原始冻结清单 SHA-256 为：

```text
620a99851cd1f9be43b2f049b9cec2e21d75cde5344dfd575f6c212e32327026
```

结果如下：

| 数据集 | NoContext Macro-AUPR | V3 Macro-AUPR | 差值 | WW 差值 | CW 差值 | WC 差值 | CC 差值 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TCMSP | 0.591945 | 0.744787 | +0.152842 | +0.003208 | +0.608158 | +0.000000 | +0.000000 | PASS |
| ETCM2.0-mention10 | 0.592802 | 0.728308 | +0.135506 | +0.001516 | +0.540508 | +0.000000 | +0.000000 | PASS |

两库的最佳 Hctx-P head 均出现在 epoch 6。验证同时确认：

- base 模型优化步数为 `0`；
- WC 与 CC 和各自 NoContext 报告完全一致；
- base checkpoint 的 data、index、meta 文件哈希训练前后不变；
- outer test 未使用。

因此 V3 通过预先规定的失败库晋级条件。该结果说明 V2 的主要问题不是
Hctx-P，而是重新训练 base 与不稳定 Hctx-Dctx 分支；冻结 base 后，CW 的
大幅收益得以保留，WC/CC 的回退被结构性消除。

但这仍是单 seed、单 four-state unit 的内层验证结果，不能据此宣称四库
泛化或最终论文有效。后续设置已经冻结到新清单：

```text
configs/frozen_base_hctx_router_four_dataset_gate.json
```

新清单完整继承 pilot 的 head 超参数，并通过父清单哈希记录来源，只新增
TCM-Suite 与 SymMap2.0。运行命令：

```bash
./run_frozen_base_hctx_router_four_dataset_gate.sh --dry-run
./run_frozen_base_hctx_router_four_dataset_gate.sh
```

只有四个数据集全部通过同一 Gate，才允许将 V3 冻结为“支持状态感知的
冻结上下文专家”候选创新点。四库通过后仍需在独立 outer unit 或固定五折上
进行最终评价，不能继续复用当前 inner-validation 数值作为主结果。

## 16. V3 四库 Inner Gate 结果

四库扩展于 2026-08-04 完成。冻结清单 SHA-256 为：

```text
3afdd445f809b7b009ddea83058e55989dbcf000799b89e34cd9df689741c2a6
```

四个数据集全部通过同一 Gate：

| 数据集 | NoContext Macro-AUPR | V3 Macro-AUPR | 差值 | WW 差值 | CW 差值 | WC 差值 | CC 差值 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite | 0.613813 | 0.680753 | +0.066940 | +0.016375 | +0.251383 | +0.000000 | +0.000000 | PASS |
| TCMSP | 0.591945 | 0.744787 | +0.152842 | +0.003208 | +0.608158 | +0.000000 | +0.000000 | PASS |
| SymMap2.0 | 0.614439 | 0.741066 | +0.126627 | +0.023289 | +0.483220 | +0.000000 | +0.000000 | PASS |
| ETCM2.0-mention10 | 0.592802 | 0.728308 | +0.135506 | +0.001516 | +0.540508 | +0.000000 | +0.000000 | PASS |

四库的 base 优化步数均为 `0`，WC/CC 精确继承 NoContext，checkpoint
哈希保持不变。最佳 head epoch 分别为 TCM-Suite `24`、TCMSP `6`、
SymMap2.0 `12` 和 ETCM2.0 `6`。

该结果支持以下受限结论：冻结 Hctx-P 专家在四库 inner unit 上都能改善
compound-cold/target-warm 状态，同时不会损害其余依赖 base 的状态。它不能
证明所有冷启动状态都得到改善，因为 WC/CC 是有意保留基线，而不是由新专家
解决。

## 17. 冻结 Outer Four-State 评价

为避免将用于 head 早停的 inner-validation 结果当作最终证据，四个 head
及其训练报告已经在查看 outer 指标前冻结。outer 清单记录：

- 四库 inner Gate 清单及 SHA-256；
- 四库 inner summary 及 SHA-256；
- 每个训练报告及 SHA-256；
- 每个 Hctx-P head 及 SHA-256；
- 禁止在 outer unit 上进行参数选择或训练。

清单与入口：

```text
configs/frozen_base_hctx_router_outer_evaluation.json
tools/evaluate_frozen_base_hctx_router_outer.py
run_frozen_base_hctx_router_outer_gate.sh
```

运行：

```bash
./run_frozen_base_hctx_router_outer_gate.sh --dry-run
./run_frozen_base_hctx_router_outer_gate.sh
```

该阶段直接评价 `HDR.supportTestDataByState`，与用于选择 head epoch 的
`supportValidationDataByState` 分离。全过程训练步数为 `0`，并再次验证 base
checkpoint 与 head 哈希不变。outer 结果一旦产生，不再依据结果修改 head、
路由或 Gate；若失败，应如实将 V3 降级为探索性结果。

## 18. V3 四库 Outer Gate 结果与冻结结论

独立 outer-unit 评价于 2026-08-04 完成。评价阶段未更新模型或 Hctx-P
head，未依据 outer 指标选择 epoch、阈值、路由或超参数。结果汇总文件为：

```text
results/batch_runs/frozen_base_hctx_router_outer_20260804_124039/summary.md
```

四个数据集全部通过预先冻结的 Gate：

| 数据集 | NoContext Macro-AUPR | V3 Macro-AUPR | 差值 | WW 差值 | CW 差值 | WC 差值 | CC 差值 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite | 0.571853 | 0.617732 | +0.045880 | +0.018818 | +0.164701 | +0.000000 | +0.000000 | PASS |
| TCMSP | 0.591204 | 0.709745 | +0.118541 | +0.005624 | +0.468538 | +0.000000 | +0.000000 | PASS |
| SymMap2.0 | 0.559741 | 0.658774 | +0.099033 | +0.024080 | +0.372052 | +0.000000 | +0.000000 | PASS |
| ETCM2.0-mention10 | 0.555118 | 0.685350 | +0.130232 | +0.010399 | +0.510530 | +0.000000 | +0.000000 | PASS |

四库 Macro-AUPR 平均提升为 `+0.098421`。收益主要来自 CW
（cold compound / warm target）状态，WW 也获得小幅正提升；WC 和 CC
与 NoContext 完全一致，这是冻结 base 路由的设计结果，而不是新专家改善了
这两个状态。

完整性检查全部通过：

- outer 阶段训练与优化步数均为 `0`；
- `parameter_selection_on_outer_units=False`；
- 四库 base checkpoint 哈希在评价前后保持不变；
- 四库 Hctx-P head 哈希在评价前后保持不变；
- WC 和 CC 预测与对应 NoContext 报告精确一致。

关键结果文件 SHA-256：

```text
summary.json  0cc3377fa531cd46ceaee3f46dbeb44a3f2e3699d3473d1899eaee5ada1a7720
TCM-Suite    2b603c09151c1ec683fda06c25446b01c55234f557ddee6305109ba0e1cec617
TCMSP        f158bfc47b637e99122d22d36c966f97ec755e45c8c33e376ba690f703250686
SymMap2.0    35a5e83076fb4cb5695a0079eb97a1b0016d9ef3bee856c2888648dac3a27de5
ETCM2.0      2e06db52fdacbbaad5e200c2d74f0326f8630a7e3d36949205226ec3a1661c24
```

据此，V3 可以冻结为“支持状态感知的冻结上下文专家”方法模块。论文中可声明：
训练支持状态决定是否启用可识别的 Hctx-P 上下文专家，从而改善 CW 状态，
并通过冻结基础分支精确保留缺少适用侧信息的 WC/CC 状态。不能声明该方法
全面解决 target-cold 或 double-cold；这两个状态仍是明确的能力边界。

outer 结果产生后不得继续针对这四个 outer unit 调整 V3。后续工作只允许：

1. 固定当前实现、配置、checkpoint/head 和证据哈希；
2. 将结果纳入论文方法证据矩阵与支持状态消融表；
3. 如需更强统计证据，使用预先生成的新 outer unit 或新 seed 独立重复，不能
   复用本次 outer unit 做模型选择。
