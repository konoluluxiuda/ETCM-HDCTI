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
