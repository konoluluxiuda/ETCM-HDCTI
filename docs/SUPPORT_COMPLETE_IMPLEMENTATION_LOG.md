# 支持状态完备协议实施记录

## 2026-07-30：数据可行性审计

新增：

```text
tools/audit_support_complete_cold_start.py
tests/test_support_complete_cold_start_audit.py
docs/SUPPORT_COMPLETE_COLD_START_FEASIBILITY.md
```

完成四库 Target-cold 与 Double-cold 无训练审计。四库均通过预设的数据量、
侧信息覆盖、实体隔离和总体未观测候选容量门槛。该结论只允许进入协议实现，
不代表 SCCI 已成为有效方法。

## 2026-07-30：Manifest v1

新增：

```text
tools/prepare_support_complete_splits.py
tests/test_support_complete_split_manifests.py
docs/SUPPORT_COMPLETE_SPLIT_MANIFESTS.md
```

冻结内容包括：

- P-D 支撑 target 的 5 组分配；
- H-C 支撑 compound 的 5 组分配；
- 5 个 Target-cold 测试折；
- 完整 25 个 Double-cold 测试格；
- 源文件、实体组、测试记录和训练正边集合的 SHA-256。

Double-cold 使用完整 `5×5` 网格，没有把 25 格压缩为结果可挑选的 5 个对角格。

## 2026-07-30：Manifest v2 与显式 Loader

新增：

```text
util/support_complete_split.py
tools/smoke_support_complete_loader.py
docs/SUPPORT_COMPLETE_LOADER_SMOKE.md
```

Manifest v2 进一步固定可重建的训练负例哈希。训练负例按 compound 在当前单元
允许的 protein 空间内确定性生成，数量与训练正例相同，并排除完整 C-P 中全部
已知正边。

显式 loader 支持：

```text
target_cold(fold)
double_cold(compound_group, protein_group)
```

四库固定 `target fold 0` 与 `double C0/P0` 均通过：

- train/test pair 交集为 0；
- Target-cold protein 交集为 0；
- Double-cold compound 与 protein 交集均为 0；
- 训练正例、训练负例和测试记录哈希全部匹配；
- 11 项相关单元测试通过。

## 2026-07-30：HDR 单单元入口

新增 `evaluation.setup=-support-unit`，并使用以下配置定位单元：

```text
support.manifest=...
support.mode=target_cold
support.target.fold=0
```

或：

```text
support.mode=double_cold
support.compound.group=0
support.protein.group=0
```

安全约束：

1. 只允许 `experiment.protocol=strict`；
2. `early.stopping=True` 时必须使用 support-state 内层验证，不能回退为随机 pair；
3. 当前要求 `negative.strategy=random`，实际训练负例已由 manifest v2 固定；
4. `datapath` 与 manifest 的 C-P 数据目录必须一致；
5. support-unit 不经过旧 `fold_assignments.tsv` loader；
6. Rating 仍只使用显式单元训练正边构造 Strict C-P 图和 PageRank。

首批 smoke 配置：

```text
configs/HDCTI_tcmsuite_target_cold_no_context_unit_smoke.conf
configs/HDCTI_tcmsuite_double_cold_no_context_unit_smoke.conf
```

二者固定 `num.max.epoch=1`、关闭上下文创新模块和稠密全节点注意力，只用于检查
训练、PageRank、保存与评价链路，不能作为性能结果。

## 2026-07-30：TCM-Suite 真实协议 Smoke

两个配置均在 `HDCTI_tfnew` 环境成功完成。执行环境未提供可用 CUDA 设备，
TensorFlow 自动使用 CPU；未启用稠密全节点注意力。

### Target-cold fold 0

```text
Manifest train: 38,572 positive + 38,572 negative
Manifest test:   5,065 positive +  5,065 negative
Strict C-P graph: 38,572 training positive edges
Epoch: 1/1
Training time: about 2 s
Total runtime: 12.40 s
AUC:  0.792774
AUPR: 0.732784
```

### Double-cold C0/P0

```text
Manifest train: 30,790 positive + 30,790 negative
Manifest test:     999 positive +    999 negative
Strict C-P graph: 30,790 training positive edges
Epoch: 1/1
Training time: about 2 s
Total runtime: 8.05 s
AUC:  0.460085
AUPR: 0.465397
```

两个单元均完成：

- 显式单元加载；
- Strict 训练 C-P 图构造；
- PageRank 仅使用单元训练正边；
- 1 epoch 训练；
- checkpoint 保存；
- 外层测试评价；
- 带 unit key 的结果文件输出。

这些指标来自未早停、只训练 1 epoch 的 NoContext 流程。它们只能证明入口和评价
链路可运行，不能与五折基线比较，也不能用于判断 C-Dctx、Hctx-Dctx 或 SCCI
是否有效。

## 2026-07-30：支持状态一致的内层验证

已实现：

```text
build_support_state_inner_validation(...)
```

其语义为：

- Target-cold：整体留出内层 protein，validation compound 必须在内层训练中可见；
- Double-cold：同时留出 compound 与 protein，验证只使用二者交叉块；
- 所有涉及 double-cold 单侧留出端点的交叉正边作为隔离缓冲边；
- 训练负例和验证负例均重新按 seed 构造，并排除完整 C-P 已知正边；
- 输出训练、验证和完整分配哈希。

四库 `target fold 0` 与 `double C0/P0` 在 `ratio=0.1`、
`validation.seed=102026` 下均通过无训练审计。详情见：

```text
docs/SUPPORT_STATE_INNER_VALIDATION.md
```

新增可运行 pilot：

```text
configs/HDCTI_tcmsuite_target_cold_no_context_early_stop_unit_pilot.conf
configs/HDCTI_tcmsuite_double_cold_no_context_early_stop_unit_pilot.conf
```

两个 CPU pilot 已真实完成：

```text
Target-cold: best validation AUPR 0.829882 at epoch 12; stopped at epoch 22
Double-cold: best validation AUPR 0.553659 at epoch 12; stopped at epoch 22
```

二者均设置 `evaluation.outer.test=False`，外层测试未执行。这些结果只证明
support-state 验证、早停、checkpoint 保存和恢复链路完整。

## 2026-07-30：缺失上下文分支 Gate 1

两个候选均使用与对应 NoContext 完全相同的 manifest、单元、inner-validation、
seed、早停和模型参数，只改变上下文项开关；外层测试均关闭。

### Target-cold C-Dctx

```text
NoContext best validation AUPR: 0.829882
C-Dctx best validation AUPR:    0.606159
delta:                         -0.223723
best epoch:                    50
context weight mean abs:        2.817455
```

结论：Gate 失败。停止 C-Dctx，不增加 seed、不调验证划分、不读取外层测试。

### Double-cold Hctx-Dctx

```text
NoContext best validation AUPR: 0.553659
Hctx-Dctx best validation AUPR: 0.578218
delta:                         +0.024559
best epoch:                    44
context weight mean abs:        2.233022
```

结论：通过单库单单元低成本 Gate，但尚未形成四库证据。

## 下一步

由于两个缺失分支没有同时通过，暂不实现原定的完整四状态路由。先对已保存的
C-Dctx 与 Hctx-Dctx checkpoint 做内层验证纯推理分解，比较 base-only、
context-only 和相加分数。只有确认 Hctx-Dctx 的独立上下文分支在 Double-cold
状态下仍具备有效排序能力，才实现同一 checkpoint 的训练支持度路由。

## 2026-07-30：上下文 checkpoint 纯推理分解

新增：

```text
tools/audit_support_context_components.py
```

脚本只恢复 checkpoint，并在 support-state inner-validation 上计算 base-only、
context-only 与相加分数；训练步数为 0，外层测试关闭。

主要结果：

```text
Target-cold C-Dctx:
  base-only AUPR       0.567753
  context-only AUPR    0.579406
  total AUPR           0.606159
  frozen NoContext     0.829882

Double-cold Hctx-Dctx:
  base-only AUPR       0.543369
  context-only AUPR    0.566562
  total AUPR           0.578218
  frozen NoContext     0.553659
```

结论：

1. C-Dctx 联合训练破坏了 base 表示，context-only 也不足以替代 NoContext；
2. Hctx-Dctx context-only 相对 NoContext 提高 `0.012903`，具有独立路由价值；
3. 下一实现采用训练支持度确定性路由，Target-cold 保留 base，Double-cold
   使用梯度隔离的 Hctx-Dctx head，不再要求每个状态都新增一个上下文公式。

## 2026-07-30：共享四状态训练单元

新增：

```text
build_four_state_support_unit(...)
tests/test_support_complete_four_state.py
docs/SUPPORT_COMPLETE_FOUR_STATE_UNIT.md
```

该单元固定一组 cold compound 和 cold protein，从同一训练图同时生成
warm-warm、cold-warm、warm-cold 和 cold-cold 四个互斥测试集合。四库真实
构造均通过，正例数如下：

| 数据集 | warm-warm | cold-warm | warm-cold | cold-cold |
|---|---:|---:|---:|---:|
| TCM-Suite | 3,079 | 6,953 | 4,074 | 999 |
| TCMSP | 3,834 | 9,113 | 6,374 | 1,645 |
| SymMap2.0 | 2,470 | 5,757 | 5,681 | 1,470 |
| ETCM2.0-mention10 | 5,657 | 14,160 | 13,932 | 3,516 |

现有 support-complete 与 Strict 回归测试共 37 项通过。下一步先把该派生单元
冻结为带哈希的磁盘 artifact，并实现四状态一致的 inner-validation；在此之前
不接模型、不运行四库 outer test。
