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
2. 当前要求 `early.stopping=False`，防止随机 pair 内层验证污染冷启动协议；
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

## 下一步

两个 TCM-Suite NoContext 单元 smoke 已通过。后续顺序更新为：

1. 固定 support-state 内层验证策略；
2. 恢复早停并得到可比较的 NoContext 单元基线；
3. 实现 `C-Dctx`，只在 Target-cold Gate 1 中评价；
4. 实现 `Hctx-Dctx`，只在 Double-cold Gate 1 中评价；
5. 两个分支均通过后，再实现同一 checkpoint 的四状态训练与路由。

在 Gate 1 前不创建四库长训练配置，不把数据可行性或入口完成度表述为模型创新。
