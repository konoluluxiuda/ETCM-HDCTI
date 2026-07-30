# 支持状态一致的内层验证

## 1. 目的

Support-complete 外层测试同时区分：

- `target_cold`：成分在训练 C-P 中可见，蛋白在训练 C-P 中不可见；
- `double_cold`：成分和蛋白均在训练 C-P 中不可见。

若早停仍使用随机 pair 内层验证，模型选择阶段看到的是 warm/warm 样本，与外层
测试状态不一致。因此 support-unit 不能复用普通随机边内层划分。

## 2. Target-cold 内层验证

仅从当前外层单元的训练正边出发：

1. 使用 `validation.seed` 对外层训练 protein 排序；
2. 整体留出 `validation.ratio` 比例的 protein；
3. 从内层训练中删除这些 protein 的全部 C-P 边；
4. 验证正例只保留 compound 在内层训练中仍有正边的 pair；
5. 对每个留出 protein，从内层 warm compound 中采等量未观测 pair；
6. 所有负例均排除原始完整 C-P 的已知正边。

因此验证集满足：

```text
compound: warm
protein: cold
```

## 3. Double-cold 内层验证

仅从当前外层单元的训练正边出发：

1. 分别使用固定 seed 留出 `validation.ratio` 比例的 compound 和 protein；
2. 内层训练删除涉及任一留出端点的全部 C-P 边；
3. 验证正例只取“留出 compound × 留出 protein”块；
4. 同一块中确定性采样等量未观测 pair；
5. 留出 compound 与 warm protein、warm compound 与留出 protein 的正边作为
   隔离缓冲边，不进入训练或验证。

因此验证集满足：

```text
compound: cold
protein: cold
```

这里 `validation.ratio` 是两类端点各自的留出比例，因此验证块约占 pair 空间的
`ratio^2`，不是普通随机边验证记录比例。

## 4. 可复现性与防泄漏

实现位于：

```text
util/support_complete_split.py
HDR.py
```

每次内层划分记录：

- seed 与策略名称；
- 训练/验证正负例数；
- 留出 compound/protein 数；
- double-cold 隔离缓冲正边数；
- 训练记录、验证记录和完整分配 SHA-256。

训练和验证负例均重新生成，不复用外层训练负例。这样可以保证内层留出的实体
不会通过负例进入训练，同时不会把完整 C-P 中的已知正边误标为负例。

## 5. 四库无训练审计

固定：

```text
outer unit: target fold 0 / double C0-P0
validation.ratio: 0.1
validation.seed: 102026
```

| 数据集 | 模式 | 内层训练正例 | 验证正例 | 留出 C | 留出 P | 缓冲正例 |
|---|---|---:|---:|---:|---:|---:|
| TCM-Suite | target-cold | 34,425 | 4,132 | 0 | 685 | 15 |
| TCM-Suite | double-cold | 23,461 | 468 | 93 | 611 | 6,861 |
| TCMSP | target-cold | 40,069 | 7,314 | 0 | 168 | 324 |
| TCMSP | double-cold | 29,155 | 607 | 534 | 143 | 8,579 |
| SymMap2.0 | target-cold | 27,429 | 3,281 | 0 | 328 | 35 |
| SymMap2.0 | double-cold | 18,999 | 355 | 121 | 304 | 5,351 |
| ETCM2.0-mention10 | target-cold | 64,748 | 5,943 | 0 | 41 | 53 |
| ETCM2.0-mention10 | double-cold | 46,484 | 539 | 742 | 40 | 9,550 |

所有单元均具有非空且 1:1 平衡的验证集，未出现已知正边误采为负例。

## 6. Pilot 配置

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_target_cold_no_context_early_stop_unit_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsuite_double_cold_no_context_early_stop_unit_pilot.conf
```

两个配置都设置 `evaluation.outer.test=False`，只用于比较验证 AUPR 与选择 epoch，
不会提前查看外层测试结果。通过后再创建 Gate 1 模块配对配置。

## 7. TCM-Suite 真实早停 Pilot

两个配置均在无可用 CUDA 的执行环境中使用 CPU 完成，稠密全节点注意力关闭。

| 模式 | 内层验证 pair | 早停 epoch | 最佳 epoch | 最佳验证 AUPR | 外层测试 |
|---|---:|---:|---:|---:|---|
| target-cold | 8,264 | 22 | 12 | 0.829882 | 未执行 |
| double-cold | 936 | 22 | 12 | 0.553659 | 未执行 |

两次运行均成功完成 checkpoint 保存与最佳 checkpoint 恢复。结果仅是
NoContext 的 Gate 1 参考点，不是最终外层性能，也不进入论文主结果表。
