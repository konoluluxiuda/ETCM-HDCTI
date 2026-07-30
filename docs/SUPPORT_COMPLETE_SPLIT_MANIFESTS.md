# 支持状态完备的冷启动划分清单

## 1. 当前状态

2026-07-30 已完成四库 `seed=2026`、`k=5` 的支持状态冷启动划分冻结，
并已升级为 manifest v2。
本阶段只生成实体分组、测试候选和可重建训练正边的哈希，没有训练模型，也没有
修改当前 Strict、Hctx-P、CHCR 或 SDIS 的实现。

生成命令：

```bash
python tools/prepare_support_complete_splits.py
```

第二次执行会校验源关系文件、所有输出工件及 SHA-256，并直接复用原 manifest。
源文件或任一冻结测试文件发生变化时，脚本会拒绝复用。

## 2. 生成位置

| 数据集 | Manifest 目录 | 占用 |
|---|---|---:|
| TCM-Suite | `dataset/TCMsuite/splits/support_complete_seed_2026_k5` | 1.4 MB |
| TCMSP | `dataset/TCMSP/splits/support_complete_seed_2026_k5` | 1.9 MB |
| SymMap2.0 | `dataset/Symmap/splits/support_complete_seed_2026_k5` | 1.9 MB |
| ETCM2.0-mention10 | `dataset/ETCM2.0_core_mention10/splits/support_complete_seed_2026_k5` | 4.2 MB |

数据目录不上传 Git；生成脚本、测试和本记录进入版本控制即可复建完全相同的划分。

## 3. 冻结内容

每个数据集包含 32 个带哈希工件：

```text
cold_target_groups.tsv
double_cold_compound_groups.tsv
target_cold/test_fold_0.tsv ... test_fold_4.tsv
double_cold/test_c0_p0.tsv ... test_c4_p4.tsv
manifest.json
```

其中：

- `cold_target_groups.tsv` 固定 P-D 支撑 protein 的五组分配；
- `double_cold_compound_groups.tsv` 固定 H-C 支撑 compound 的五组分配；
- Target-cold 测试折只保留训练 C-P 中仍可见的 compound；
- Double-cold 使用完整 `5×5` 网格，每条双侧支撑正边恰好进入一个测试格；
- 每个测试文件具有等量正例与未观测候选；
- 所有未观测候选均排除完整 C-P 文件中的已知正边；
- manifest 保存源文件哈希、实体组哈希、测试文件哈希，以及每个单元的训练
  正边与确定性训练负例哈希。

## 4. 四库统计

| 数据集 | Target-cold 测试正例 | Double-cold 测试正例 | Double-cold 单元 | 完整覆盖 |
|---|---:|---:|---:|---:|
| TCM-Suite | 25,315 | 25,482 | 25 | 100% |
| TCMSP | 40,606 | 41,762 | 25 | 100% |
| SymMap2.0 | 35,481 | 35,960 | 25 | 100% |
| ETCM2.0-mention10 | 87,411 | 88,431 | 25 | 100% |

Target-cold 总数略低于双侧支撑正边，是因为部分 compound 的全部已知 C-P
关系都连接到当折 held-out protein。若保留这些 pair，它们实际属于
`cold compound / cold target`，会污染 `warm compound / cold target` 评价，
因此按协议排除。

## 5. 为什么没有直接接入旧 Fold Loader

旧 `fold_assignments.tsv` 的语义是：

```text
当前测试 fold 以外的全部 pair = 训练集
```

该语义适用于随机 pair 五折和单端 compound cold-start，但不适用于 double-cold。
对于测试格 `C_i × P_j`，训练集必须同时排除：

```text
C_i × 所有 protein
所有 compound × P_j
```

如果直接复用旧 loader，同一 compound 行或 protein 列中的其他格会进入训练，
造成实体泄漏。因此没有向旧 `fold_assignments.tsv` 增加特殊分支，而是新增独立
的 `util/support_complete_split.py`。该 loader 由源 C-P 与 manifest 重建训练集，
并同时核对：

```text
training_positives_sha256
training_negatives_sha256
test_records_sha256
```

训练负例按 compound 进行确定性的互素步长遍历，在当前单元允许的 protein
空间中按 1:1 生成，且排除完整 C-P 中的全部已知正边。manifest 只保存负例
数量和哈希，不需要为 30 个单元重复存储训练文件。

## 6. 已验证不变量

专用测试覆盖：

1. Double-cold 全网格中的正边不重复且完整覆盖；
2. 测试负例不与任一已知 C-P 正边重叠；
3. 每折/每格正负样本数相等；
4. 相同源文件、seed 和 fold 数重复运行时逐字节复用；
5. 源文件变化时拒绝复用；
6. 任一冻结工件被修改时拒绝复用。

运行结果：

```text
11 tests passed
4/4 real manifests passed integrity assertions
4/4 fixed loader smoke units passed
```

## 7. 下一步

显式单元 loader 已完成以下工作：

1. `target_cold` loader 根据 protein 组重建训练 C-P，并核对训练哈希；
2. `double_cold` loader 接收 `(compound_group, protein_group)`，同时排除对应行列；
3. 加载冻结测试文件并再次检查 train/test 实体交集为 0；
4. 重建与正例等量的确定性训练负例，并核对负例哈希；
5. 四库固定 `target fold 0` 与 `double C0/P0` 数据加载 smoke 均通过。

下一阶段是增加单单元实验入口，使 HDR 可以直接接收 loader 返回的
`training_records` 和 `test_records`。先用当前 NoContext/Hctx-P 运行协议 smoke，
确认 PageRank 只使用当前单元训练正边、权重保存和评价链路均正常；通过后才实现
`C-Dctx` 与 `Hctx-Dctx`。

必须避免把 25 个 double-cold 单元压缩为仅五个结果较好的对角单元。低成本
Gate 1 可以使用预先固定的 `C0/P0`，最终结论仍需覆盖完整网格或预注册的
Latin-square 轮换。
