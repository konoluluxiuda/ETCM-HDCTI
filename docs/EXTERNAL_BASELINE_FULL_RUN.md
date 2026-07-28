# 外部同输入基线正式五折运行说明

## 1. 实验目的

本工作在相同 Strict 数据划分、训练监督和评价流程下比较四类拓扑基线：

| 模型 | 输入图 | 论文中的准确命名 |
|---|---|---|
| `Dual-HGNN-CTI` | H-C 成分超图、P-D 蛋白超图 | `HGHDA-inspired Dual-HGNN-CTI` |
| `LightGCN-CTI` | 当前 fold 训练 C-P 二部图 | `LightGCN-CTI (same-input BCE adaptation)` |
| `R-GCN-CTI` | H-C、当前 fold 训练 C-P、P-D 六类有向关系 | `R-GCN-CTI (same-input adaptation)` |
| `HGT-CTI` | 同一六关系异构图，统一稀疏关系注意力 | `HGT-CTI (same-input sparse attention adaptation)` |

这些模型用于补充相同输入条件下的结构比较，不能写成对原论文中依赖 SMILES、
蛋白序列或其他属性的模型进行了原样复现。

## 2. 冻结协议

前三种模型的 12 个正式配置和后续 HGT 的 4 个正式配置统一使用：

```text
experiment.protocol=strict
split.strategy=pair_stratified
split.seed=2026
evaluation.setup=-cv 5
evaluation.outer.test=True
early.stop.metric=aupr
early.stop.eval.every=2
early.stop.patience=5
early.stop.min.delta=0.0001
num.factors=64
num.max.epoch=50
batch_size=2000
pair.decoder=dot
```

正式配置只在相应 pilot 基础上进行三项协议切换：

1. 删除 `evaluation.fold.limit=1`；
2. 将 `evaluation.outer.test` 从 `False` 改为 `True`；
3. 将 `run.variant` 的 `_pilot_v1` 改为 `_full_v1`。

不得根据单折 pilot 的结果为某个数据库单独修改层数、学习率、正则、训练预算
或早停规则。

## 3. 运行命令

先检查任务清单：

```bash
./run_external_baselines_full_batch.sh --dry-run
```

顺序运行前三种模型的 12 个正式五折任务：

```bash
./run_external_baselines_full_batch.sh
```

HGT 通过四库 pilot 后使用独立冻结批次：

```bash
./run_hgt_cti_full_batch.sh
```

批处理会写入：

```text
results/batch_runs/external_baselines_full_<timestamp>/
├── results.tsv
├── summary.md
├── environment.txt
└── *.log
```

中断后复用同一目录即可跳过已经成功的任务：

```bash
HDCTI_BATCH_DIR=results/batch_runs/external_baselines_full_<timestamp> \
  ./run_external_baselines_full_batch.sh
```

已有结果时只刷新 Markdown 汇总：

```bash
HDCTI_BATCH_DIR=results/batch_runs/external_baselines_full_<timestamp> \
  ./run_external_baselines_full_batch.sh --summarize-only
```

## 4. 结果使用边界

`summary.md` 中的五折均值和 fold 标准差可进入同输入基线表。单折 pilot 只证明
实现能够稳定训练，不进入最终性能主表。当前批次不自动进行显著性检验，也不
把不同模型间的描述性差异表述为统计显著差异。

原 HDCTI 论文报告的八模型结果应保留在独立的 Legacy 文献比较表中，不与本批
Strict 结果合并计算增量或显著性。

## 5. 已完成批次

2026-07-28 的冻结批次已完成 12/12 个任务：

```text
results/batch_runs/external_baselines_full_20260728_152020/
```

结果文件 SHA-256 为：

```text
bf03faffbec11ad809f4b63f418d45c6ad2c66c7df973ffbae48472eb4095770
```

该哈希和逐行配置哈希已写入 `configs/paper_results_manifest.json`，并由
`tools/build_paper_results_tables.py` 在生成
`FINAL_RESULTS_TABLES.md` 前自动核验。

HGT 独立冻结批次也已完成 4/4 个任务：

```text
results/batch_runs/hgt_cti_full_20260728_181436/
```

结果文件 SHA-256：

```text
64cff4189c0a8b7166b4d27dadac9097a17eff320c96066581a0eb9b50776a64
```

至此，同输入外部比较共包含 4 个方法、16 个四库外层五折任务。HGT 采用统一
每关系/目标节点 64 入邻居上限，其 ETCM 结果不能解释为无采样完整 HGT 性能。
