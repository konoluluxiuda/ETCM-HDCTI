# Hctx-P 无稠密注意力四库直接消融

## 1. 目的

本实验补齐最终统一口径下 `Strict-HDCTI` 与 `Strict-HDCTI + Hctx-P` 的直接比较，用于回答候选级药材上下文—靶点交互是否在四个数据库上提供稳定收益。

该实验不是新的模型搜索。Hctx-P 的结构、训练参数和已有 checkpoint 均保持冻结，只训练当前缺失的 NoContext 对照。

## 2. 冻结协议

四组配对实验统一使用：

```text
Strict protocol
pair-stratified five-fold split
random/split seed = 2026
validation seed = 102026
inner-validation early stopping by AUPR
Dot decoder
attention.max.nodes = 0
CHCR = off
outer-test evaluation = on
```

每对配置只允许以下字段不同：

```text
model.variant
context.interaction
context.herb_protein
```

配置路径、SHA-256、已有 Hctx-P 结果和判定门槛统一冻结在 `configs/hctx_ablation_manifest.json`。运行前的自动校验会拒绝 batch size、epoch、seed、split、decoder 或其他模型开关发生变化的配置。

## 3. 复用的 Hctx-P 结果

已有四库 Hctx-P 五折结果来自：

```text
results/batch_runs/no_dense_chcr_full_20260717_171403/results.tsv
```

| 数据集 | Hctx-P AUPR |
|---|---:|
| TCM-Suite | 0.991955 (±0.000486) |
| TCMSP | 0.984146 (±0.001740) |
| SymMap2.0 | 0.954348 (±0.002973) |
| ETCM2.0 mention10 | 0.973997 (±0.000780) |

## 4. 预注册判定

Hctx-P 只有同时满足以下条件才通过本轮四库直接消融：

1. 至少 3/4 数据集平均 AUPR 不下降；
2. 四库 macro AUPR 增量至少为 `+0.001`；
3. 任一数据集 AUPR 下降不超过 `0.003`；
4. 至少 3/4 数据集在 5 折中有不少于 3 折 AUPR 提高。

判定门槛在 NoContext 结果产生前写入 manifest，不能根据结果事后修改。若为 `NO-GO`，Hctx-P 不能作为四库共享主干创新；可以降级为特定数据库或特定协议模块，但必须如实报告。

## 5. 运行命令

先检查四库配置、哈希和冻结参考结果：

```bash
python tools/validate_hctx_ablation_configs.py
./run_no_dense_hctx_ablation_batch.sh --dry-run
```

正式运行四个缺失的 NoContext 五折任务：

```bash
./run_no_dense_hctx_ablation_batch.sh
```

中断后使用首次运行打印的结果目录续跑：

```bash
HDCTI_BATCH_DIR=/path/to/results/batch_runs/no_dense_hctx_ablation_TIMESTAMP \
  ./run_no_dense_hctx_ablation_batch.sh
```

脚本会跳过已经成功的任务。全部完成后自动生成：

```text
paired/paired_results.tsv
paired/paired_folds.tsv
paired/decision.json
paired/summary.md
```

这些文件分别保存四库均值差、逐折 AUPR 差、冻结门槛判定和可直接引用的中文汇总。

## 6. 正式结果

正式批次：

```text
results/batch_runs/no_dense_hctx_ablation_20260722_164221
```

| 数据集 | NoContext AUPR | Hctx-P AUPR | 增量 | Hctx-P 提高折数 |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.992210 | 0.991955 | -0.000255 | 1/5 |
| TCMSP | 0.972821 | 0.984146 | +0.011325 | 5/5 |
| SymMap2.0 | 0.940266 | 0.954348 | +0.014082 | 5/5 |
| ETCM2.0 mention10 | 0.962150 | 0.973997 | +0.011847 | 5/5 |

四库 macro AUPR 增量为 `+0.009250`，3/4 数据集平均 AUPR 不下降，3/4 数据集达到至少 3/5 folds 提高，最大单库下降仅为 `0.000255`。四项冻结条件全部满足，最终判定为 **PASS**。

结果支持将 Hctx-P 保留为共享候选级交互模块，但主张边界必须清楚：TCMSP、SymMap2.0 和 ETCM2.0 mention10 获得稳定提升，TCM-Suite 已接近饱和且出现轻微下降。因此不能表述为“四库均显著提升”或“20/20 folds 同向”。

首次批处理在四个训练任务完成后因直接脚本启动时的 Python 包路径错误而未自动汇总；训练结果本身完整有效。修复导入路径后复用同一 `results.tsv` 和日志完成纯汇总，没有重新训练或更改判定门槛。
