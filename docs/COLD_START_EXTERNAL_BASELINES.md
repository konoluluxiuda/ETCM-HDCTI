# 成分冷启动外部基线协议

## 目的

最终方法 SCHPT 的主要研究问题是：当测试成分没有训练集 C-P 监督时，能否利用 H-C 与 P-D 侧信息完成成分—靶点预测。因此，仅报告随机边协议下的外部基线不足以支持该结论；外部模型必须在与 SCHPT 完全相同的成分冷启动五折上重新评价。

本实验补充四个同输入外部基线：

| 基线 | 主要结构 | 冷启动预期 |
|---|---|---|
| Dual-HGNN-CTI | H-C / P-D 双超图传播 | 可通过侧图获得未见成分表示 |
| LightGCN-CTI | 训练 C-P 二部图传播 | 无 C-P 邻居时能力受限，是重要下界 |
| R-GCN-CTI | 多关系异构图传播 | 可利用 H-C、P-D 等关系迁移信息 |
| HGT-CTI | 类型与关系感知异构图注意力 | 可利用侧关系，但计算成本较高 |

## 冻结协议

所有 16 个数据集—模型组合统一采用：

```text
split.strategy=compound_cold_start
random.seed=52026
split.seed=52026
validation.seed=152026
evaluation.setup=-cv 5
evaluation.outer.test=True
validation.metric=AUPR
attention.max.nodes=0
```

四库分别为 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0-mention10。它们复用 SCHPT 已生成的 `strict_compound_cold_start_seed_52026_k5` 划分，不重新采样测试实体，不按数据库调节基线超参数。

配置及 SHA-256 固定在 `configs/cold_start_external_baselines_manifest.json`。校验命令：

```bash
python -m tools.validate_cold_start_external_baselines
./run_cold_start_external_baselines_full.sh --dry-run
```

## 正式运行

```bash
./run_cold_start_external_baselines_full.sh
```

16 个作业按顺序运行，结果写入：

```text
results/batch_runs/cold_start_external_baselines_full_<timestamp>/
```

如中途终止，可复用已有目录继续运行：

```bash
HDCTI_BATCH_DIR=results/batch_runs/cold_start_external_baselines_full_<timestamp> \
  ./run_cold_start_external_baselines_full.sh
```

已成功解析的作业会跳过，失败或未完成的作业会重新运行。

## 冻结结果

正式 16 作业已于 2026-08-06 完成：

```text
results/batch_runs/cold_start_external_baselines_full_20260806_134426/
```

Ours-full 的四库 AUPR 为 `0.721718 / 0.957323 / 0.837607 / 0.913655`，
四库 macro 为 `0.857576`。外部基线中，统一 macro 最强的是
Dual-HGNN-CTI（`0.852469`），Ours-full 相对其 macro 提高 `0.005107`。

逐库结果并非全部第一：

| 数据集 | Ours-full | 最佳外部基线 | 差值 |
|---|---:|---:|---:|
| TCM-Suite | 0.721718 | Dual-HGNN-CTI 0.729257 | -0.007539 |
| TCMSP | 0.957323 | Dual-HGNN-CTI 0.939348 | +0.017975 |
| SymMap2.0 | 0.837607 | HGT-CTI 0.839690 | -0.002083 |
| ETCM2.0-mention10 | 0.913655 | Dual-HGNN-CTI 0.910730 | +0.002925 |

因此论文可主张 Ours-full 取得最高四库 macro AUPR，并在 TCMSP 与 ETCM2.0
上排名第一；不能主张在四个数据集上全部最优。

## 论文使用边界

1. 该表用于回答“同一侧信息和同一冷启动划分下，SCHPT 是否优于常用图基线”。
2. LightGCN 在冷启动下的不足是模型归纳偏置造成的预期现象，不能据此宣称所有图协同过滤模型普遍失效。
3. 只有完整五折外层测试结果可进入主表；pilot 或内层验证结果不得混入。
4. 本实验不替代随机边结果。随机边表衡量已见实体上的关系补全，冷启动表衡量未见成分的迁移能力。
