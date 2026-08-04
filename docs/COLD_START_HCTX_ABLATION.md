# Compound cold-start 递进消融

## 1. 目的

补齐同一 Strict compound cold-start 协议下的三阶段结果：

```text
NoContext
   -> Hctx-P
   -> Hctx-P + SDIS
```

该实验不是继续筛选模型，而是回答两个已冻结机制如何在同一任务中递进：

1. Hctx-P 是否为训练中未见 C-P 关系的 compound 提供可迁移药材上下文；
2. SDIS 是否进一步消除零训练 C-P 支持时不可迁移的 ID 基础分。

## 2. 冻结边界

四库均固定：

```text
experiment.protocol=strict
split.strategy=compound_cold_start
random.seed=2026
split.seed=2026
validation.seed=102026
evaluation.setup=-cv 5
early.stopping=True
pair.decoder=dot
attention.max.nodes=0
num.max.epoch=50
```

NoContext 与 Hctx-P 配置只允许三项不同：

```text
model.variant
context.interaction
context.herb_protein
```

Hctx-P 和 SDIS 结果直接复用：

```text
results/batch_runs/sdis_full_20260718_212240/results.tsv
```

配置路径、SHA-256 和结果来源冻结在：

```text
configs/cold_start_hctx_ablation_manifest.json
```

## 3. 运行

先检查四库配置、哈希和冻结结果：

```bash
./run_cold_start_hctx_ablation_batch.sh --dry-run
```

随后只运行缺失的四个 NoContext 五折：

```bash
./run_cold_start_hctx_ablation_batch.sh
```

如中断，复用终端给出的结果目录：

```bash
HDCTI_BATCH_DIR=/path/to/existing/run ./run_cold_start_hctx_ablation_batch.sh
```

脚本会跳过该目录中已经成功的配置。

## 4. 输出

```text
results/batch_runs/cold_start_hctx_ablation_<timestamp>/
├── results.tsv
├── summary.md
├── environment.txt
├── 01_tcmsuite.log
├── 02_tcmsp.log
├── 03_symmap.log
├── 04_etcm_mention10.log
└── paired/
    ├── results.tsv
    └── summary.md
```

`paired/summary.md` 同时报告：

```text
Hctx-P - NoContext
SDIS - Hctx-P
```

该汇总不设置新的 Go/No-Go 门槛，不允许根据结果修改 split、seed、epoch、
attention 或模块参数。

## 5. 完整五折结果

运行目录：

```text
results/batch_runs/cold_start_hctx_ablation_20260730_133133
```

| 数据集 | NoContext AUPR | Hctx-P AUPR | Hctx-P 增量 | 正向 folds | SDIS AUPR | SDIS 增量 |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 0.440040 | 0.643662 | +0.203622 | 5/5 | 0.702967 | +0.059305 |
| TCMSP | 0.334094 | 0.918342 | +0.584248 | 5/5 | 0.941233 | +0.022891 |
| SymMap2.0 | 0.390552 | 0.797639 | +0.407087 | 5/5 | 0.809854 | +0.012215 |
| ETCM2.0 mention10 | 0.324931 | 0.881279 | +0.556348 | 5/5 | 0.898965 | +0.017686 |
| **Macro 增量** | - | - | **+0.437826** | **20/20** | - | **+0.028024** |

结果形成两段一致递进：

1. Hctx-P 在四库 20/20 folds 均优于 NoContext，说明候选 compound 的药材
   上下文是当前模型在 compound cold-start 下产生可迁移分数的关键来源；
2. SDIS 在相同 Hctx-P 骨干上继续提高四库 AUPR，说明关闭零训练 C-P 支持实体
   的不可迁移 ID 基础分具有额外价值。

## 6. 解释边界

NoContext 的 AUC 在四库均低于 `0.5`，说明原始 ID 主分支在未见 compound 上
发生严重评分失配。Hctx-P 的大幅增益部分来自修复这一明确失败模式，不能直接
解释为相对现代归纳式或属性式 cold-start 方法的同等幅度优势。

该协议允许测试 compound 使用完整 H-C 侧信息，因此准确名称是：

```text
side-information-assisted compound cold-start
```

不能表述为不依赖任何侧信息的 de novo compound cold-start。NoContext 没有
执行事后阈值校准，其固定 `0.5` 分类指标只用于披露；主机制判断优先依据与阈值
无关的 AUC/AUPR。
