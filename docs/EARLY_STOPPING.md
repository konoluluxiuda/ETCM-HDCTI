# Strict 内层验证与早停协议

## 1. 目的

该协议用于统一后续 Strict-HDCTI、HerbOnly 和新增模型的 checkpoint 选择。早停属于实验基础设施，不作为模型创新点，也不使用外层测试折选择 epoch。

每个外层 fold 的数据流为：

```text
outer-train
    -> 固定分层划分
    -> 90% inner-train + 10% validation

inner-train positives
    -> C-P adjacency / PageRank / supervised training

validation pairs
    -> 每 2 epoch 计算一次 AUPR
    -> 选择并恢复最佳 checkpoint

outer-test
    -> 训练和模型选择结束后只评估一次
```

validation 正边不会进入 C-P 训练图、PageRank 或 BCE 训练样本。H-C 和 P-D 仍作为固定侧信息使用，现有 H-D 在 Strict 模式下保持关闭。

## 2. 当前预注册配置

```ini
early.stopping=True
validation.ratio=0.1
validation.seed=102026
validation.metric=AUPR
validation.interval=2
validation.patience=5
validation.min.delta=0.0001
num.max.epoch=50
```

含义：

| 配置 | 含义 |
|---|---|
| `validation.ratio` | 从每个 outer-train 中按正负类别分别抽取 10% |
| `validation.seed` | fold 1 的内层划分 seed；后续 fold 依次加 1 |
| `validation.metric` | checkpoint 选择指标，支持 AUPR 或 AUC，当前固定为 AUPR |
| `validation.interval` | 每 2 epoch 验证一次 |
| `validation.patience` | 连续 5 次验证未达到最小改进后停止 |
| `validation.min.delta` | 有效改进至少为 0.0001 |
| `num.max.epoch` | 最大训练轮数，早停不会超过该值 |

每次划分会输出 inner-train/validation 数量、seed 和 assignment SHA-256 前缀。划分先按实体 pair 排序，再使用局部随机数生成器分层打乱，因此不依赖输入记录顺序，也不会改变训练全局随机状态。

## 3. 模型保存与恢复

第一次 validation 必定建立最佳 checkpoint。后续只有当：

```text
current_metric > best_metric + min_delta
```

时才覆盖 checkpoint。达到 patience 后，训练循环恢复最佳 checkpoint，再生成嵌入、保存模型并评估 outer-test。

关键日志示例：

```text
validation: epoch 12 AUPR=... best=... best_epoch=... stale=.../5
Early stopping triggered at epoch ...
Restored best validation checkpoint: epoch ... AUPR=...
```

validation 只计算清单中的 C-P pairs，不构造完整 compound x protein 预测矩阵。

## 4. 配置文件与命令

HerbOnly 单 fold 试运行：

```bash
./run_hdcti.sh configs/HDCTI_herb_only_early_stop_pilot.conf
```

该配置复用固定五折 manifest，但通过 `evaluation.fold.limit=1` 只执行 fold 1。输出会明确标记为 `first-1-of-5-fold-pilot`，不能作为五折结果引用。

HerbOnly 完整五折：

```bash
./run_hdcti.sh configs/HDCTI_herb_only_early_stop.conf
```

无上下文匹配基线：

```bash
./run_hdcti.sh configs/HDCTI_no_context_early_stop.conf
```

历史固定 50 epoch 配置保留在 `configs/HDCTI_herb_only.conf`，不得与早停结果混合汇总。

## 5. 当前验证状态

已通过确定性分层划分、pair-only 打分、patience/min-delta 和配置校验测试。CPU 端到端冒烟测试已在第 2 epoch 触发早停，并成功恢复第 1 epoch checkpoint。

TCMSP fold 1 的实际划分审查结果为：

| 分区 | 正例 | 负例 | 总数 |
|---|---:|---:|---:|
| inner-train | 40,393 | 40,393 | 80,786 |
| validation | 4,488 | 4,488 | 8,976 |

inner-train 与 validation 的 pair overlap 为 `0`，assignment SHA-256 为 `d6f0e47eb9485bdcfb6fd8b0ffd0ce6b2fa9bda9ac3d86aebdeda18ba855873c`。

已尝试从 Codex 沙箱启动 TCMSP fold 1 pilot，数据划分、Strict 构图和早停配置均正确进入训练；但该运行环境无法执行 `cuInit`，TensorFlow 回退到 CPU，首轮 ETA 超过 1 小时，因此在第 1 epoch 提前停止。该次中止不产生实验结果，也不用于评价模型；GPU pilot 需要在用户的 WSL 终端执行上节命令。

## 6. TCMSP Fold 1 GPU Pilot

用户已于 2026-07-14 在 `HDCTI_tfnew` 环境完成 fold 1 pilot：

```text
best epoch: 36
best validation AUPR: 0.983863
stopped epoch: 46
runtime: 443.389921 s
checkpoint: ./saved_model/2026-07-14 18-43-19/hdcti_model.ckpt
```

第 46 epoch 的 validation AUPR 为 `0.983875`，比已保存最佳值高 `0.000012`，但未超过 `min_delta=0.0001`，因此计入第 5 次 stale check，并恢复 epoch 36。这符合预注册规则，不是 checkpoint 恢复错误。

外层 fold 1 结果为：

| 指标 | Early-stop HerbOnly |
|---|---:|
| AUC | 0.987889 |
| AUPR | 0.984692 |
| Recall | 0.956599 |
| Precision | 0.952694 |
| F1-score | 0.954642 |

与历史固定 50 epoch HerbOnly 的同一 outer fold 描述性比较：

| 指标 | 固定 50 epoch | Early-stop | 差值 |
|---|---:|---:|---:|
| AUC | 0.988789 | 0.987889 | -0.000900 |
| AUPR | 0.986288 | 0.984692 | -0.001596 |
| Recall | 0.955530 | 0.956599 | +0.001069 |
| Precision | 0.957407 | 0.952694 | -0.004713 |
| F1-score | 0.956467 | 0.954642 | -0.001825 |

该差值不能单独归因于早停：固定 50 epoch 模型使用完整 outer-train，而当前模型保留 10% outer-train 作为 validation，只使用 90% 数据训练和构图。单折结果也不能代替五折或多 seed 统计。当前 pilot 只用于验收协议，不据此修改 patience、min-delta 或 validation 比例。

协议验收结论：GPU 训练、周期验证、stale 计数、提前停止、最佳 checkpoint 恢复、外层单次评估和 pilot 结果命名均工作正常。协议已经冻结并用于 Dot/Bilinear/MLP decoder 选择，详见 [PAIR_DECODERS.md](PAIR_DECODERS.md)。

## 7. TCMSP 完整五折结果

2026-07-14 至 2026-07-15，修复后的无上下文基线与最终 HerbOnly 模型在同一预注册早停协议下完成了完整五折。两者复用相同的 Strict manifest、outer/inner 划分、seed、负样本、Dot decoder 和评价代码。

### 7.1 最佳 checkpoint 分布

| Fold | w/o Context 最佳 epoch | Validation AUPR | HerbOnly 最佳 epoch | Validation AUPR |
|---:|---:|---:|---:|---:|
| 1 | 48 | 0.976864 | 32 | 0.983666 |
| 2 | 4 | 0.970684 | 30 | 0.984288 |
| 3 | 4 | 0.972259 | 24 | 0.983850 |
| 4 | 50 | 0.975059 | 32 | 0.985626 |
| 5 | 4 | 0.968011 | 18 | 0.983448 |

无上下文基线在 fold 1 和 fold 4 训练至第 50 epoch 附近，但在 fold 2、3、5 均于 epoch 4 取得最佳值并在 epoch 14 停止，呈现明显的双峰训练轨迹。HerbOnly 的最佳 epoch 位于 18 至 32，五折轨迹更集中。

### 7.2 外层测试五折汇总

| 指标 | w/o Context | HerbOnly | HerbOnly - w/o Context |
|---|---:|---:|---:|
| AUC | 0.977039(±0.004187) | 0.987095(±0.001265) | +0.010056 |
| AUPR | 0.973252(±0.003039) | 0.984085(±0.001782) | +0.010833 |
| Recall | 0.982443(±0.009727) | 0.958290(±0.005606) | -0.024153 |
| Precision | 0.811758(±0.108594) | 0.951731(±0.001752) | +0.139973 |
| F1-score | 0.885411(±0.059742) | 0.954991(±0.002525) | +0.069580 |

| 运行信息 | w/o Context | HerbOnly |
|---|---:|---:|
| 总运行时间 | 1393.354603 s | 1760.008411 s |
| Fold 5 最佳 epoch | 4 | 18 |
| Fold 5 checkpoint | `2026-07-14 23-28-12` | `2026-07-15 00-02-55` |

HerbOnly 的 AUC、AUPR、Precision 和 F1 在 5/5 折均高于无上下文基线，Recall 在 5/5 折降低。它同时显著降低了 fold 波动，尤其是 Precision 标准差由 `0.108594` 降至 `0.001752`，说明药材上下文交互不仅改变阈值下的 Precision/Recall 平衡，也提高了该早停协议下的训练稳定性。

### 7.3 解释边界

早停对照中的大幅 F1 差距不能全部解释为 Hctx-P 的纯结构贡献。无上下文基线有三折在 epoch 4 达到局部最佳，导致 Precision 和 F1 明显下降；该现象放大了两个模型之间的均值差距。由于协议已预注册且两组完全一致，结果可作为最终早停协议下的有效比较，但不能事后根据外层测试结果修改 patience 或 min-delta。

模块作用的保守估计仍以固定 50 epoch 匹配对照为主：HerbOnly 相对 w/o Context 的 AUC、AUPR、Precision 和 F1 分别提高 `0.004972`、`0.006944`、`0.023925` 和 `0.002736`。早停五折则作为额外证据，说明 HerbOnly 对内层模型选择更稳健。HerbOnly 早停结果相对其固定 50 epoch 结果的 AUC、AUPR 和 F1 分别变化 `-0.001385`、`-0.001351` 和 `-0.001729`，运行时间减少 `824.530942 s`（约 `31.9%`）。
