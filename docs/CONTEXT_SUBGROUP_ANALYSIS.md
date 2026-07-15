# Hctx-P Checkpoint 分组机制分析

## 1. 目的

该分析不重新训练模型，直接比较同一 Strict fold 的无上下文基线与 HerbOnly checkpoint，回答：

1. Hctx-P 的收益是否集中在特定 H-C degree、训练 C-P degree 或 mention_count 区间；
2. Recall 下降是否主要来自低证据成分；
3. 显式上下文项主要完成 `FP -> TN`，还是同时造成大量 `TP -> FN`；
4. 是否有直接证据支持下一步实现“药材上下文可靠性门控”。
5. 分别由 inner validation 选择 F1 阈值后，Recall/F1 损失是否恢复。

## 2. 固定评价对象

当前首轮分析使用 ETCM2.0_core_mention10 Strict fold 1：

| 对象 | 配置 | Checkpoint |
|---|---|---|
| 保守基线 | `configs/HDCTI_etcm_mention10_no_context_early_stop_max80.conf` | `saved_model/2026-07-15 13-38-30/hdcti_model.ckpt` |
| HerbOnly | `configs/HDCTI_etcm_mention10_herb_only_early_stop.conf` | `saved_model/2026-07-15 12-56-37/hdcti_model.ckpt` |

选择 Max-80 基线是为了比 Max-50 更保守地估计 Hctx-P 效应。两份配置复用相同 Strict manifest、inner-train、validation 和 outer-test pair。脚本会校验对应哈希，不匹配时拒绝运行。

## 3. 运行命令

先执行不加载 TensorFlow 的协议检查：

```bash
python tools/analyze_context_subgroups.py \
  --baseline-config configs/HDCTI_etcm_mention10_no_context_early_stop_max80.conf \
  --baseline-checkpoint "saved_model/2026-07-15 13-38-30/hdcti_model.ckpt" \
  --herb-config configs/HDCTI_etcm_mention10_herb_only_early_stop.conf \
  --herb-checkpoint "saved_model/2026-07-15 12-56-37/hdcti_model.ckpt" \
  --fold 1 \
  --dry-run
```

正式纯推理分析：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0

python tools/analyze_context_subgroups.py \
  --baseline-config configs/HDCTI_etcm_mention10_no_context_early_stop_max80.conf \
  --baseline-checkpoint "saved_model/2026-07-15 13-38-30/hdcti_model.ckpt" \
  --herb-config configs/HDCTI_etcm_mention10_herb_only_early_stop.conf \
  --herb-checkpoint "saved_model/2026-07-15 12-56-37/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/context_subgroups/etcm_mention10_fold1
```

该命令顺序恢复两个 checkpoint，不执行优化器或训练步骤，也不同时保留两个 TensorFlow session。工具会同时输出固定 `0.5` 阈值结果，以及分别在相同 inner validation 上选择最大 F1 阈值后原样应用于 outer-test 的校准结果。

## 4. 输出文件

| 文件 | 内容 |
|---|---|
| `report.json` | 配置、checkpoint、split 哈希及全部嵌套指标 |
| `report.md` | 总体与分组指标的可读报告 |
| `pair_scores.tsv` | 每个测试 pair 的基线分数、Herb base、Hctx-P logit、Herb 总分和混淆转移 |
| `subgroup_metrics.tsv` | 各分组的 AUC/AUPR/Recall/Precision/F1 差值和转移计数 |
| `subgroup_metrics_calibrated.tsv` | 使用 inner-validation 阈值后的分组指标与混淆转移 |

固定分组：

```text
H-C degree: 0 / 1 / 2-3 / 4-10 / >10
training C-P degree: 0 / 1-2 / 3-5 / 6-10 / >10
mention_count: <10 / 10-19 / 20-49 / 50-99 / >=100 / missing
```

## 5. 分数解释

脚本输出三种模型分数：

```text
Baseline：独立训练的 Max-80 无上下文模型
Herb base only：HerbOnly checkpoint 的节点表示，但关闭显式 Hctx-P 加法项
Herb total：HerbOnly checkpoint 的完整 Hctx-P 分数
```

`Herb total - Herb base only` 描述同一 checkpoint 中显式上下文项的直接作用；`Herb total - Baseline` 同时包含训练期间节点表示变化和显式上下文作用。

## 6. 门控 Go/No-Go 条件

支持实现可靠性门控的证据：

* 低 H-C degree 或低 mention_count 组的 Recall/F1 降幅明显大于高证据组；
* `TP -> FN` 主要集中在低证据组；
* 低证据组的 context logit 对正负样本区分较弱；
* 高证据组仍保持稳定 AUPR/Precision 收益。

停止门控方向的情形：

* Recall 损失在各证据组近似均匀；
* 低证据组反而获得最大的稳定收益；
* 分组样本量或正例数过低，无法形成可靠模式；
* 观察模式只来自一个很窄的 degree 区间。

该分析使用 outer-test 标签进行事后机制解释。它可以决定是否值得提出新假设，但不能用于调节门控超参数后再把同一 fold 当作无偏确认性测试；门控若实现，只能先使用 inner validation 筛选。

## 7. Fold 1 审查结果

本次分析于 2026-07-15 完成，输出位于：

```text
results/context_subgroups/etcm_mention10_fold1
```

### 7.1 总体变化

| 指标 | Baseline | HerbOnly | 差值 |
|---|---:|---:|---:|
| AUC | 0.971394 | 0.977748 | +0.006354 |
| AUPR | 0.966567 | 0.972871 | +0.006304 |
| Recall | 0.947984 | 0.935885 | -0.012099 |
| Precision | 0.891814 | 0.930780 | +0.038966 |
| F1-score | 0.919042 | 0.933326 | +0.014284 |

混淆转移为：

```text
FN -> TP: 349
TP -> FN: 563
FP -> TN: 1119
TN -> FP: 316
```

Hctx-P 共净修正 803 个负例，同时净损失 214 个正例，因此总体表现为 Precision 明显提高、Recall 小幅降低。上下文 logit 在正例上的均值为 `0.372232`，在负例上的均值为 `-6.533650`，说明该项具备明确的正负区分能力，并非无效噪声。

### 7.2 分组结论

* H-C degree 与 Recall 损失不呈单调关系：degree=1 的 Recall 仅下降 `0.007561`，而 `>10` 组下降 `0.051282`。因此不支持直接按 H-C degree 设计可靠性门控。
* mention_count 越低时 Recall 损失总体略大，但差异不足以单独支持 mention 门控。
* 训练 C-P degree 呈现最清晰的梯度：degree=0、1-2、3-5、6-10、>10 的 Recall 差值依次为 `-0.107692`、`-0.051710`、`-0.010875`、`-0.008786`、`-0.004237`。
* 但 degree=0 组的 AUC/AUPR 分别提高 `0.076222/0.113820`，Precision 提高 `0.187059`。这说明低度组的排序质量显著改善，F1 下降更接近固定 `0.5` 阈值下的分数校准问题，而不是表示失效。

### 7.3 Go/No-Go 决策

**No-Go：暂不实现基于 H-C degree 或 mention_count 的药材上下文可靠性门控。** 当前证据没有显示 Hctx-P 在低药材证据实体上系统失效，贸然门控可能削弱已经获得的 AUPR 和 Precision 收益。

工具现已集成 inner-validation F1 阈值校准；重新运行第 3 节正式命令即可在同一报告中得到固定阈值和校准阈值结果。该诊断不使用 outer-test 选择阈值，也不作为新的模型创新点。

### 7.4 阈值校准结果

inner validation 选择的阈值为：

```text
Baseline: 0.695987
HerbOnly: 0.520982
```

将阈值原样应用于 outer-test 后：

| 指标 | Baseline calibrated | HerbOnly calibrated | 差值 |
|---|---:|---:|---:|
| AUC | 0.971394 | 0.977748 | +0.006354 |
| AUPR | 0.966567 | 0.972871 | +0.006304 |
| Recall | 0.927517 | 0.933624 | +0.006106 |
| Precision | 0.913470 | 0.932095 | +0.018626 |
| F1-score | 0.920440 | 0.932859 | +0.012419 |

校准后的混淆转移为 `FN->TP=584`、`TP->FN=476`、`FP->TN=755`、`TN->FP=404`。相对差异中的 Recall 已由负转正，但这主要来自 Baseline 的验证最优阈值高于 `0.5`，不能表述为校准提高了 HerbOnly 的绝对 Recall。

训练 C-P degree=0 组的 Recall 差值从 `-0.107692` 缩小到 `-0.011538`，F1 差值从 `-0.038868` 转为 `+0.023403`；其余 C-P degree 组的 Recall、Precision 和 F1 均为正增益。由此正式关闭 H-C/mention gate 和 degree-aware calibration 两个分支。当前证据支持冻结静态 Hctx-P，并将下一阶段转向候选蛋白条件化药材选择这一独立启用的细化模块。
