# Checkpoint 纯推理排名评价

## 目的

该工具直接恢复已经训练完成的 HDCTI checkpoint，不重新训练，不执行优化器，也不重新进行早停选择。它用于回答两个问题：

1. 在固定的完整蛋白候选集合中，模型能否把留出的真实 C-P 正边排到前面？
2. 当前结果是否提供了尝试 PU Learning 的充分动机？

需要注意：纯推理评价只能判断“PU 是否值得进入下一轮小规模实验”，不能仅凭内部未标注数据证明 PU 是必要的。未记录的 C-P pair 可能是真负例，也可能是尚未收录的真阳性。

## 固定候选协议

对外层测试折中每个至少包含一个正例的 compound：

1. 候选集合为当前模型实体全集中的全部 protein；
2. 过滤该外层训练折中已经观测到的 C-P 正边；
3. 保留外层测试折正边作为待找回目标；
4. 其余 pair 统一称为 `unlabeled`，不称为生物学负例；
5. 分数相同时按 protein ID 升序打破平局；
6. 按 compound 计算 Precision@K、Recall@K、Hits@K 和 MRR，再进行 macro 汇总。

内层 validation 正边属于外层训练折，因此排名时也会被过滤。这样不会把参与早停选择的已知正边错误地当作未标注候选。

## 使用命令

当前 HerbOnly + Dot + early stopping 的 fold 1 checkpoint：

```bash
conda activate HDCTI_tfnew
./tools/evaluate_checkpoint_ranking.py \
  --config configs/HDCTI_herb_only_early_stop_pilot.conf \
  --checkpoint "saved_model/2026-07-14 18-43-19/hdcti_model.ckpt" \
  --fold 1 \
  --ks 10 20 50 \
  --export-top 20
```

只检查 fold、配置和 checkpoint 路径，不加载 TensorFlow：

```bash
./tools/evaluate_checkpoint_ranking.py \
  --config configs/HDCTI_herb_only_early_stop_pilot.conf \
  --checkpoint "saved_model/2026-07-14 18-43-19/hdcti_model.ckpt" \
  --fold 1 \
  --dry-run
```

## 输出文件

默认写入 `results/checkpoint_ranking/<timestamp>/`：

| 文件 | 内容 |
|---|---|
| `report.json` | 配置/checkpoint 哈希、Strict split 哈希、固定候选指标和 PU 证据判断 |
| `per_compound_metrics.tsv` | 每个 compound 的候选数、首个正例排名、P@K/R@K/Hits@K/MRR |
| `top_candidates.tsv` | 每个 compound 的 Top 候选、原始 score、测试正例标记和未标注标记 |

## 当前 checkpoint 结果

评价对象：HerbOnly + Dot + early stopping，Strict fold 1，checkpoint `saved_model/2026-07-14 18-43-19/hdcti_model.ckpt`。

| 指标 | 结果 |
|---|---:|
| 固定采样测试 AUPR（仅历史对照） | 0.984693 |
| 完整候选 pair 数 | 7,445,607 |
| 测试正例占完整候选比例 | 0.1507% |
| MRR | 0.391629 |
| 首个正例排名中位数 / 均值 | 5 / 16.93 |
| Precision@10 / Recall@10 / Hits@10 | 0.112941 / 0.488747 / 0.699369 |
| Precision@20 / Recall@20 / Hits@20 | 0.080811 / 0.674996 / 0.833217 |
| Precision@50 / Recall@50 / Hits@50 | 0.044789 / 0.905710 / 0.955618 |
| Top-10 相对随机正例率的富集倍数 | 约 74.94 倍 |

结果表明，模型并非只在 1:1 随机采样测试对上有效：在每个 compound 约 1,739 个候选 protein 的完整空间中，90.57% 的测试正例可在 Top-50 内找回，95.56% 的测试 compound 至少有一个正例进入 Top-50。MRR 为 0.3916，说明首位排序仍有改进空间，但这不能直接归因于未标注真阳性。

当前结论：**不实现 PU Learning，PU 暂不作为优先创新点。** 后续固定抽取的 30 个 Top 未标注候选已完成 ChEMBL 与文献核验：严格可信阳性 `0/30`；即使宽松计入一条直接但低于效力阈值的关系，也只有 `1/30`，Wilson 95% 上限 `16.67%`，低于预设的 `20%` 实质比例门槛。详见 [TOP_CANDIDATE_VALIDATION.md](TOP_CANDIDATE_VALIDATION.md)。

## 如何判断是否继续 PU

### 暂不优先 PU

如果固定候选的 Recall@K 和 MRR 已经较高，说明当前模型在完整候选空间中仍能较好找回已知正例。此时 PU 会增加训练复杂度，但没有直接证据表明漏标正例是主要瓶颈。

### 先做外部验证，再决定 PU

如果随机采样测试集的 AUPR 很高，但固定候选 Recall@K/MRR 明显较低，说明原有随机负例评价可能过于乐观。不过，这一差距既可能来自未标注真阳性，也可能来自模型排序能力不足。因此应先抽取 `top_candidates.tsv` 中的高分未标注 pair，利用数据库或文献进行盲法核验。

只有在高分未标注候选中发现具有实质比例的可信真阳性时，才有直接理由开展小规模 nnPU/PU pilot。若外部核验命中率很低，应优先考虑排序损失、表示学习或候选级交互，而不是 PU。

以下自动结论是外部核验完成前使用的实验分流规则，当前已经由 [TOP_CANDIDATE_VALIDATION.md](TOP_CANDIDATE_VALIDATION.md) 的实际核验结果取代：

| 条件 | 分流结论 |
|---|---|
| 采样测试 AUPR >= 0.90 且最大 K 的 macro Recall@K < 0.50 | 先外部核验 Top 未标注候选，再决定是否做 PU pilot |
| 最大 K 的 macro Recall@K >= 0.80 且 MRR >= 0.50 | PU 暂不作为当前优先方向 |
| 其他情况 | 证据不足，先核验 Top 未标注候选 |

这些阈值不是 PU 有效性的统计检验，也不能替代外部标签。`report.json` 会始终将“PU 是否必要”记为仅靠内部标签无法识别。

## 公平性约束

- 同一 checkpoint 只评价一次外层测试折，不用外层结果重新选择模型或参数。
- 比较多个 checkpoint 时必须使用相同 Strict manifest、相同 fold、相同候选全集和过滤规则。
- `AUPR` 仍基于 Strict manifest 中固定采样的测试 pair，只作为与历史结果对照；PU 判断以完整候选排名和外部验证为主。
- checkpoint 与配置必须一一对应；工具会检查 TensorFlow 变量名和形状，不匹配时直接终止。
