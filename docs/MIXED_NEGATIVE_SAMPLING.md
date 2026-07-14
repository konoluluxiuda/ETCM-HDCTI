# 结构感知混合未观测样本

## 1. 目的

当前 Strict 协议从未记录 C-P pairs 中固定抽取与正例等量的随机样本，并在 BCE 中将其作为标签 0。未记录关系并不等于经过实验确认的生物学负关联，而且随机样本通常较容易区分。

本模块在保持正负数量、outer-test 和 inner-validation 清单不变的前提下，将部分 inner-train 随机负例替换为结构上更相似的未观测 pairs。第一版只用于训练增强，不改变模型结构。

## 2. 配置

```ini
negative.strategy=random
negative.hard.ratio=0.25
negative.seed=202026
```

可选策略：

* `random`：完全保留 Strict manifest 中的原始训练记录，是默认值；
* `mixed`：保留部分随机负例，并加入固定比例的结构困难未观测 pairs。

`negative.hard.ratio` 表示希望替换的训练负例比例，必须位于 `(0, 1]`。若侧信息可生成的有效 pair 不足，程序保留更多原随机负例，并报告实际比例。

## 3. 困难 pair 构造

对 inner-train 正例 `(c, p)`，按以下两种方式寻找替换 pair：

1. P-D 靶点替换：选择与 `p` 共享疾病上下文的 `p'`，构造 `(c, p')`；
2. H-C 成分替换：选择与 `c` 共享药材上下文的 `c'`，构造 `(c', p)`。

同侧候选按上下文 Jaccard 相似度从高到低排序。程序交替优先尝试 P-D 和 H-C；当 TCMSP 的 P-D 共享邻居不足时，使用 H-C 候选补足目标比例。

第一版有意不使用完整 C-P 图、模型预测分数或 H-D，因此不是动态困难负例，也不会引入完整标签派生视图。

## 4. 无泄漏约束

处理顺序固定为：

```text
Strict outer fold
        ↓
inner-train / validation 固定划分
        ↓
仅修改 inner-train negatives
        ↓
训练模型
```

生成 hard pairs 时：

* 只使用固定 H-C、P-D 侧关系；
* 只使用 inner-train 正例作为锚点；
* 不读取 validation 或 outer-test 的标签；
* 将 validation 和 outer-test 的 pair ID 作为保留清单，禁止训练 pair 重叠；
* 排除 inner-train 正例、原随机负例和已经生成的 hard pairs；
* 限制替换实体必须已存在于当前 inner-train 实体空间。

这里的 pair 保留清单只用于维持数据分区互斥，不参与相似度或标签判断。

## 5. 可复现性与审计

每折生成：

```text
<strict split dir>/training_negatives/
  mixed_seed_<seed>_ratio_<ratio>/
    fold_<index>.tsv
    fold_<index>.json
```

TSV 记录 `positive`、`random`、`hard_h_c` 或 `hard_p_d` 类型。JSON 记录：

* 输入训练清单哈希；
* 保留 pair 清单哈希；
* 最终 assignment 哈希；
* H-C/P-D 文件路径和 SHA-256；
* 请求及实际 hard ratio；
* H-C 与 P-D 来源数量；
* fold 和随机 seed。

相同输入集合、配置和 seed 不受文件读取顺序影响，会产生相同训练序列与 assignment 哈希。

## 6. TCMSP Fold 1 预检

在 TCMSP Strict fold 1 的 `80,786` 条 inner-train records 上，`hard_ratio=0.25` 得到：

| 项目 | 数量 |
|---|---:|
| 正例 | 40,393 |
| 训练负例总数 | 40,393 |
| 保留随机负例 | 30,295 |
| 结构困难未观测 pair | 10,098 |
| H-C 来源 | 9,599 |
| P-D 来源 | 499 |
| 实际 hard ratio | 24.999% |
| validation/outer-test pair 重叠 | 0 |

P-D 来源占比较低反映 TCMSP P-D 关系覆盖有限，实验报告必须保留该来源统计，不能将所有 hard pairs 表述为“疾病相似靶点负例”。

## 7. Pilot

运行命令：

```bash
./run_hdcti.sh configs/HDCTI_mixed_negative_pilot.conf
```

该配置只运行 TCMSP fold 1，并设置：

```ini
evaluation.fold.limit=1
evaluation.outer.test=False
```

首轮仅与相同 inner-validation 清单下的随机采样参考 `Validation-AUPR=0.983863` 比较。运行前固定以下判定规则：

* `AUPR >= 0.982863`：随机 validation AUPR 下降不超过 `0.001`，通过非劣性安全筛选；下一步增加固定困难候选和 Top-K 排名评价，不直接运行五折；
* `AUPR < 0.982863`：随机 validation 性能明显下降，停止当前 25% mixed 分支；不根据该 fold 继续搜索比例。

该随机 validation 指标只用于低成本安全初筛，不能证明困难候选排序已经改善。即使 mixed 策略通过，也必须在最终实验补充固定困难候选或 Top-K 排名指标，才能支持相应结论。

## 8. 解释边界

本模块生成的是“结构困难未观测 pair”，不是经实验验证的真负例。高相似 pair 反而更可能包含潜在真阳性，因此：

* 第一版只替换 25%，保留 75% 随机样本；
* 不使用“可靠生物学负例”表述；
* 不在同一个 validation fold 上反复搜索比例；
* 动态挖掘、低权重 hard-unlabeled loss 和 nnPU 留作后续独立方案。

## 9. Pilot 结果

TCMSP Strict fold 1 于 2026-07-14 完成：

| 项目 | Random 参考 | Mixed 25% |
|---|---:|---:|
| 最佳 validation AUPR | 0.983863 | 0.980860 |
| 最佳 epoch | 36 | 26 |
| 停止 epoch | 46 | 36 |
| 运行时间 | 443.390 s | 未单独归档 |

Mixed 相对 Random 下降 `0.003003`，并比预注册的非劣性下限 `0.982863` 低 `0.002003`，因此未通过安全筛选。该配置设置了 `evaluation.outer.test=False`，没有使用 outer-test 指标。

本次实际采样与预检一致：

```text
hard pairs: 10,098 / 40,393 (24.999%)
H-C source: 9,599
P-D source: 499
assignment SHA-256: 6d909d04f65396b1f1ea11e34a25d88850bac9d740c3e157cbf0a4812dbfa1c5
best checkpoint: ./saved_model/2026-07-14 19-58-43/hdcti_model.ckpt
```

按照运行前规则，当前 `25% mixed` 分支停止：

* 不运行 outer-test 或完整五折；
* 不在相同 fold 上继续搜索 hard ratio；
* 不把这些 pairs 表述为可靠负例；
* 主模型继续使用固定随机未观测样本。

结果表明，直接把 H-C/P-D 相似未观测 pairs 以标签 0 等权加入 BCE 可能引入潜在假负例。若后续继续研究未观测关系，应改用低权重 hard-unlabeled loss、可靠负例选择或 PU 风险，而不是继续调整当前静态替换比例。
