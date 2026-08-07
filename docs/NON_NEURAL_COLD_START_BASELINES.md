# 非神经成分冷启动基线审计

## 1. 审计目的

本审计用于回答一个投稿前必须排除的替代解释：最终模型的冷启动结果是否只是
训练折蛋白流行度或简单药材—靶点共现带来的，而不需要神经网络表示学习。

审计完全复用 seed `52026` 的四库 compound cold-start 五折，不重新采样正负
样本，不执行模型训练，也不读取测试 C-P 标签构建分数。所有 C-P 统计均在当前
fold 的训练正边上重建。

运行入口：

```bash
./run_non_neural_cold_start_baselines.sh
```

机器可读结果默认写入：

```text
results/non_neural_cold_start_baselines/summary.json
results/non_neural_cold_start_baselines/summary.md
```

## 2. 基线定义

| 方法 | 输入 | 定义 |
|---|---|---|
| GlobalPrior | 训练折 C-P | 每个蛋白在受支持训练成分中的出现率 |
| HerbPrototype-EB | 训练折 C-P + 固定 H-C | 对测试成分的药材靶点后验取平均，先验强度为 1 |
| HC-Jaccard-LP | 训练折 C-P + 固定 H-C | 从 H-C Jaccard 相似的受支持训练成分传播 C-P 标签 |

当测试成分没有任何包含受支持训练成分的药材时，后两种方法回退到
`GlobalPrior`。由于当前任务是 compound cold-start，测试成分不存在训练 C-P
边，因此 HerbPrototype-EB 不会读取待评价成分自身标签。

## 3. 冻结结果

| 数据集 | GlobalPrior AUPR | HerbPrototype-EB AUPR | HC-Jaccard-LP AUPR | Ours-full AUPR | Ours 相对每库最佳启发式 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.776458 | 0.764420 | 0.698812 | 0.721718 | -0.054740 |
| TCMSP | 0.953035 | 0.959657 | 0.947788 | 0.957323 | -0.002334 |
| SymMap2.0 | 0.848328 | 0.857528 | 0.860923 | 0.837607 | -0.023316 |
| ETCM2.0-mention10 | 0.894044 | 0.949023 | 0.919744 | 0.913655 | -0.035368 |
| **Macro** | **0.867966** | **0.882657** | **0.856817** | **0.857576** | **-0.028939** |

药材证据覆盖率为 TCM-Suite `98.26%`、TCMSP `99.73%`、SymMap2.0
`99.33%` 和 ETCM2.0-mention10 `99.47%`。

逐折配对结果进一步表明，这不是少数异常 fold：

| 数据集 | Ours vs GlobalPrior | 正向 folds | Ours vs HerbPrototype-EB | 正向 folds | Ours vs HC-Jaccard-LP | 正向 folds |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | -0.054740 | 0/5 | -0.042702 | 0/5 | +0.022905 | 5/5 |
| TCMSP | +0.004288 | 5/5 | -0.002334 | 1/5 | +0.009535 | 5/5 |
| SymMap2.0 | -0.010721 | 0/5 | -0.019920 | 0/5 | -0.023315 | 0/5 |
| ETCM2.0-mention10 | +0.019611 | 5/5 | -0.035368 | 0/5 | -0.006089 | 1/5 |

## 4. 独立复核与偏倚诊断

使用不导入本评价器的独立字典实现重新计算 `GlobalPrior`，得到完全相同的
AUC/AUPR。训练折蛋白先验在测试正例与负例上的平均值为：

| 数据集 | 测试正例平均先验 | 测试负例平均先验 |
|---|---:|---:|
| TCM-Suite | 0.017541 | 0.004586 |
| TCMSP | 0.113599 | 0.003764 |
| SymMap2.0 | 0.041622 | 0.004654 |
| ETCM2.0-mention10 | 0.124100 | 0.015575 |

这说明当前固定 1:1 负样本虽然按 compound 匹配，却没有匹配 protein 流行度；
测试正例的靶点在训练折中显著更热门。高 AUPR 因而同时包含模型能力和靶点
流行度可分性。

## 5. 冻结结论

该审计为 **投稿阻塞项**。现有 cold-start 主表仍可作为历史的固定采样协议结果，
但不能单独支持“Ours-full 优于简单归纳策略”或“药材上下文神经表示是主要性能
来源”的结论。当前论文蓝图中关于投稿准备度的判断必须降级。

这项负结果不证明 Hctx-P、SDIS 或 SCHPT 完全无效：Ours-full 仍在部分数据库
优于 GlobalPrior 或 HC-Jaccard-LP，并且此前递进消融是在相同神经骨干内完成。
但从模型必要性角度，简单 HerbPrototype-EB 在四库 macro 上更强，必须进一步
改变或补充评价协议。

## 6. 下一步 Gate

优先执行不重新训练的评价修复：

1. 在完整 protein 候选集合上比较 Ours-full、GlobalPrior、HerbPrototype-EB 和
   HC-Jaccard-LP 的 MRR、Recall@K 与 Hits@K；
2. 另建 protein-degree-matched 测试负例，仅用于冻结 checkpoint 的纯推理压力
   测试，不据其重新选择模型；
3. 若 Ours-full 在完整候选排名中仍未优于简单基线，停止以当前模型投稿，不通过
   增加 seed 或统计检验掩盖模型必要性问题；
4. 只有通过排名 Gate 后，才补多 seed、置信区间和最终 cold-start 案例解释。

