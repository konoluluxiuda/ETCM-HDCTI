# 候选级 H-C/P-D 上下文交互

本文档记录 Strict-HDCTI 之后的第一版模型修改。Strict 基线 Git checkpoint 为 `96537e2`。

## 1. 修改边界

该模块只使用：

```text
H-C：候选成分所属药材上下文
P-D：候选蛋白关联疾病上下文
```

明确不使用：

```text
H-D 关系
测试 fold 的 C-P 正边
全节点 compound × protein 上下文特征张量
```

因此，它不会重新引入 H-D 来源审查中发现的闭包泄漏问题。Strict 模式下，C-P PageRank 仍然只使用当前 fold 的训练正边。

## 2. 上下文表示

设 HDCTI 编码后的成分和蛋白表示分别为 $z_c$、$z_p$，药材超边和疾病超边表示分别为 $e_h$、$e_d$。

对候选成分 $c$，使用归一化 H-C 关联聚合其药材上下文：

$$
h_c=\operatorname{norm}\left(\sum_{h\in H(c)}\bar A_{ch}e_h\right)
$$

对候选蛋白 $p$，使用归一化 P-D 关联聚合其疾病上下文：

$$
d_p=\operatorname{norm}\left(\sum_{d\in D(p)}\bar A_{pd}e_d\right)
$$

## 3. 因子化候选交互

第一版不构造逐 pair MLP 特征，而采用三个可学习的对角交互向量：

$$
\begin{aligned}
s_{cp}={}&z_c^Tz_p\\
&+(z_c\odot w_{CD})^Td_p\\
&+(h_c\odot w_{HP})^Tz_p\\
&+(h_c\odot w_{HD}^{latent})^Td_p
\end{aligned}
$$

最终概率：

$$
\hat y_{cp}=\sigma(s_{cp})
$$

其中 $w_{HD}^{latent}$ 只表示药材上下文向量与疾病上下文向量的潜在维度交互，**不是 H-D 数据边或 H-D 邻接矩阵**。

三个新增权重均从零初始化，所以第一次前向传播严格退化为原始点积解码器；训练随后学习各上下文项的贡献。

## 4. 为什么不直接使用 pair MLP

TCMSP 全候选空间约为：

```text
13,677 compounds × 1,748 proteins ≈ 23.9 million pairs
```

若为每个 pair 拼接多个 64 维向量，推理时会产生很大的临时特征张量。当前因子化形式可用四次矩阵乘法完成全量打分：

$$
S=Z_CZ_P^T+(Z_C\odot w_{CD})D_P^T+(H_C\odot w_{HP})Z_P^T+(H_C\odot w_{HD}^{latent})D_P^T
$$

其计算复杂度仍为 $O(|C||P|d)$，峰值输出内存与原始全量打分矩阵同阶，不额外保存 $|C||P|\times d$ 的 pair 特征。

## 5. 正则化修复

原实现把 embedding 正则写在 `for key in self.weights` 循环内部，导致成分和蛋白 embedding 正则被重复累加，次数随模型参数字典大小变化。新增模块参数后，这会使实际正则强度发生无关变化。

当前修正为：

$$
L_{reg}=\lambda_W\sum_k\frac{\lVert W_k\rVert_2^2}{2}
+\lambda_C\frac{\lVert Z_C\rVert_2^2}{2}
+\lambda_P\frac{\lVert Z_P\rVert_2^2}{2}
$$

每个 embedding 正则项只计算一次。`weight.reg` 控制 $\lambda_W$，`reg.lambda` 中的 `-u` 和 `-i` 分别控制成分与蛋白正则。

## 6. 配置

`context.interaction` 是总开关，三个打分项分别由独立开关控制：

```ini
context.interaction=True
context.compound_disease=True
context.herb_protein=True
context.herb_disease=True
weight.reg=0.01
```

对应关系：

| 配置项 | 打分项 |
|---|---|
| `context.compound_disease` | $z_c$ × $d_p$，即 C-Dctx |
| `context.herb_protein` | $h_c$ × $z_p$，即 Hctx-P |
| `context.herb_disease` | $h_c$ × $d_p$，即 Hctx-Dctx |

兼容规则：

* `context.interaction=False` 会覆盖三个子开关并关闭全部上下文项；
* 总开关为 `True` 且某个子开关未配置时，该项默认开启，因此旧版 Context v1 配置仍等价于三项全开；
* 布尔值只接受 `True/False`、`1/0`、`yes/no` 或 `on/off`，其他值直接报错；
* 关闭项不进入训练 logits，也不进入 NumPy 全量排名分数；其零初始化权重不会得到更新。

常用实验组合：

| 变体 | C-Dctx | Hctx-P | Hctx-Dctx |
|---|---:|---:|---:|
| `w/o Context` | off | off | off |
| `DiseaseOnly` | on | off | off |
| `HerbOnly` | off | on | off |
| `DirectCrossOnly` | off | off | on |
| `w/o Direct Cross` | on | on | off |
| `Context v1 full` | on | on | on |

当前活动配置采用比较后保留的 HerbOnly：

```ini
model.variant=context_herb_only_v1
context.interaction=True
context.compound_disease=False
context.herb_protein=True
context.herb_disease=False
```

将总开关设为 `False` 只会得到修复后的 `w/o Context`，正则化修复仍然生效。严格复现修复前的 Strict checkpoint 应使用 Git commit `96537e2`，不能只关闭配置后混称为同一实现。

## 7. 验证

组件测试覆盖：

1. 矩阵化上下文得分与逐 pair 公式一致。
2. 三个上下文权重为零时，得分严格等于点积基线。
3. embedding 正则不随权重数量重复累加。
4. tiny Strict 数据上能够完成一次前向与优化，且上下文权重从零得到更新。
5. 三个上下文项可独立解析和关闭，非法配置会报错。
6. 关闭项不会进入全量排名，且单步训练后仍保持零权重。

运行：

```bash
python -m unittest discover -s tests -v
```

这些测试只验证接口、公式和数值有限性，不替代一次正式五折性能实验。

## 8. GPU 稳定性修复

首次 TCMSP 正式运行在第 34 个 epoch 出现：

```text
CUDA_ERROR_ILLEGAL_ADDRESS
Unexpected Event status: 1
node MatMul_3
```

当时 loss 为有限值 `34.464615`，日志中也没有 BFC allocator 的 OOM 报告，因此该故障不属于普通的 loss 发散或显存不足。按 TensorFlow 计算图的建图顺序，`MatMul_3` 对应第一层 H-C 超图的“药材超边到成分节点”传播；它不是新增的候选上下文交互项。

原实现先把 H-C/P-D 稀疏关联矩阵整体转换为稠密张量，再向 `tf.matmul` 传入 `a_is_sparse=True`。这会增加内存占用，并让 TensorFlow 对实际为稠密的输入采用稀疏优化提示。当前改为：

```text
tf.sparse_tensor_dense_matmul(SparseTensor, DenseTensor)
```

该修改覆盖两层 H-C/P-D 超图传播及候选上下文聚合，并为关键算子增加稳定名称。关联矩阵、归一化方式、传播公式和可训练参数均未改变，因此它属于等价的实现与稳定性修复，不是新的模型模块。

TF 2.21 环境下的 tiny Strict 前向和单步反向测试已经通过。由于 RTX 5060 Ti 的 compute capability 12.0a 仍由当前 TensorFlow wheel 通过 PTX JIT 生成内核，若后续仍出现非法地址，应优先记录新的具名算子；其次再单独测试全节点 self-attention，而不是把故障直接归因于新增上下文交互。

`run_hdcti.sh` 会显式执行 `unset TF_GPU_ALLOCATOR`，防止当前 shell 中曾经导出的 `cuda_malloc_async` 设置残留到后续运行。该分配器曾在本机 TensorFlow/WSL 组合中触发段错误，因此不作为默认配置。

## 9. TCMSP 首轮五折结果

运行于 2026-07-14，采用固定 50 epoch，不使用早停。主要配置为：

```ini
datapath=./dataset/TCMSP/one1.txt
model.variant=context_interaction_v1
experiment.protocol=strict
random.seed=2026
split.reuse=True
context.interaction=True
weight.reg=0.01
num.factors=64
num.max.epoch=50
batch_size=2000
```

`attention.max.nodes` 未启用，因此保留 full self-attention。Strict split manifest 与 2026-07-13 基线相同，测试 C-P 正边不参与每折 PageRank 或训练图构建。

Fold 5：

| 指标 | 数值 |
|---|---:|
| AUC | 0.987857701742178 |
| AUPR | 0.984012739801992 |
| Recall | 0.9624777183600713 |
| Precision | 0.9552410437859354 |
| F1-score | 0.9588457269700333 |

五折汇总：

| 指标 | Context Interaction v1 |
|---|---:|
| AUC | 0.987372 (±0.000944) |
| AUPR | 0.983825 (±0.001173) |
| Recall | 0.958522 (±0.002978) |
| Precision | 0.953921 (±0.002844) |
| F1-score | 0.956212 (±0.001986) |

Fold 5 最后一批 loss 为 `27.659155`。Fold 5 学到的上下文权重平均绝对值为：

| 交互项 | mean(abs(weight)) |
|---|---:|
| Compound–Disease context | 2.120275 |
| Herb context–Protein | 0.823767 |
| Herb context–Disease context | 0.233718 |

上述权重大小只能说明参数尺度，不能直接等同于各交互项对预测的贡献；不同输入表示的尺度和分布并不相同。后续应比较各打分项在测试候选上的平均绝对贡献，或使用 `HerbOnly`、`DiseaseOnly` 与 `w/o Context` 消融。

与 2026-07-13 Strict-HDCTI 基线比较：

| 指标 | Strict 基线 | Context v1 | Context v1 - Strict |
|---|---:|---:|---:|
| AUC | 0.985893 | 0.987372 | +0.001479 |
| AUPR | 0.982425 | 0.983825 | +0.001400 |
| Recall | 0.976275 | 0.958522 | -0.017753 |
| Precision | 0.939059 | 0.953921 | +0.014862 |
| F1-score | 0.957303 | 0.956212 | -0.001091 |

运行时间为 `2305.836417 s`（约 `38.43 min`），Fold 5 checkpoint 为：

```text
./saved_model/2026-07-14 13-13-02/hdcti_model.ckpt
```

### 结果解释

1. AUC 和 AUPR 同时提高，说明该候选版本的整体排序能力优于已归档 Strict 基线。
2. 固定 `0.5` 阈值下 Precision 明显提高而 Recall 明显下降，说明输出分布或校准发生变化，模型更偏向保守地预测正例；F1 因此略低于 Strict 基线。
3. 本轮同时包含候选上下文交互和 embedding 正则修复；稀疏传播属于数学等价的稳定性修复。因此当前结果证明“修复后候选版本值得继续”，但不能把性能差值全部归因于上下文交互。
4. 下一项必要对照是保持正则修复和其余代码完全一致，仅设置 `context.interaction=False` 的 `w/o Context`。该对照用于隔离上下文模块贡献，不需要重新验证旧 Strict 数据协议。
5. 不根据本轮外层测试结果调整分类阈值。若后续需要阈值优化或早停，只能在训练折内部的 validation 上完成。

五折标准差仍然只反映 fold 差异，不代表多随机种子方差。

## 10. 修复后 w/o Context 对照

运行于 2026-07-14，使用与 Context Interaction v1 相同的 Strict split、seed、正则化、50 epoch、batch size 和 full self-attention，仅修改：

```ini
context.interaction=False
```

这会关闭三个候选上下文打分项，但保留 embedding 正则修复、稀疏超图传播和其他稳定性修复，因此是隔离上下文模块贡献的匹配对照。

Fold 5：

| 指标 | 数值 |
|---|---:|
| AUC | 0.984605150275959 |
| AUPR | 0.9803213261459471 |
| Recall | 0.9762032085561497 |
| Precision | 0.937676568786919 |
| F1-score | 0.9565521156281385 |

五折汇总及配对配置差值：

| 指标 | w/o Context | Context v1 | Context v1 - w/o Context |
|---|---:|---:|---:|
| AUC | 0.983508 (±0.001183) | 0.987372 (±0.000944) | +0.003864 |
| AUPR | 0.978492 (±0.001944) | 0.983825 (±0.001173) | +0.005333 |
| Recall | 0.974386 (±0.001094) | 0.958522 (±0.002978) | -0.015864 |
| Precision | 0.934423 (±0.003798) | 0.953921 (±0.002844) | +0.019498 |
| F1-score | 0.953984 (±0.002325) | 0.956212 (±0.001986) | +0.002228 |

Fold 5 最后一批 loss 为 `53.675636`。运行时间为 `2302.556250 s`（约 `38.38 min`），Fold 5 checkpoint 为：

```text
./saved_model/2026-07-14 13-56-27/hdcti_model.ckpt
```

Context v1 的运行时间为 `2305.836417 s`，仅比 w/o Context 增加 `3.280167 s`（约 `0.14%`），表明当前因子化上下文打分没有形成明显的运行时间负担。

### 对照结论

1. 在修复后代码和完全一致的实验协议下，候选上下文交互提高了 AUC、AUPR 和 F1，因此其收益不再能由 embedding 正则修复解释。
2. Precision 提高 `0.019498`、Recall 降低 `0.015864`，再次确认上下文模块使固定 `0.5` 阈值下的预测更保守；但相对匹配对照，F1 仍净提高 `0.002228`。
3. 该轮结果当时支持将完整 Context v1 保留到单项筛选阶段；后续 `HerbOnly` 与 `DiseaseOnly` 对照已经完成，并最终只保留 Hctx-P。
4. 目前只有五折汇总和 fold 标准差；若要报告配对显著性检验，需要保存并使用五个 fold 的逐折指标，不能根据标准差是否重叠代替显著性检验。
5. 不使用外层测试结果调整 `0.5` 阈值。阈值校准和早停仍应等待训练折内部 validation 协议。

已从两次运行保存的 `results/cv` 预测文件恢复逐 fold 配对差值：

| Fold | ΔAUC | ΔAUPR | ΔRecall | ΔPrecision | ΔF1 |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.004251 | +0.006408 | -0.018091 | +0.019691 | +0.001262 |
| 2 | +0.004206 | +0.005859 | -0.013635 | +0.019240 | +0.003220 |
| 3 | +0.003699 | +0.004676 | -0.018360 | +0.019350 | +0.000815 |
| 4 | +0.003915 | +0.006031 | -0.015508 | +0.021647 | +0.003553 |
| 5 | +0.003253 | +0.003691 | -0.013725 | +0.017564 | +0.002294 |

AUC、AUPR、Precision 和 F1 在五个 fold 上均为正增益，Recall 在五个 fold 上均下降，因此均值提升并非由单个异常 fold 驱动。但五个外层 fold 的训练集彼此重叠，不能把它们当成五次完全独立实验；正式显著性结论仍需多训练 seed 或适当的配对重采样分析。

固定阈值 `0.5` 下，五折全部 `112,204` 个测试样本的混淆矩阵变化为：

| 统计量 | w/o Context | Context v1 | 差值 |
|---|---:|---:|---:|
| TP | 54,665 | 53,775 | -890 |
| FP | 3,837 | 2,598 | -1,239 |
| FN | 1,437 | 2,327 | +890 |
| TN | 52,265 | 53,504 | +1,239 |

在当前平衡采样测试集上，该模块以少找回 `890` 个正例为代价，减少了 `1,239` 个假阳性。若任务目标是输出少量高可信候选靶点，这种 Precision–Recall 交换可能具有实际价值；若目标是尽可能找全潜在靶点，则需要在内层验证集上进行阈值校准或继续改进 Recall。该结论不能直接外推到真实未知正负比例。

## 11. w/o Direct Cross 对照

运行于 2026-07-14，使用：

```ini
model.variant=context_no_direct_cross_v1
context.interaction=True
context.compound_disease=True
context.herb_protein=True
context.herb_disease=False
```

它保留 C-Dctx 和 Hctx-P，只关闭直接的 Hctx-Dctx。其他数据、固定划分、seed、正则化和训练配置均与 Full Context v1 相同。

Fold 5：

| 指标 | 数值 |
|---|---:|
| AUC | 0.9876613691174086 |
| AUPR | 0.983610249716917 |
| Recall | 0.9631016042780749 |
| Precision | 0.9545936395759718 |
| F1-score | 0.9588287488908607 |

五折汇总：

| 指标 | w/o Direct Cross |
|---|---:|
| AUC | 0.987295 (±0.001200) |
| AUPR | 0.983780 (±0.001530) |
| Recall | 0.958611 (±0.003525) |
| Precision | 0.953235 (±0.003037) |
| F1-score | 0.955910 (±0.001977) |

Fold 5 上下文权重平均绝对值为 C-Dctx `2.135056`、Hctx-P `0.816721`，Hctx-Dctx 关闭。运行时间为 `2463.348743 s`（约 `41.06 min`），Fold 5 checkpoint 为：

```text
./saved_model/2026-07-14 15-07-03/hdcti_model.ckpt
```

与 Full Context v1 配对比较：

| 指标 | w/o Direct Cross | Full Context | Full - w/o Direct Cross |
|---|---:|---:|---:|
| AUC | 0.987295 | 0.987372 | +0.000077 |
| AUPR | 0.983780 | 0.983825 | +0.000045 |
| Recall | 0.958611 | 0.958522 | -0.000089 |
| Precision | 0.953235 | 0.953921 | +0.000686 |
| F1-score | 0.955910 | 0.956212 | +0.000302 |

逐 fold 中，Full 相对 w/o Direct Cross 的 AUC 在 3/5 折提高，AUPR 在 2/5 折提高，Precision 在 5/5 折提高，F1 在 4/5 折提高。固定阈值下，Full 累计减少 `41` 个 FP，但同时减少 `5` 个 TP。直接 Hctx-Dctx 项主要产生极小的 Precision 增益，对排序指标没有一致贡献。

### 取舍结论

1. Full 与 w/o Direct Cross 的 AUC/AUPR 差值小于 `0.0001`，F1 差值约 `0.0003`，不足以证明直接 Hctx-Dctx 带来实质性改进。
2. 该轮中间结论曾按简约原则将 `C-Dctx + Hctx-P` 作为候选结构；后续单项比较进一步确认应只保留 Hctx-P。
3. 该结构选择使用了 TCMSP 外层五折结果，只能视为开发阶段决策。最终无偏结果应在训练折内部 validation 确定结构后，再在未参与选择的数据集或预留测试协议上报告。
4. 后续已使用独立开关完成 `DiseaseOnly` 与 `HerbOnly` 比较；结果见下一节，不再围绕 Hctx-Dctx 做更多长时间实验。

## 12. DiseaseOnly 与 HerbOnly 比较

两个变体使用独立配置文件，除单项上下文开关和 `model.variant` 外，其余设置与 `w/o Direct Cross` 完全一致：

```text
configs/HDCTI_disease_only.conf
configs/HDCTI_herb_only.conf
```

运行命令：

```bash
./run_hdcti.sh configs/HDCTI_disease_only.conf
./run_hdcti.sh configs/HDCTI_herb_only.conf
```

不要在同一块 GPU 上并行执行。启动日志必须分别显示：

```text
DiseaseOnly: C-Dctx=on,  Hctx-P=off, Hctx-Dctx=off
HerbOnly:    C-Dctx=off, Hctx-P=on,  Hctx-Dctx=off
```

`main.py` 支持 `--config`，`run_hdcti.sh` 的第一个参数会传入配置路径。逐 fold 预测和五折汇总文件名会包含 `model.variant`，例如：

```text
results/cv/context_disease_only_v1@<timestamp>[1].txt
results/context_disease_only_v1@<timestamp>-5-fold-cv.txt
```

完成后的比较表：

| 指标 | w/o Context | DiseaseOnly | HerbOnly | C-Dctx + Hctx-P |
|---|---:|---:|---:|---:|
| AUC | 0.983508 | 0.982299 | **0.988480** | 0.987295 |
| AUPR | 0.978492 | 0.978216 | **0.985436** | 0.983780 |
| Recall | **0.974386** | 0.970358 | 0.955100 | 0.958611 |
| Precision | 0.934423 | 0.937367 | **0.958348** | 0.953235 |
| F1-score | 0.953984 | 0.953575 | **0.956720** | 0.955910 |

解释规则：

1. 分别比较 DiseaseOnly/HerbOnly 与 `w/o Context`，判断单项是否独立提供增益。
2. 比较两项单独结果与 `C-Dctx + Hctx-P`，判断两侧信息是否互补。
3. 若某个单项不优于 `w/o Context`，但双项组合更好，说明其价值可能来自协同而非独立贡献。
4. 保留逐 fold 预测做配对比较，不用 fold 标准差重叠替代显著性检验。

### DiseaseOnly 结果

DiseaseOnly 已于 2026-07-14 完成，Fold 5 checkpoint 为：

```text
./saved_model/2026-07-14 16-42-31/hdcti_model.ckpt
```

Fold 5 的 C-Dctx 权重平均绝对值为 `2.066902`，Hctx-P 和 Hctx-Dctx 均关闭。五折结果为：

| 指标 | DiseaseOnly |
|---|---:|
| AUC | 0.982299 (±0.001296) |
| AUPR | 0.978216 (±0.001826) |
| Recall | 0.970358 (±0.001877) |
| Precision | 0.937367 (±0.003236) |
| F1-score | 0.953575 (±0.002080) |

运行时间为 `2616.448082 s`（约 `43.61 min`）。与修复后 `w/o Context` 比较：

| 指标 | w/o Context | DiseaseOnly | DiseaseOnly - w/o Context |
|---|---:|---:|---:|
| AUC | 0.983508 | 0.982299 | -0.001209 |
| AUPR | 0.978492 | 0.978216 | -0.000276 |
| Recall | 0.974386 | 0.970358 | -0.004028 |
| Precision | 0.934423 | 0.937367 | +0.002944 |
| F1-score | 0.953984 | 0.953575 | -0.000409 |

逐 fold 中，DiseaseOnly 的 AUC 在 0/5 折优于 `w/o Context`，AUPR 在 2/5 折提高，Recall 在 0/5 折提高，Precision 在 5/5 折提高，F1 仅在 1/5 折提高。固定阈值下累计减少 `199` 个 FP，但同时损失 `226` 个 TP。

阶段结论：C-Dctx 单独使用没有提供排序收益，只产生较小的 Precision–Recall 交换。双项 `C-Dctx + Hctx-P` 相对 DiseaseOnly 的 AUC、AUPR 和 F1 分别提高 `0.004996`、`0.005564` 和 `0.002335`。HerbOnly 的后续结果表明，该差值主要来自 Hctx-P，而不是两个单项之间的正向协同。

### HerbOnly 结果

HerbOnly 已于 2026-07-14 完成，Fold 5 checkpoint 为：

```text
./saved_model/2026-07-14 17-41-21/hdcti_model.ckpt
```

Fold 5 的 Hctx-P 权重平均绝对值为 `1.079263`，C-Dctx 和 Hctx-Dctx 均关闭。五折结果为：

| 指标 | HerbOnly |
|---|---:|
| AUC | 0.988480 (±0.000973) |
| AUPR | 0.985436 (±0.001741) |
| Recall | 0.955100 (±0.001104) |
| Precision | 0.958348 (±0.001682) |
| F1-score | 0.956720 (±0.001002) |

运行时间为 `2584.539353 s`（约 `43.08 min`）。与修复后 `w/o Context` 比较：

| 指标 | w/o Context | HerbOnly | HerbOnly - w/o Context |
|---|---:|---:|---:|
| AUC | 0.983508 | 0.988480 | +0.004972 |
| AUPR | 0.978492 | 0.985436 | +0.006944 |
| Recall | 0.974386 | 0.955100 | -0.019286 |
| Precision | 0.934423 | 0.958348 | +0.023925 |
| F1-score | 0.953984 | 0.956720 | +0.002736 |

逐 fold 中，HerbOnly 的 AUC、AUPR、Precision 和 F1 均在 5/5 折优于 `w/o Context`，Recall 在 0/5 折提高。固定阈值下累计减少 `1,508` 个 FP，同时损失 `1,082` 个 TP。这说明 Hctx-P 不只是改变阈值行为，也稳定提高了与阈值无关的排序指标。

与双项 `C-Dctx + Hctx-P` 比较，HerbOnly 的 AUC、AUPR、Precision 和 F1 分别提高 `0.001186`、`0.001656`、`0.005112` 和 `0.000810`，Recall 降低 `0.003511`。AUC、AUPR 和 Precision 均为 5/5 折提高，F1 为 3/5 折提高。加入 C-Dctx 没有形成正向协同，反而削弱了 Hctx-P 的排序收益。

最终结论：当前 TCMSP Strict 结果支持保留 **HerbOnly（Hctx-P）**，删除 C-Dctx 和 Hctx-Dctx。该结论基于同一组固定五折的配对方向与效应大小；五个交叉验证 fold 不是五次独立重复，因此不把 fold 标准差或小样本显著性检验解释为训练随机性证据。最终投稿结果仍需在预先选定的外部数据集或多 seed 实验中验证。

## 冻结决定

2026-07-15 完成 ETCM2.0_core_mention10 的 Max-80 收敛对照、checkpoint 分组机制分析和 inner-validation 阈值校准后，静态 Hctx-P 正式冻结。后续不再向该版本加入 H-C/mention gate、degree-aware calibration、C-Dctx 或 Hctx-Dctx；固定 `0.5` 结果保留用于原协议比较，验证集阈值结果只作为补充校准分析。

第二创新使用独立 `target_attention` 模式尝试靶点条件化药材上下文注意力。关闭新模式时必须保持本节静态 Hctx-P 的图结构、参数和 checkpoint 兼容性。设计见 [TARGET_CONDITIONED_HERB_ATTENTION.md](TARGET_CONDITIONED_HERB_ATTENTION.md)。
