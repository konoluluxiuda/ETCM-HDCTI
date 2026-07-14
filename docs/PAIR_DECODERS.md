# Pair Decoder 对照

## 1. 目的与控制变量

在已保留的 HerbOnly（Hctx-P）结构上，只替换基础 compound-protein 匹配函数：

```text
Dot
Bilinear
Residual MLP
```

H-C/P-D 编码器、Strict inner split、validation AUPR、早停参数、随机种子、负样本和 Hctx-P 打分项保持不变。最终 logits 为：

$$
s(c,p)=s_{decoder}(z_c,z_p)+s_{Hctx-P}(h_c,z_p)
$$

因此 decoder 对照不会重新引入已删除的 C-Dctx 或 Hctx-Dctx。

## 2. 三种 Decoder

### Dot

$$
s_{dot}(c,p)=z_c^Tz_p
$$

配置：

```ini
pair.decoder=dot
```

### Bilinear

$$
s_{bilinear}(c,p)=z_c^TWz_p
$$

$W$ 使用单位矩阵初始化，因此第一次前向传播严格等于 Dot；训练后允许不同嵌入维度发生可学习交互。

```ini
pair.decoder=bilinear
```

### Residual MLP

构造 pair feature：

$$
x_{cp}=[z_c;z_p;z_c\odot z_p;|z_c-z_p|]
$$

使用残差 MLP：

$$
s_{mlp}(c,p)=z_c^Tz_p+w_2^T\operatorname{LeakyReLU}(W_1x_{cp}+b_1)+b_2
$$

$w_2$ 和 $b_2$ 从零初始化，因此第一次前向传播同样严格等于 Dot。当前隐藏维度为 64：

```ini
pair.decoder=mlp
pair.mlp.hidden=64
```

三种 decoder 的新增参数都进入现有 `weight.reg` L2 正则。

## 3. Pair-only 评估

训练、validation 和 outer-test 都只对清单中的 C-P pairs 计算 logits。尤其是 MLP，不会构造：

```text
compound_count x protein_count x pair_feature_dimension
```

的稠密三维张量。`predictForRanking()` 仍保留完整排名兼容接口，MLP 会按 `pair.prediction.batch.size` 分批计算；交叉验证默认使用更高效的 `predictForPairs()`。

## 4. 无泄漏模型选择

decoder pilot 使用：

```ini
evaluation.fold.limit=1
evaluation.outer.test=False
```

训练结束后只输出最佳 inner-validation AUPR 和 epoch，不评估 outer-test。选择规则预先固定为：

1. 首先比较同一 fold、同一 split 下的最佳 validation AUPR；
2. 改进小于 `0.0001` 时优先选择参数更少的 decoder；
3. 不使用 outer-test AUC/AUPR 选择 decoder；
4. 选定后再为匹配基线和最终 decoder 运行完整外层评估。

已完成的 Dot early-stop pilot 最佳 validation AUPR 为 `0.983863`（epoch 36），可直接作为 Dot 参考，不需要重复运行。

## 5. 运行命令

Bilinear：

```bash
./run_hdcti.sh configs/HDCTI_decoder_bilinear_pilot.conf
```

MLP：

```bash
./run_hdcti.sh configs/HDCTI_decoder_mlp_pilot.conf
```

Dot 配置保留用于复查：

```bash
./run_hdcti.sh configs/HDCTI_decoder_dot_pilot.conf
```

三个配置均复用 TCMSP Strict fold 1 的 `80,786` 条 inner-train 和 `8,976` 条 validation。输出文件明确标注为 pilot，不能作为完整五折结果。

## 6. 实现验证

当前测试覆盖：

* decoder 配置默认值和非法值检查；
* Bilinear 单位矩阵初始化等价于 Dot；
* MLP 零残差初始化等价于 Dot；
* TensorFlow Bilinear 单步优化；
* TensorFlow MLP 训练、早停、checkpoint 恢复；
* MLP pair-only 分数与分批完整排名对应位置一致。

## 7. Pilot 结果

三个 decoder 使用同一 TCMSP Strict fold 1、相同 inner split、随机种子和早停协议，仅比较最佳 validation AUPR。

| Decoder | 最佳 validation AUPR | 最佳 epoch | 停止 epoch | 运行时间 | 筛选结论 |
|---|---:|---:|---:|---:|---|
| Dot | 0.983863 | 36 | 46 | 443.390 s | 当前参考 |
| Bilinear | 0.980768 | 12 | 22 | 218.879 s | 淘汰 |
| MLP | 0.974163 | 2 | 12 | 122.180 s | 淘汰 |

Bilinear 相对 Dot 的 validation AUPR 下降 `0.003095`，明显超过预设的 `0.0001` 近似持平阈值。虽然它更早停止、运行时间更短，但主要模型选择指标下降，因此不进入外层测试或完整五折。

Bilinear 最佳 checkpoint：

```text
./saved_model/2026-07-14 19-26-18/hdcti_model.ckpt
```

该结果只用于 decoder 结构筛选，`evaluation.outer.test=False`，没有使用外层测试折。日志中的 Hctx-P 平均绝对权重为 `0.320451`；由于 decoder 改变了基础打分函数的参数化和尺度，该数值不能单独解释为上下文贡献降低。

MLP 相对 Dot 的 validation AUPR 下降 `0.009700`，相对 Bilinear 下降 `0.006605`。它在 epoch 2 达到最佳值，随后验证性能下降并于 epoch 12 早停，表明当前残差 MLP 在冻结协议下没有带来有效增益。其最佳 checkpoint 为：

```text
./saved_model/2026-07-14 19-32-25/hdcti_model.ckpt
```

MLP 同样只用于结构筛选，没有执行外层测试。Hctx-P 平均绝对权重为 `0.094328`，也不应脱离整体打分尺度单独解释。

## 8. 筛选结论

在预先固定的 TCMSP fold 1 inner-validation 协议下，排序为：

```text
Dot (0.983863) > Bilinear (0.980768) > MLP (0.974163)
```

因此当前主模型保留参数最少、验证性能最高的 Dot decoder。Bilinear 和 MLP 不再运行外层测试或完整五折，也不通过反复调整同一个 validation fold 继续筛选。

下一步在完全相同的早停协议下，对匹配基线和最终 HerbOnly 模型分别运行完整五折：

```bash
./run_hdcti.sh configs/HDCTI_no_context_early_stop.conf
./run_hdcti.sh configs/HDCTI_herb_only_early_stop.conf
```
