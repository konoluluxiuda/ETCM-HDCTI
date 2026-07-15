# 靶点条件化药材上下文残差注意力 V2

## 1. 研究动机

第一版 Target-conditioned Herb Attention 使用候选蛋白选择关联药材，并以动态上下文直接替换静态 Hctx-P 上下文。TCMSP fold 1 validation-only Pilot 的 AUPR 为 `0.982828`，低于静态 Hctx-P 的 `0.983863`，因此该替换式 V1 已按预注册规则停止。

V1 同时改变了两个变量：

```text
是否使用靶点条件化药材选择
是否保留已经验证有效的静态药材聚合
```

因此 V1 不能单独判断条件化选择是否能够在静态 Hctx-P 之外提供增量信息。V2 保留静态上下文，只让注意力学习候选级残差。

## 2. 方法定义

静态药材上下文记为：

$$
h_c^{static}=Aggregate\{e_h:h\in H(c)\}
$$

靶点条件化注意力与 V1 相同：

$$
\alpha_{h|c,p}=\operatorname{softmax}_{h\in H(c)}
\left(\frac{(W_he_h)^T(W_pz_p)}{\sqrt d}\right)
$$

$$
h_{c|p}^{att}=\sum_{h\in H(c)}\alpha_{h|c,p}e_h
$$

定义条件化上下文增量：

$$
\Delta h_{c|p}=h_{c|p}^{att}-h_c^{static}
$$

最终打分为：

$$
s(c,p)=z_c^Tz_p+
(h_c^{static}\odot w_{HP})^Tz_p+
(\Delta h_{c|p}\odot w_{TA})^Tz_p
$$

其中 $w_{TA}$ 是 V2 新增的维度级残差权重，并初始化为全零。由此可得：

$$
s_{V2}^{initial}(c,p)=s_{static}(c,p)
$$

这保证 V2 的首个前向传播严格等价于静态 Hctx-P，动态注意力只能学习附加修正，不能在初始化时删除静态信息。

## 3. 实现边界

新增模式：

```ini
context.herb_protein.mode=target_residual_attention
context.herb_attention.temperature=1.0
```

实现遵守以下边界：

1. `static`、替换式 `target_attention` 和残差式 `target_residual_attention` 三种模式独立保留。
2. 静态模式不创建任何注意力参数，继续兼容已有静态 checkpoint。
3. V1 不创建 V2 残差权重，其 checkpoint 和历史结论保持不变。
4. V2 的 `context_target_herb_residual` 使用零初始化并纳入原有权重正则。
5. 只在候选成分的 H-C incidence 内计算注意力，不读取 H-D 或测试 C-P 标签。
6. 不增加 MLP、多头注意力、温度搜索、熵正则或新的负样本策略。

训练后额外输出：

```text
Target herb residual weight mean abs
mean_delta_norm
mean_abs_logit
validation attention entropy / max weight
Top-3 herb attention TSV
```

`mean_abs_logit` 用于判断残差是否真正影响预测；权重或残差接近零时，应解释为模型回退到静态 Hctx-P，而不是把注意力图本身当作有效机制证据。

## 4. 预注册 Pilot

V2 不继续使用已经多次参与模型筛选的 TCMSP fold 1，而是在 ETCM2.0_core_mention10 fold 1 上进行一次 validation-only Pilot。固定配置为：

```text
Strict split: strict_seed_2026_k5
Outer fold: 1
Outer-test: disabled
Decoder: Dot
Temperature: 1.0
Maximum epoch: 50
Early stopping metric: inner-validation AUPR
```

静态 Hctx-P 在相同 ETCM fold 1 的最佳 validation AUPR 为 `0.975782`。运行前固定判定规则：

| V2 validation AUPR | 决策 |
|---:|---|
| `< 0.974782` | 劣于静态版本，停止 V2 |
| `0.974782 - 0.976781` | 未形成至少 0.001 的实质增益，停止完整五折 |
| `>= 0.976782` | 进入完整五折候选 |

即使达到性能门槛，也必须同时满足：无 NaN/CUDA 异常、注意力未整体坍缩、残差权重和 `mean_abs_logit` 不接近零。Pilot 不计算 outer-test 指标。

## 5. Pilot 结果与决策

```text
替换式 Target Attention V1：Pilot No-Go，已冻结
Residual Target Attention V2：Pilot 未形成实质增益，停止完整五折
```

ETCM2.0_core_mention10 fold 1 validation-only Pilot 于 2026-07-15 完成：

| 项目 | 结果 |
|---|---:|
| 静态 Hctx-P 参考 AUPR | 0.975782 |
| Residual Attention V2 AUPR | 0.974943 |
| 相对静态版本 | -0.000839 |
| 预注册停止下限 | 0.974782 |
| 相对停止下限 | +0.000161 |
| 进入完整五折门槛 | 0.976782 |
| 最佳 epoch | 30 |
| 早停 epoch | 40 |
| 运行时间 | 557.740369 s |

V2 高于预注册停止下限，但未达到静态参考，更未达到至少提高 `0.001` 的完整五折门槛。该结果属于“未明显劣化但没有实质增益”，按运行前规则停止 outer-test、完整五折和继续调参。

残差及注意力审计为：

```text
Hctx-P weight mean abs: 1.776822
Target residual weight mean abs: 1.598463
mean_delta_norm: 0.214344
mean_abs_residual_logit: 0.311660
validation_pairs: 14148
incidences: 34908
mean_entropy: 0.174099
mean_max_weight: 0.933917
```

残差权重和平均残差 logit 均不接近零，说明 V2 没有简单回退到静态 Hctx-P，而是学习了具有实际幅度的候选级修正。TSV 中 `14148` 个候选均有记录，其中 `5390` 个候选至少关联两个药材；这些多药材候选的最大注意力均值为 `0.826542`，最大权重不低于 `0.90/0.95/0.99` 的比例分别为 `48.50%/35.36%/14.40%`。注意力较尖锐，但当前不能在看到结果后追加温度搜索或熵正则来改变预注册模型。

最终结论：条件化注意力确实改变了 pair 打分，但 V1 和 V2 均未证明其优于静态 Hctx-P。当前第二创新分支到此关闭，静态 Hctx-P 继续作为主模型。

## 6. 实现与复核

配置文件：

```text
configs/HDCTI_etcm_mention10_target_residual_attention_pilot.conf
```

运行命令：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_target_residual_attention_pilot.conf
```

复核运行日志应包含：

```text
Herb context mode: target_residual_attention
Target herb residual weight mean abs
Target herb residual: mean_delta_norm=... mean_abs_logit=...
Pilot result for first 1 fold(s) of 5-fold cross validation
Validation-AUPR
```

不得出现 `Predicting [1]` 或 outer-test AUC/AUPR。
