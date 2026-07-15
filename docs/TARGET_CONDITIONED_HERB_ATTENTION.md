# 靶点条件化药材上下文注意力

## 1. 研究定位

当前静态 Hctx-P 已冻结为第一创新版本。它将候选成分关联的药材超边聚合为固定上下文 $h_c$，再与候选蛋白表示 $z_p$ 显式匹配：

$$
s_{static}(c,p)=z_c^Tz_p+(h_c\odot w_{HP})^Tz_p
$$

静态聚合的限制是：同一成分面对所有候选蛋白时使用完全相同的药材上下文。第二创新优先尝试 **Target-conditioned Herb Context Attention**，让候选蛋白选择与当前 C-P pair 更相关的药材超边。

该模块是 Hctx-P 的候选级细化，不是通用局部—全局多视图、H-D 路径或全节点注意力。

## 2. 方法定义

设成分 $c$ 关联的药材集合为 $H(c)$，药材超边表示为 $e_h$，候选蛋白表示为 $z_p$。注意力分数定义为：

$$
a_{h|c,p}=\frac{(W_he_h)^T(W_pz_p)}{\sqrt d}
$$

只在当前候选成分关联的药材集合内归一化：

$$
\alpha_{h|c,p}=\operatorname{softmax}_{h\in H(c)}(a_{h|c,p})
$$

得到 pair-specific 药材上下文：

$$
h_{c|p}=\sum_{h\in H(c)}\alpha_{h|c,p}e_h
$$

最终分数保留当前 Dot 基础解码器：

$$
s(c,p)=z_c^Tz_p+(h_{c|p}\odot w_{HP})^Tz_p
$$

第一版不加入额外 MLP、H-D、P-D 上下文交互、对比损失或困难负样本，确保只检验靶点条件化药材选择本身。

## 3. 冻结与兼容边界

静态 Hctx-P 保持当前默认行为和参数命名，不直接改写。新增模式计划使用：

```text
context.herb_protein.mode=static
context.herb_protein.mode=target_attention
```

必须满足：

1. 未配置该选项时等价于 `static`。
2. `static` 模式能够加载现有 Hctx-P checkpoint。
3. `target_attention` 使用独立配置、`model.variant` 和 checkpoint 目录。
4. 关闭新模式时，现有 Hctx-P 单元测试和预测结果不受影响。
5. 不读取 H-D，不使用完整 C-P 图构造注意力。

## 4. 无泄漏约束

注意力只允许使用：

```text
当前 fold 固定的 H-C 侧信息
候选成分关联的药材超边表示
模型编码得到的候选蛋白表示
当前 fold inner-train C-P 监督标签
```

禁止使用：

```text
outer-test C-P 标签
完整 C-P 派生 H-D
静态完整 C-P-C / P-C-P 视图
outer-test 指标选择注意力参数或阈值
```

## 5. 计算与解释输出

注意力仅在 $H(c)$ 内计算，复杂度随 batch 中实际 H-C incidence 数增长，不构造全体 compound、herb 或 protein 的稠密两两注意力矩阵。

实现后至少记录：

```text
每个 batch 的展开 H-C incidence 数
药材注意力熵均值
每个 pair 的 Top-3 药材及权重
训练时间与峰值显存
```

药材权重只能解释为模型对候选 pair 的结构贡献，不能直接表述为临床机制或因果证据。

## 6. Pilot 协议

首轮只运行 TCMSP Strict fold 1，使用与静态 HerbOnly 相同的 split、seed、Dot decoder、inner validation 和 early stopping，并设置：

```text
evaluation.pilot.folds=1
evaluation.outer.test=False
```

静态 Hctx-P 参考 validation AUPR 为 `0.983863`。预先规定：

| Target-attention validation AUPR | 决策 |
|---:|---|
| `< 0.982863` | 明显劣于静态版本，停止 |
| `0.982863 - 0.984862` | 未形成实质改进，停止完整五折 |
| `>= 0.984863` | 至少提高 0.001，进入完整五折候选 |

即使通过 AUPR 门槛，也必须确认无 NaN/CUDA 错误、显存可接受且注意力没有完全坍缩到固定单一药材。Pilot 不执行 outer-test，不能作为最终模型效果。

## 7. Pilot 结果与决策

```text
静态 Hctx-P：已冻结
Target-conditioned attention：Pilot 未通过，停止完整五折
```

TCMSP Strict fold 1 的 validation-only Pilot 于 2026-07-15 完成：

| 项目 | 结果 |
|---|---:|
| 静态 Hctx-P 参考 AUPR | 0.983863 |
| Target-attention AUPR | 0.982828 |
| 相对静态版本 | -0.001035 |
| 预注册非劣性下限 | 0.982863 |
| 相对非劣性下限 | -0.000035 |
| 最佳 epoch | 28 |
| 早停 epoch | 38 |
| 运行时间 | 438.181101 s |

结果以 `0.000035` 的微小差距低于预注册非劣性下限。虽然该差距不足以说明模块存在明显伤害，但它没有提供优于静态 Hctx-P 的证据。按照运行前规则，不事后放宽阈值，不搜索温度或额外结构，也不运行 outer-test 和完整五折。静态 Hctx-P 继续作为当前主模型。

注意力审计输出为：

```text
validation_pairs=8976
incidences=38969
mean_entropy=0.345131
mean_max_weight=0.864964
```

`mean_max_weight` 会受到单药材候选的影响。进一步按 TSV 中的候选分组后，具有至少两个已记录药材的 `3361` 个候选，其最大权重均值为 `0.646807`；最大权重不低于 `0.90/0.95/0.99` 的比例分别为 `23.18%/17.58%/8.93%`。因此没有证据表明所有候选都坍缩到固定单一药材，但部分候选的注意力较尖锐。当前 No-Go 的主要原因是预测性能没有超过静态聚合，而不是明显的全局注意力坍缩。

该实现和配置保留为可复核的负结果及后续案例分析工具，不进入当前论文主模型贡献。

## 8. 实现与复核

实现配置：

```text
configs/HDCTI_target_herb_attention_pilot.conf
```

运行命令：

```bash
./run_hdcti.sh configs/HDCTI_target_herb_attention_pilot.conf
```

预期日志必须包含：

```text
Herb context mode: target_attention
Pilot result for first 1 fold(s) of 5-fold cross validation
Validation-AUPR
```

并且不能出现 `Predicting [1]` 或 outer-test AUC/AUPR。训练结束后会输出 validation pair 的注意力熵、平均最大权重，以及：

```text
results/target_attention_top_herbs<timestamp>.tsv
```

该文件保存每个 validation pair 权重最高的三个关联药材，仅用于结构审计和后续案例解释。
