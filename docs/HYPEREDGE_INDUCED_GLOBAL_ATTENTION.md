# 超边诱导的潜在全局注意力

## 1. 定位

本模块暂称 **Hyperedge-Induced Latent Global Attention（HILGA）**。它用于替换 HDCTI 原有的无 mask 全节点多头自注意力，不是 SP-FBHA 的继续调参，也不恢复 `attention.max.nodes=2000` 造成的数据集依赖混合协议。

当前状态：实现、单元测试和四库 validation-only Pilot 已完成。四库 macro AUPR 下降 `0.000116`，仅 TCMSP 提升，且诊断显示 32 个潜在 token 几乎完全坍缩。因此 HILGA 按预注册规则判定为 **No-Go**，不进入 cold-start、完整五折或论文贡献。

## 2. 动机

原始全节点注意力需要为每个 head 构造节点两两矩阵，复杂度和主要中间张量均为 $O(N^2)$。它在 TCMSP 等数据集上可能提供有用的全局交互，但无法在 ETCM2.0 上稳定运行，也不能在四库使用统一结构。

完全删除全注意力会使模型只剩局部 H-C/P-D 传播、PageRank 标量和候选级 Hctx-P。HILGA 保留“所有节点能够访问全局语义”的功能，但将全局信息压缩为固定数量的药材/疾病超边 token。

诱导点或 latent bottleneck 注意力本身已有先例，不能单独作为创新。当前方法的待验证区别在于：token 由 TCM 关系超边表示生成，而不是与输入无关的自由参数；药材和疾病两侧分别形成全局语义，再返回对应成分和蛋白节点。

## 3. 模型

第 $l$ 层局部超图传播产生节点表示 $X^{(l)}$ 和超边表示 $E^{(l)}$。令 token 数为 $M$，首先学习超边到 token 的分配：

$$
A^{(l)}=\operatorname{softmax}_{edge}
\left(\frac{\operatorname{norm}(E^{(l)})W_A^{(l)}}{\tau}\right)
$$

$$
T^{(l)}=(A^{(l)})^TE^{(l)}
$$

每一列 $A_{:,m}$ 在全部有效超边上归一化，因此一个 token 是多个药材或疾病超边的加权全局摘要。

节点查询这些 token：

$$
Q=\operatorname{norm}(X^{(l)})W_Q,
\quad K=\operatorname{norm}(T^{(l)})W_K,
\quad V=T^{(l)}W_V
$$

$$
G^{(l)}=\operatorname{softmax}
\left(\frac{QK^T}{\sqrt{d_h}\tau}\right)VW_O
$$

最终通过 ReZero 式残差加入局部表示：

$$
\widetilde X^{(l)}=X^{(l)}+\tanh(\gamma_l)G^{(l)}
$$

$\gamma_l$ 初始化为 0，使候选模型的初始前向严格等价于无稠密注意力基线。H-C 和 P-D 使用独立参数，但采用相同公式。

## 4. 复杂度与信息边界

双头 HILGA 的注意力 pair 数近似为：

$$
|E|M+HNM
$$

原全节点注意力为：

$$
HN^2
$$

固定 $M=32$ 时，计算和中间注意力张量随节点与超边数量线性增长。HILGA 只读取当前 fold 固定的 H-C/P-D 侧信息，不读取 H-D，也不使用 validation 或 outer-test C-P 标签。

## 5. 冻结配置

```ini
attention.max.nodes=0
hyperedge.attention=False
global.token.attention=True
global.token.attention.mode=hyperedge_induced
global.token.attention.hc=True
global.token.attention.pd=True
global.token.attention.tokens=32
global.token.attention.heads=2
global.token.attention.temperature=1.0
counterfactual.context=False
```

本轮固定 `tokens=32`、`heads=2` 和 `temperature=1.0`，不同时启用 CHCR。先隔离结构贡献；只有 HILGA 通过四库门槛后，才允许与已冻结 CHCR 组合。

实现会拒绝以下混合配置：

* HILGA 与原全节点注意力同时启用；
* HILGA 与 SP-FBHA 同时启用；
* `num.factors` 不能被 head 数整除。

## 6. 配对基线

不重新训练基线，直接复用统一 `attention.max.nodes=0`、fold 1、validation-only 的已归档结果：

| 数据集 | 冻结 Hctx-P validation AUPR |
|---|---:|
| TCM-Suite | 0.992845 |
| TCMSP | 0.984166 |
| SymMap2.0 | 0.951155 |
| ETCM2.0 mention10 | 0.975974 |
| 四库 macro | 0.976035 |

候选配置除 `model.variant` 和 `global.token.attention.*` 外，必须与对应 Hctx-P 基线完全一致。配置配对由单元测试检查。

## 7. 预注册 Pilot

运行命令：

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_hilga_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_hilga_pilot.conf
./run_hdcti.sh configs/HDCTI_symmap_pair_stratified_hilga_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_pair_stratified_hilga_pilot.conf
```

四库全部满足以下条件才进入 cold-start：

1. 四库 validation AUPR macro 增量不低于 `+0.001`；
2. 至少 3/4 数据集不低于冻结 Hctx-P；
3. 任一数据集下降不超过 `0.003`；
4. 无 OOM、NaN、非法地址或 $N\times N$ 注意力张量。

若普通随机折通过，再构建统一无稠密注意力的四库 compound cold-start 配置。cold-start 进入完整实验的门槛为 macro AUPR 增量不低于 `+0.005`、至少 3/4 不下降、任一数据集下降不超过 `0.005`。

不根据单个数据集结果搜索 token 数、温度、head 数或单侧开关。只有固定双侧版本通过主门槛后，才做 H-C-only/P-D-only 贡献消融。

## 8. Pilot 结果与判定

四库 fold 1 validation-only 配对结果如下：

| 数据集 | 冻结 Hctx-P | HILGA | 增量 | 最佳 epoch | 运行时间 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.992845 | 0.992801 | -0.000044 | 6 | 24.24 s |
| TCMSP | 0.984166 | 0.985749 | +0.001583 | 38 | 56.86 s |
| SymMap2.0 | 0.951155 | 0.949803 | -0.001352 | 22 | 41.81 s |
| ETCM2.0 mention10 | 0.975974 | 0.975322 | -0.000652 | 34 | 166.41 s |
| 四库 macro | 0.976035 | 0.975919 | -0.000116 | - | - |

预注册门槛判定：

| 条件 | 结果 | 判定 |
|---|---:|---|
| macro 增量不低于 `+0.001` | `-0.000116` | 未通过 |
| 至少 3/4 数据集不下降 | 1/4 | 未通过 |
| 任一数据集下降不超过 `0.003` | 最大下降 `0.001352` | 通过 |
| 无 OOM、NaN、非法地址或 $N\times N$ 张量 | 四库均正常完成 | 通过 |

因此普通随机折主门槛未通过。按第 7 节冻结的规则，不运行 HILGA cold-start、完整五折、H-C-only/P-D-only 消融，也不搜索 token 数、head 数或温度。

## 9. 诊断输出与失败机制

每次训练会写出：

```text
saved_model/<time>/global_token_attention.json
```

其中记录：

* 每侧节点、超边和理论 pair 比例；
* 每层学习后的残差系数；
* 超边到 token 分配熵和最大权重；
* 节点到 token 注意力熵和最大权重；
* token 两两余弦相似度。

这些量只用于判断 token 是否完全均匀、坍缩或未激活。结构是否保留仍由预注册 validation AUPR 门槛决定，不能用诊断量替代预测性能。

四库 checkpoint 的诊断结果表明：

* 四层残差尺度均已离开 0，绝对值约为 `0.116-0.426`，因此模块并非未接入或未激活；
* 节点到 token 的归一化熵全部为 `1.0`，平均最大权重全部为 `0.03125=1/32`；
* token 两两平均绝对余弦相似度不低于约 `0.999956`；
* 超边到 token 的分配熵也接近 1，未形成可区分的药材或疾病语义簇。

这说明 32 个 token 基本学习为相同的全局摘要，节点查询退化为均匀读取；HILGA 实际注入的是近似全局均值残差，而不是可选择的全局语义。TCMSP 的单库增益不足以推翻四库预注册判定。

可以设想正交约束、token 多样性损失或非对称初始化等修补，但它们属于新的事后结构搜索。当前实验不据此继续调参；若未来重新研究，应作为新的独立假设重新预注册，而不是将本次 HILGA Pilot 改写为成功结果。

## 10. 后续决策

* **通过普通随机折和 cold-start**：冻结 HILGA，再测试 `HILGA + CHCR`，随后单独审查是否可以删除 C-P PageRank。
* **只在 cold-start 明显有效**：将 HILGA 定位为归纳/冷启动专用模块，不进入随机折主模型。
* **四库门槛未通过**：保留负结果并停止结构搜索，最终模型继续使用 Hctx-P + CHCR，不进行 token 数调参。

本轮对应第三种情况：HILGA 已冻结为可复核的负结果，目标主模型回到 Hctx-P + CHCR。由于既有 CHCR 配置未显式固定 `attention.max.nodes=0`，删除稠密全节点注意力后的最终四库配对结果仍需单独确认，不能直接沿用旧协议结果。

## 11. 方法边界

HILGA 的学术表述不能是“首次提出 latent attention”或“首次将 attention 用于超图”。更稳妥的待验证贡献是：

> 面向中药成分—靶点关系双超图，使用药材和疾病超边诱导固定规模的全局语义 token，在不构造全节点二次注意力矩阵的条件下向成分与蛋白节点注入全局上下文。

由于本轮实验未通过，上述表述仅保留为候选方法记录，不进入论文摘要、方法贡献或最终模型图。
