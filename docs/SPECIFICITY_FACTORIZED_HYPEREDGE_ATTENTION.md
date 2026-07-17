# 特异性先验因子化双向超边注意力

## 1. 定位

本模块暂称 **Specificity-Prior Factorized Bidirectional Hyperedge Attention（SP-FBHA）**。它是继 Hctx-P 与 CHCR 后筛选的独立编码器候选，不是已经确认的论文创新，也不与失败的 SACR 组合。

此前固定特异性权重冻结审计在传播后重算上下文，H-C/P-D 平均余弦变化仅为 `0.000890/0.000008`，说明超边表示形成后再调权过晚。SP-FBHA 改在每层传播内部、节点表示被平均前学习权重，因此检验的是不同假设，不能视为对已终止 IDF 路线的参数补搜。

## 2. 模型

令二值 incidence 为 $I_{ev}$，节点表示为 $x_v$。为避免在 ETCM2.0 的约两百万 P-D incidence 上生成 $|I|\times d$ 稠密张量，注意力分解为节点标量和超边标量。

节点到超边：

$$
q_v^{(l)}=(a_N^{(l)})^T x_v^{(l)}
$$

$$
\alpha_{ev}^{(l)}=
\operatorname{softmax}_{v\in e}(q_v^{(l)}/\tau)
$$

$$
m_e^{(l)}=\sum_{v\in e}\alpha_{ev}^{(l)}x_v^{(l)}
$$

超边到节点：

$$
r_e^{(l)}=(a_E^{(l)})^T m_e^{(l)}+\lambda \widetilde{s}_e
$$

其中：

$$
s_e=\log(1+|V|/\deg(e))
$$

$\widetilde{s}_e$ 是仅在当前 H-C 或 P-D 图内标准化并截断到 $[-3,3]$ 的特异性先验。随后：

$$
\beta_{ve}^{(l)}=
\operatorname{softmax}_{e\ni v}(r_e^{(l)}/\tau)
$$

$$
x_v^{(l+1)}=\sum_{e\ni v}\beta_{ve}^{(l)}m_e^{(l)}
$$

当前冻结设置为 $\tau=1$、$\lambda=0.1$。$a_N$ 与 $a_E$ 从零初始化，使可学习部分初始为均匀聚合；固定的小幅先验只在超边到节点阶段起作用。H-C 和 P-D 每层独立学习参数。

## 3. 计算边界

实现只生成：

```text
节点标量: |V|
超边标量: |E|
incidence 权重: |I|
节点/超边表示: (|V| + |E|) x d
```

时间复杂度约为：

$$
O(|I|+(|V|+|E|)d)
$$

不会生成全节点 $N\times N$ 注意力，也不会生成 $|I|\times d$ 的 incidence 特征张量。该约束是 ETCM2.0 和 SymMap2.0 可运行性的必要条件。

## 4. 与现有模块的关系

* 原 HDCTI：H-C/P-D 两个方向均为固定归一化平均；
* Hctx-P：只在候选打分阶段加入药材上下文残差；
* CHCR：只约束事实与反事实药材上下文的相对得分；
* SP-FBHA：在编码器内部改变 H-C/P-D 消息形成方式；
* SACR：已 No-Go，本轮保持 `support.router=False`；
* CHCR：本轮关闭，避免训练正则增益混入结构筛选。

因此本轮比较为：

```text
冻结静态 Hctx-P
vs.
静态 Hctx-P + SP-FBHA
```

## 5. 配置

```text
hyperedge.attention=True
hyperedge.attention.mode=factorized_specificity
hyperedge.attention.hc=True
hyperedge.attention.pd=True
hyperedge.attention.temperature=1.0
hyperedge.attention.prior.scale=0.1
counterfactual.context=False
attention.max.nodes=2000
```

默认 `hyperedge.attention=False`，旧配置和 checkpoint 不受影响。训练后写出 `saved_model/<time>/hyperedge_attention.json`，记录结构规模和每层节点/超边打分向量的平均绝对值。

## 6. 四库 validation-only Pilot

四个配置复用冻结的 compound cold-start fold 1、seed、早停、Dot decoder、Hctx-P 和 `attention.max.nodes=2000`。`evaluation.outer.test=False`，不再使用 outer-test 选择结构。

冻结 Hctx-P validation AUPR：

| 数据集 | Hctx-P validation AUPR |
|---|---:|
| TCM-Suite | 0.604804 |
| TCMSP | 0.948989 |
| SymMap2.0 | 0.806072 |
| ETCM2.0 | 0.861716 |

运行命令：

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_cold_start_sp_fbha_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_cold_start_sp_fbha_pilot.conf
./run_hdcti.sh configs/HDCTI_symmap_cold_start_sp_fbha_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_sp_fbha_pilot.conf
```

## 7. Go/No-Go

进入标准 Strict 随机折审计必须同时满足：

1. 四库 validation AUPR macro 增量不小于 `+0.002`；
2. 至少 3/4 数据集的 validation AUPR 不低于冻结 Hctx-P；
3. 任一数据集下降不得超过 `0.005`；
4. H-C/P-D 至少一侧存在非零学习后的节点和超边打分向量；
5. 四库均无 OOM、非有限 loss 或稀疏算子错误，运行时间不超过对应 Hctx-P 的 2 倍。

未通过时停止该结构，不搜索 temperature、prior scale、单侧开关或更高维 attention MLP。通过后才在四库标准 Strict 随机 fold 1 上验证总体性能和 degree 分层效果；仍通过才考虑完整五折及 H-C/P-D 侧消融。

## 8. 首轮 Pilot 结果

| 数据集 | Hctx-P validation AUPR | SP-FBHA validation AUPR | 增量 | Hctx-P 时间 | SP-FBHA 时间 | 时间倍率 |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 0.604804 | 0.655715 | +0.050911 | 25.72 s | 30.36 s | 1.18x |
| TCMSP | 0.948989 | 0.948049 | -0.000940 | 40.96 s | 68.26 s | 1.67x |
| SymMap2.0 | 0.806072 | 0.817453 | +0.011381 | 27.05 s | 36.24 s | 1.34x |
| ETCM2.0 | 0.861716 | 0.871596 | +0.009880 | 75.16 s | 161.77 s | 2.15x |
| Macro mean | 0.805395 | 0.823203 | **+0.017808** | - | - | - |

准确性条件全部通过：macro 增量超过 `+0.002`，3/4 数据集提高，TCMSP 的最大下降仅为 `0.000940`。四库参数均获得非零梯度；学习主要集中在 H-C node-to-edge 向量，P-D 和 edge-to-node 向量整体较小，后续若进入正式消融需要验证增益是否主要来自 H-C 侧。

首轮 ETCM 运行时间为基线的 `2.15x`，超过预注册 `2x` 上限，因此当前判定为：

```text
准确性 Gate：通过
可学习性 Gate：通过
效率 Gate：暂未通过
总体：等待等价实现效率复核
```

性能分析发现，原实现对每层、每侧、每个 batch 都调用 `tf.sparse_reorder`，ETCM 的约两百万 P-D incidence 会被重复排序。现已在 NumPy 初始化阶段一次性生成 `(edge,node)` 与 `(node,edge)` 两种字典序索引，并删除训练图中的动态 SparseReorder。该修改不改变 attention logits、分段 softmax、稀疏矩阵值或模型参数，只消除重复排序。

效率复核只重跑 ETCM 相同配置：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_sp_fbha_pilot.conf
```

复核时 validation AUPR 仍需不低于 `0.856716`（冻结 Hctx-P `0.861716` 减去最大允许下降 `0.005`），运行时间需不超过 `150.32 s`。若仍超过，SP-FBHA 按预注册效率条件停止；不通过降低 P-D 覆盖、修改 temperature 或关闭单侧来规避门槛。

## 9. 研究边界

SP-FBHA 是当前仓库中的候选实现名称，不据此声称文献首创。进入论文主模型前仍需完成针对 hypergraph attention、incidence attention 和 degree/specificity prior 的近邻工作核验，并明确与已有 HyperGAT/HNHN/AllSet 类方法的结构差异。
