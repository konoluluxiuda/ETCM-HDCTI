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

### 8.1 等价实现效率复核

预排序版本使用完全相同的 ETCM 配置得到：

```text
Best epoch: 14
Validation AUPR: 0.870811
Running time: 166.830888 s
```

相对冻结 Hctx-P，validation AUPR 仍提高 `0.009095`，高于准确性下限；相对优化前 SP-FBHA 只变化 `-0.000785`。但运行时间仍为 Hctx-P 的约 `2.22x`，高于 `150.32 s` 上限，且未较优化前的 `161.77 s` 改善。这说明主要成本来自两百万级 P-D incidence 上每个训练步骤的可微分 segment softmax 与稀疏值梯度，而不是动态 SparseReorder。

最终判定为：

```text
准确性：通过
跨数据集方向：通过（3/4 提高，1/4 轻微下降）
可学习性：通过
可扩展性：未通过
SP-FBHA 效率优先路线：No-Go
```

按照原预注册规则，SP-FBHA 未通过“运行时间不超过基线 2 倍”的效率门槛。该结论保持不变，不能将本次复核描述为原 Gate 已通过，也不能据此声称模型兼具高效性或可扩展性。

### 8.2 事后协议修订：准确率优先分支

2026-07-17 对研究目标作如下显式修订：`2.22x` 是效率代价，不等同于模型无效。ETCM 单折绝对运行时间约为 `166.83 s`，四库均未出现 OOM、非有限 loss 或稀疏算子错误，同时 compound cold-start 的四库 macro validation AUPR 提高 `0.017808`。因此保留 SP-FBHA 作为 **accuracy-oriented** 候选继续验证。

该修订遵守以下边界：

1. 不回溯修改或宣称原效率 Gate 已通过；
2. 冻结 `temperature=1.0`、`prior.scale=0.1`、H-C/P-D 双侧启用和现有实现，不根据已见结果继续调参；
3. 运行时间、峰值显存和相对倍率改为必须报告的代价，不再作为准确率优先分支的硬淘汰条件；
4. OOM、NaN、非有限 loss 和稀疏算子错误仍是硬安全门槛；
5. 先在四库标准 Strict `pair_stratified` fold 1 上进行 validation-only 配对复核，候选与冻结 Hctx-P 复用相同 manifest、seed 和早停，并统一设置 `attention.max.nodes=0`，关闭原 HDCTI 的全节点稠密自注意力；
6. 标准随机折复核仍要求 macro validation AUPR 增量不小于 `+0.002`、至少 3/4 数据集不下降、任一数据集下降不超过 `0.005`；通过后才进入完整五折。

配对配置的运行顺序为每个数据集先基线、后候选，以确保首次生成的 split manifest 被候选复用：

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_herb_only_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_sp_fbha_pilot.conf

./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_herb_only_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_sp_fbha_pilot.conf

./run_hdcti.sh configs/HDCTI_symmap_pair_stratified_herb_only_pilot.conf
./run_hdcti.sh configs/HDCTI_symmap_pair_stratified_sp_fbha_pilot.conf

./run_hdcti.sh configs/HDCTI_etcm_mention10_pair_stratified_herb_only_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_pair_stratified_sp_fbha_pilot.conf
```

这些配置均设置 `evaluation.outer.test=False` 和 `attention.max.nodes=0`。该阶段只用于结构选择，不能把内层验证结果当作最终测试结果。`attention.max.nodes=0` 只关闭原 HDCTI 的成分/蛋白全节点稠密自注意力；候选中的 SP-FBHA 稀疏超边注意力仍然启用。

### 8.3 标准 Strict 随机折复核

2026-07-17 将四库协议进一步统一为 `attention.max.nodes=0`。原因是阈值 `2000` 会令不同数据集、不同实体侧执行不同结构，不适合作为最终四库统一模型。所有新结果均为 fold 1 的 inner-validation AUPR，不包含 outer-test。

阈值 `2000` 下已经完成的 TCM-Suite 配对结果保留为历史探索记录，但不进入新协议的四库 Gate：

| 数据集 | Hctx-P AUPR | SP-FBHA AUPR | 增量 | Hctx-P 时间 | SP-FBHA 时间 | 时间倍率 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite (`max.nodes=2000`) | 0.992808 | 0.993092 | +0.000284 | 23.82 s | 21.20 s | 0.89x | 历史探索，不纳入统一 Gate |

TCM-Suite 基线在 epoch 16 取得最佳 AUPR，SP-FBHA 在 epoch 14 取得最佳 AUPR。候选提升仅为 `0.000284`，属于小幅正向变化，单个数据集不足以证明总体收益；但它满足“不下降”和安全门槛。运行时间下降主要来自候选更早触发早停，不能据此声称单步计算更快。

学习后的参数仍主要集中于 H-C node-to-edge：HC-L1/HC-L2 node mean-abs 分别为 `0.414173/0.589344`；对应 edge 参数为 `0.004070/0.003061`。P-D 两层 node 参数仅为 `0.001428/0.000134`，说明当前增益更可能来自药材—成分侧的成员选择，而不是疾病—蛋白侧。

统一关闭稠密自注意力后的四库复核已完成：

| 数据集 | Hctx-P AUPR | SP-FBHA AUPR | 增量 | Hctx-P 时间 | SP-FBHA 时间 | 时间倍率 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite | 0.992845 | 0.993335 | +0.000490 | 19.46 s | 27.93 s | 1.44x | 不下降 |
| TCMSP | 0.984166 | 0.984216 | +0.000050 | 39.56 s | 57.26 s | 1.45x | 不下降 |
| SymMap2.0 | 0.951155 | 0.949096 | -0.002059 | 28.29 s | 37.76 s | 1.33x | 下降但未越过 -0.005 |
| ETCM2.0 | 0.975974 | 0.975442 | -0.000532 | 139.96 s | 233.63 s | 1.67x | 下降 |
| Macro mean | 0.976035 | 0.975522 | **-0.000513** | - | - | - | 未通过 |

TCM-Suite 无稠密注意力基线在 epoch 16 取得最佳 AUPR，SP-FBHA 在 epoch 20 取得最佳 AUPR。候选提高 `0.000490`，满足不下降条件，但仍属于接近饱和区间的小幅变化。候选运行时间为基线的 `1.44x`，同时其训练持续到 epoch 30，而基线在 epoch 26 停止，因此该倍率不能分解为纯单步开销。

SP-FBHA 参数仍主要由 H-C node-to-edge 学习：HC-L1/HC-L2 node mean-abs 为 `0.438936/0.635311`；H-C edge 参数仅为 `0.000909/0.000028`，P-D node 参数仅为 `0.001026/0.000292`。这与此前 cold-start 和阈值 2000 结果一致，说明主要机制是药材超边内部的成分成员选择。

TCMSP 基线在 epoch 44 取得最佳 AUPR，SP-FBHA 在 epoch 34 取得最佳 AUPR。候选只提高 `0.000050`，应解释为近似持平而不是实质提升。候选运行时间为基线的 `1.45x`；虽然最佳 epoch 更早，但 SP-FBHA 的单步稀疏 segment softmax 与梯度计算仍带来额外开销。

TCMSP 参数同样集中于 H-C node-to-edge：HC-L1/HC-L2 node mean-abs 为 `0.289143/0.429722`，P-D node 参数仅为 `0.000553/0.000022`。前两库平均 AUPR 增量为 `+0.000270`，尚低于最终四库 macro Gate `+0.002`，需继续观察 SymMap2.0 和 ETCM2.0。

SymMap2.0 基线和 SP-FBHA 均在 epoch 22 取得最佳 AUPR，候选下降 `0.002059`。同配置另一次基线为 `0.951087`，与本次相差 `0.000068`；即使按较低基线计算，候选仍下降 `0.001991`，因此下降不是由这一级别的 GPU 数值波动造成。该结果未越过单库最大允许下降 `0.005`，但不满足“不下降”。

SymMap2.0 的参数模式仍以 H-C node-to-edge 为主：HC-L1/HC-L2 node mean-abs 为 `0.430133/0.494534`，P-D node 参数仅为 `0.000878/0.000142`。候选与基线训练到相同停止轮次，`1.33x` 时间倍率更接近 SP-FBHA 的实际单步额外开销。

前三库增量之和为 `-0.001519`，平均为 `-0.000506`。要满足四库 macro 增量不小于 `+0.002`，ETCM2.0 必须至少提高 `+0.009519`；同时 ETCM2.0 必须不下降，才能满足至少 3/4 数据集不下降的条件。

ETCM2.0 基线在 epoch 34 取得最佳 AUPR，SP-FBHA 在 epoch 22 取得最佳 AUPR。候选下降 `0.000532`；虽然幅度较小，但没有提供补偿 SymMap2.0 下降所需的增益。候选运行时间为基线的 `1.67x`，主要成本来自约 `1,824,967` 条 P-D incidence 上的分段 softmax 与梯度计算。

ETCM2.0 的 H-C node-to-edge 参数仍明显学习，HC-L1/HC-L2 mean-abs 为 `0.230090/0.497021`；P-D 第一层 node 参数为 `0.004391`，其余 P-D/edge 参数接近零。由此可见 No-Go 并非模块完全没有梯度，而是学习到的超边成员重加权未改善普通随机划分的总体排序。

### 8.4 标准随机折最终判定

统一无稠密注意力协议的四库结果为：

```text
Macro validation AUPR: 0.976035 -> 0.975522 (-0.000513)
不下降数据集: 2/4
最大单库下降: -0.002059 (SymMap2.0)
安全门槛: 通过（无 OOM、NaN 或稀疏算子错误）
准确率 Gate: 未通过
```

SP-FBHA 未满足 macro 增量不小于 `+0.002` 和至少 3/4 数据集不下降两项条件，因此不进入普通 `pair_stratified` 协议的完整五折，也不作为通用随机划分增强模块。四库 compound cold-start 在阈值 2000 口径下曾得到 macro `+0.017808`，说明它仍可保留为冷启动专用候选；但在将其写为最终冷启动模块前，需要在统一 `attention.max.nodes=0` 的 cold-start 配置下重新确认，不能把两种 attention 口径混合为同一证据。

## 9. 研究边界

SP-FBHA 是当前仓库中的候选实现名称，不据此声称文献首创。进入论文主模型前仍需完成针对 hypergraph attention、incidence attention 和 degree/specificity prior 的近邻工作核验，并明确与已有 HyperGAT/HNHN/AllSet 类方法的结构差异。
