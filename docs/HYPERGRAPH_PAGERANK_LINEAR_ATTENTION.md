# HPLGA：超图 PageRank 引导的线性全局注意力

## 1. 候选定位

当前统一模型只有两个已验证机制：

```text
Hctx-P：显式建模候选成分的药材上下文与候选靶点的交互
SDIS：在 compound cold-start 中关闭零训练支持的 ID 基础评分
```

原 HDCTI 的无 mask 全节点注意力具有 $O(N^2)$ 时间和显存复杂度，无法在
ETCM2.0 上稳定训练。当前统一协议使用 `attention.max.nodes=0`，因此完全删除了
该全局依赖模块。

第三项候选为 **Hypergraph PageRank-guided Linear Global Attention
(HPLGA)**。目标是在不恢复 $N\times N$ 注意力矩阵的前提下，为 H-C 和 P-D
两侧恢复全节点感受野，并将 H-C/P-D 超图 PageRank 作为注意力信息源的结构先验。
Gate 1 保留 Strict-HDCTI 原有的标量 PageRank，以保证候选与冻结基线只相差
HPLGA 分支；只有候选通过后，才单独比较“保留标量 PageRank”与“由 HPLGA
完全替代标量 PageRank”，避免一次改变两个变量。

该模块与 Hctx-P、SDIS 的职责不同：

| 模块 | 作用层级 | 解决问题 |
|---|---|---|
| HPLGA | 节点编码 | 可扩展的全局结构依赖 |
| Hctx-P | 候选 pair | 药材上下文与靶点的显式交互 |
| SDIS | 最终评分 | 零训练支持时的基础分支失配 |

三者可以在同一个训练和推理配置中联合启用，不按数据库或评估场景切换。

## 2. 选择该方向的原因

### 2.1 原论文证明全局注意力具有功能价值

HDCTI 论文中删除多头注意力后，TCMSP AUC 从 `0.9890` 降至 `0.9788`，
F1 从 `0.9607` 降至 `0.9473`。当前删除全注意力是计算约束，而不是已经证明
全局节点依赖无效。

### 2.2 既有替代候选没有恢复查询相关的全局读取

* SP-FBHA 只重加权局部节点—超边传播；
* HILGA 将全局信息压缩到固定 token，四库审计中 token 基本坍缩；
* Hctx-P 只在候选 pair 层面读取药材上下文；
* SDIS 不改变表示学习。

HPLGA 保留每个节点自己的 query，并从全体节点的结构加权 key/value 统计量中
读取信息，不使用固定潜在 token。

### 2.3 四库输入条件一致

HPLGA 只使用四库都具备的 H-C 和 P-D 关系，不依赖 H-D、SMILES、蛋白序列或
完整 C-P 测试标签。它不会遇到疾病集合规模差异导致的 pair 级集合匹配成本爆炸。

## 3. 方法定义

### 3.1 侧信息超图 PageRank

对 H-C 或 P-D 关联矩阵 $H$，定义归一化超图传播算子：

$$
\Theta=D_v^{-1}HD_e^{-1}H^T.
$$

仅使用侧信息计算节点结构先验：

$$
r^{(t+1)}
=(1-\rho)\frac{\mathbf 1}{N}
+\rho\Theta^Tr^{(t)}.
$$

收敛后的 $r$ 归一化为均值 1。成分侧只读取 H-C，蛋白侧只读取 P-D，因此
PageRank 不依赖当前 fold 的测试 C-P 边，也不会把 compound/protein 的整数 ID
合并为同一节点。

### 3.2 核化线性全局注意力

局部超图传播得到节点表示 $X\in\mathbb R^{N\times d}$。每个 head 计算：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

并使用正值特征映射：

$$
\phi(x)=ELU(x)+1.
$$

将超图 PageRank 作为 key/value 的结构测度：

$$
S=\phi(K)^T\left(r\odot V\right),
$$

$$
z_i=
\frac{
\phi(q_i)S
}{
\phi(q_i)\left(\phi(K)^Tr\right)+\epsilon
}.
$$

最终使用零初始化残差尺度：

$$
X'=X+\eta Z,\qquad \eta_0=0.
$$

零初始化使关闭 HPLGA 时严格退化为当前冻结编码器，并降低新全局分支在训练早期
破坏已有表示的风险。

### 3.3 复杂度

标准全节点注意力需要：

$$
O(N^2d)
$$

时间和 $O(N^2)$ 注意力存储。HPLGA 先计算全局统计量
$\phi(K)^T(r\odot V)$，复杂度为：

$$
O(Nd^2)
$$

时间和 $O(Nd+d^2)$ 额外存储，不创建 `[N,N]` 张量。

## 4. 与近邻工作的边界

HPLGA 不是“首次提出线性注意力”或“首次结合 PageRank 与注意力”。

直接近邻包括：

1. [Personalized PageRank Graph Attention Networks](https://arxiv.org/abs/2205.14259)
   已将 PPR 融入 GAT；
2. [HGraphormer](https://arxiv.org/abs/2312.00336)
   已将超图局部结构与 Transformer 全局信息结合；
3. [ParaFormer](https://arxiv.org/abs/2512.14619)
   已提出 PageRank-enhanced graph Transformer；
4. 核化线性注意力本身属于已有高效 Transformer 技术。

可审查的任务化差异只能表述为：

* 在双侧 TCM 超图中，用 H-C/P-D 超图随机游走先验调制线性全局读取；
* 用该模块替换 HDCTI 不可扩展的稠密全节点注意力；
* 与候选级 Hctx-P 和 cold-start SDIS 形成编码、交互、评分三级统一模型；
* 在四个数据库和 compound cold-start 协议下验证可扩展性与有效性。

只有模型效果、复杂度和消融共同成立后，HPLGA 才能作为第三项模型机制。若只是
把普通线性 attention 接入代码，不能单独声称方法创新。

## 5. 与已失败方向的区别

| 方向 | 核心限制 | HPLGA 的区别 |
|---|---|---|
| 原 full self-attention | 显式生成 $N\times N$ softmax | 核化重排，不生成二次矩阵 |
| SP-FBHA | 只改变局部超边传播权重 | 提供全节点感受野 |
| HILGA | 固定 token 压缩并出现 token 坍缩 | 每个节点保留独立 query，无固定 token |
| Target-conditioned herb attention | 只在一个 pair 内选择药材 | 在节点编码阶段做全局读取 |
| SCHE/支持掩码训练 | 围绕 warm/cold 路由或模拟 | 不使用 pseudo-cold，不建立双专家 |

## 6. 实施顺序

### Gate 0：代码结构与复杂度

必须先满足：

1. `hplga.enabled=False` 时预测与当前 checkpoint 路径一致；
2. 图中不存在形状为 `[compound_count, compound_count]` 或
   `[protein_count, protein_count]` 的张量；
3. H-C/P-D PageRank 只读取侧信息关联矩阵；
4. 空超边、零度节点和极端 ETCM P-D 密度不会产生 NaN；
5. 单元测试覆盖公式、残差关闭、双 head 拼接和 NumPy 参考实现。

### Gate 1：四库 validation-only Pilot

固定：

```text
random.seed=2026
attention.max.nodes=0
Hctx-P=on
SDIS=off
decoder=dot
HPLGA heads=2
HPLGA kernel=elu_plus_one
HPLGA residual_init=0
outer test=off
```

只允许一个预注册配置，不搜索 kernel、head 数、PageRank 阻尼或残差初值。

进入完整实验要求：

```text
四库 validation AUPR macro 增量 > 0
至少 3/4 数据库不下降
任一数据库下降不超过 0.002
ETCM 无 OOM、illegal-address 或 NaN
```

### Gate 2：统一 cold-start 组合（预注册，未执行）

原定在 Gate 1 通过后，使用同一 HPLGA 参数配置运行：

```text
HPLGA + Hctx-P + SDIS
```

相对冻结 `Hctx-P + SDIS`，要求四库 compound cold-start macro AUPR 不下降，
至少 3/4 数据库不下降，任一单库下降不超过 `0.005`。不重新搜索 SDIS gate。
Gate 1 最终未通过，因此本阶段按协议取消。

### Gate 3：论文消融（预注册，未执行）

原定通过后至少报告：

```text
Strict-HDCTI
+ HPLGA
+ Hctx-P
+ HPLGA + Hctx-P
+ HPLGA + Hctx-P + SDIS（compound cold-start）
```

同时报告理论复杂度和 ETCM 峰值张量形状。无需为了 HPLGA 恢复原稠密注意力的
完整 ETCM 训练；原注意力只在可运行的小数据集作为功能参考。
Gate 1 最终未通过，因此 HPLGA 不进入论文主模型消融表。

## 7. Gate 1 结果

四库固定 fold 1、inner-validation AUPR Pilot 已于 2026-07-29 完成。对照为
相同 split、seed、早停、Dot decoder、`attention.max.nodes=0` 和 Hctx-P 的
冻结 Pilot，不重复训练对照。

| 数据集 | Hctx-P | Hctx-P + HPLGA | 增量 | 最佳 epoch | 用时 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.992845 | 0.993293 | +0.000448 | 16 | 22 s |
| TCMSP | 0.984166 | 0.982840 | -0.001326 | 40 | 43 s |
| SymMap2.0 | 0.951155 | 0.946996 | -0.004159 | 16 | 33 s |
| ETCM2.0 mention10 | 0.975974 | 0.975907 | -0.000067 | 16 | 96 s |
| 四库 macro | 0.976035 | 0.974759 | -0.001276 | - | - |

HPLGA 只在 TCM-Suite 提高，未达到“至少 3/4 数据库不下降”；macro 增量为负，
且 SymMap2.0 的 `-0.004159` 超过预注册的最大单库下降 `0.002`。因此 Gate 1
判定为 **No-Go**。

四库两层残差尺度均明显偏离 0，例如 ETCM2.0 为
`HC-L1=0.262229/HC-L2=-0.348564/PD-L1=-0.245606/PD-L2=0.108620`。
这说明分支已参与训练，失败不能归因于零初始化导致模块未激活。

结果来源：

```text
results/batch_runs/hplga_pilot_20260729_195506/results.tsv
results/batch_runs/hplga_pilot_20260729_195506/summary.md
```

按照预注册协议，不运行 HPLGA 完整五折、Gate 2 cold-start 组合，也不搜索
kernel、head 数、PageRank 阻尼、残差初值或数据库特定设置。

## 8. 最终判定

```text
问题必要性：Go
数据适配：Go
工程可行性：Go
跨库有效性：No-Go
论文模型创新：No-Go
当前动作：保留实现与负结果，停止后续训练
```

HPLGA 证明了线性复杂度全局注意力能够稳定运行，但没有形成跨库预测增益，不能
进入最终模型或作为第三项有效创新。
