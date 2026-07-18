# 角色引导稀疏超图注意力与跨数据库泛化计划

## 1. 为什么考虑该方向

当前 Hctx-P 与 CHCR 的主要优势集中在 H-C 药材上下文和 compound cold-start，但它们属于同一机制族。若需要第三项相对独立的模型贡献，新方向必须同时满足：

1. 能在 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0 上使用同一输入；
2. 不依赖四库目前缺失的统一 SMILES、蛋白序列或实体名称映射；
3. 不从完整 C-P 图提取特征，避免把预测标签结构重新输入模型；
4. 面向跨数据库与冷启动，而不是继续在接近饱和的随机边测试上堆模块。

四库都具有 H-C 和 P-D 侧超图，因此可以从与 C-P 监督边独立的侧关系中提取结构角色。

## 2. 近邻工作边界

以下思想已经存在，不能单独作为创新：

* [GraphWave](https://cs.stanford.edu/~marinka/papers/graphwave-kdd18.pdf) 已证明仅依据图扩散模式学习结构角色表示是可行的；
* [GraphSAGE](https://arxiv.org/abs/1706.02216) 已提出面向未见节点和未见图的归纳聚合；
* [MMDG-DTI](https://doi.org/10.1016/j.patcog.2024.110887) 已将域对抗和对比学习用于跨域 DTI；
* [DrugBAN](https://eprints.whiterose.ac.uk/id/eprint/195230/) 已使用条件域适应改善跨域 DTI；
* 近期已有 role-aware DTI 预印本，说明“拓扑角色用于 DTI”本身也不足以形成稳固新颖性。

当前候选的可区分点只能限定为：

> 在缺少跨库实体对齐和生化属性的 TCM CTI 场景中，只从 H-C/P-D 侧超图提取标签独立的结构角色，用它构造可扩展的稀疏注意力邻域，并学习跨数据库可迁移的 pair 表示。

暂称 **Role-Guided Sparse Hypergraph Attention with Domain Generalization（RG-SHADG）**。SHR-DG 作为其中的角色迁移子模块保留。该名称只是工作名，在完成近邻工作全文对照和实验前不写入论文题目或贡献声明。

## 3. 原 HDCTI 注意力不能直接丢弃

原论文 `HDCTI-a` 删除多头注意力。完整 HDCTI 相对该变体的均值增益如下：

| 数据集 | Delta AUC | Delta AUPR | Delta Recall | Delta Precision | Delta F1 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | +0.0016 | +0.0011 | +0.0001 | +0.0024 | +0.0012 |
| TCMSP | +0.0102 | +0.0118 | +0.0145 | +0.0124 | +0.0134 |
| SymMap2.0 | +0.0028 | +0.0023 | +0.0044 | +0.0004 | +0.0023 |

三库 AUC/AUPR/F1 均为正增益，因此不能根据 HILGA、SP-FBHA 或靶点条件化注意力的失败推断“注意力无效”。这些实验只否定对应替代结构。

当前 `attention.max.nodes=0` 的作用是同时跳过 compound/protein 的全节点多头注意力，以统一四库协议并避免 $O(N^2)$ 显存开销。它不会关闭后续的特征维度 softmax 加权，也不等同于删除所有注意力机制。

### 3.1 当前模型中的配对复验

先在 final Hctx-P + CHCR 上只切换原论文全节点注意力，复用相同 Strict fold、seed 和 inner-validation：

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_chcr_full_attention_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_chcr_full_attention_pilot.conf
```

对应无稠密注意力 fold 1 validation AUPR 为：

```text
TCM-Suite: 0.993277
TCMSP:     0.985701
```

本轮只运行 fold 1 且设置 `evaluation.outer.test=False`。它回答“原论文注意力在当前 Strict + Hctx-P + CHCR 中是否仍提供净增益”，不用于最终性能汇报。

判断规则：若 full attention 在两个数据集均下降超过 `0.001`，则只作为 Legacy 基线；否则保留为设计新稀疏注意力的实证依据。结果后不调整 seed、epoch 或早停参数。

### 3.2 配对复验结果（2026-07-18）

| 数据集 | 无稠密注意力 AUPR | 全节点注意力 AUPR | Delta | 全节点注意力最佳 epoch | 全节点注意力运行时间 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.993277 | 0.993390 | +0.000113 | 22 | 89.06 s |
| TCMSP | 0.985701 | 0.985309 | -0.000392 | 48 | 498.48 s |
| Macro | 0.989489 | 0.989350 | -0.000140 | - | - |

该结果通过了预注册的“保留为稀疏注意力设计依据”条件，但不支持恢复全节点注意力作为四库最终默认模块：两个数据集方向不一致，绝对变化均小于 `0.0004`，macro 还略有下降，同时训练时间明显增加。结合原论文三库消融，可得出更窄且可复核的结论：**注意力依赖建模本身仍值得保留，但无 mask 的全节点实现不是当前 Strict + Hctx-P + CHCR 的合适最终形态。**

因此停止 full-attention 五折、seed 和 ETCM2.0/SymMap2.0 扩展；后续仅将其作为 Legacy 机制对照，并转入角色引导稀疏注意力。不能把本轮结果表述为“注意力无效”，也不能把原论文随机五折中的提升直接移植到当前 Strict 协议。

## 4. 阶段 0：冻结角色特征可行性审计

先不修改 HDCTI，不训练 GNN。为每个 compound/target 构造固定维度的侧超图角色特征。

### Compound 侧 H-C 特征

```text
H-C degree
incident herb size: min / mean / std / max
incident herb rarity/IDF: min / mean / max
C-H-C two-hop reachable compound count
two-hop neighbor H-C degree quantiles
```

### Target 侧 P-D 特征

```text
P-D degree
incident disease size: min / mean / std / max
incident disease rarity/IDF: min / mean / max
P-D-P two-hop reachable target count
two-hop neighbor P-D degree quantiles
```

所有特征只从 H-C/P-D 计算，不使用 C-P degree、C-P PageRank、C-P-C/P-C-P 路径或测试标签。

### Pair probe

将 compound/target 角色特征构造成：

```text
[role_c, role_p, role_c * role_p, |role_c - role_p|]
```

使用线性 Logistic Regression 作为容量受限 probe。每轮留出一个数据库作为目标域，在另外三个数据库拟合；目标域只使用固定 H-C/P-D 图做无标签归一化，不使用目标域 C-P 标签调参。

## 5. 角色引导稀疏注意力

若结构角色 probe 通过，不直接使用通用 role embedding 拼接，而将角色用于限制和偏置原 HDCTI 注意力。

每个节点只关注以下邻域并取并集：

```text
Local hypergraph neighbors:
  compound: 共享 herb 的 C-H-C 邻居
  target:   共享 disease 的 P-D-P 邻居

Role-similar global neighbors:
  根据 H-C/P-D 侧超图角色选取 Top-K
```

注意力分数为：

$$
s_{ij}^{(h)}=
\frac{(Q_i^{(h)})^T K_j^{(h)}}{\sqrt{d_h}}
+\beta_h\,sim(r_i,r_j)
+b_{relation(i,j)}
$$

其中 $r_i$ 只由 H-C/P-D 侧关系生成，不能含 C-P degree 或 C-P PageRank。最终仅在固定邻域内 softmax：

$$
\alpha_{ij}^{(h)}=
softmax_{j\in\mathcal N_K(i)}(s_{ij}^{(h)})
$$

复杂度由全节点注意力的 $O(N^2d)$ 降为 $O(NKd)$。该结构保留论文注意力的自适应依赖建模，同时使 SymMap2.0 和 ETCM2.0 可运行。

第一版固定：

```text
local Top-K = 16
role Top-K  = 16
heads       = 2
使用残差连接
不同时启用 HILGA/SP-FBHA
```

K 值只允许在源域 inner-validation 中选择，不根据目标数据库或 outer-test 调整。

## 6. 必须保留的对照

```text
Class-prior baseline
Degree-only probe
Full side-hypergraph role probe
Label-permutation probe
Degree-matched negative evaluation
```

Degree-matched negative 是必要对照：若 full role 只在随机负样本上有效、在度数匹配负样本上退化到随机水平，则结果主要反映数据库观测热度，不能支持结构角色迁移。

## 7. 预注册闸门

RG-SHADG 进入神经模型 Pilot 必须同时满足：

1. 四个目标库至少 95% 的评价 pair 能生成有限角色向量；无侧关系实体保留为带 support 指示位的 `isolated-role`，不得通过删除无支持实体提高覆盖率；
2. full role probe 相对 degree-only AUPR 在至少 3/4 个目标库提高不少于 0.01；
3. degree-matched negative 下至少 3/4 个目标库 AUC 不低于 0.55；
4. 没有目标库出现 full role AUC 低于 0.49；
5. label-permutation 的 AUC/AUPR 回到随机水平；
6. 所有预处理、归一化和超参数均在源域确定，不读取目标域标签。

若未通过，不搜索更复杂 role 指标、域损失或神经架构，停止第三结构模块路线。

## 8. 冻结角色审计结果（2026-07-18）

固定命令：

```bash
python tools/audit_side_hypergraph_role_dg.py
```

每个数据库最多抽取 30,000 个 C-P 正例和等量固定未观测 pair。每轮使用其余三个数据库拟合 source-only 标准化的线性 Logistic probe，目标数据库不参与拟合、调参或阈值选择。

| 目标库 | Degree AUPR | Full-role AUPR | Delta | Full-role AUC | Degree-matched AUC |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.470277 | 0.508050 | +0.037773 | 0.459539 | 0.533216 |
| TCMSP | 0.678814 | 0.624869 | -0.053945 | 0.676442 | 0.497821 |
| SymMap2.0 | 0.519554 | 0.412983 | -0.106571 | 0.394296 | 0.486459 |
| ETCM2.0-mention10 | 0.471495 | 0.466653 | -0.004842 | 0.461393 | 0.477502 |

侧关系支持率也存在明显数据库差异：TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0 正例 pair 的双侧支持率分别为 `58.43%/74.53%/94.78%/100.00%`。所有 pair 均通过 isolated-role 生成有限向量，没有选择性删除低支持样本。

正式判定为 **No-Go**：

* full role 相对 degree-only 只在 1/4 目标库提高至少 `0.01`；
* degree-matched AUC 在 0/4 目标库达到 `0.55`；
* 3/4 目标库的 full-role AUC 低于 `0.49`；
* 单次标签置换在 TCMSP/SymMap2.0 上仍形成较高目标库排序，提示严重跨库协变量偏移，未满足随机化控制。

标签置换在跨域场景中可能因任意随机线性方向与目标域结构相关而偏离 `0.5`，因此不能单独据此声称泄漏；但本轮 No-Go 不依赖该条件，因为 role gain、degree-matched AUC 和最低目标 AUC 三项门槛已经独立失败。按预注册规则，不实现 RG-SHADG 神经模块，不搜索 role 特征、bin 数、分类器容量、Top-K 或域对齐损失。

完整机器可读结果见 `results/side_hypergraph_role_dg/frozen_role_seed2026/`。

### 8.1 失败原因

当前结果至少包含四类可区分因素：

1. **P-D 结构尺度不一致。** 受支持 protein 的 P-D degree 中位数在 TCMSP、TCM-Suite、SymMap2.0、ETCM2.0 中分别为 `1/5/6/4325`。ETCM2.0 的绝对角色值远超三个源库的取值范围，source-only 标准化无法把它映射到同一语义空间。
2. **侧关系覆盖不一致。** 正例 pair 的双侧支持率分别为 `58.43%/74.53%/94.78%/100.00%`。模型可利用的侧信息量随数据库变化，`isolated-role` 的含义也不是跨库恒定的。
3. **负例来源不统一。** TCM-Suite、SymMap2.0 和 ETCM2.0 使用各自已有 ZERO 文件，TCMSP 使用确定性均匀未观测采样。不同数据库的 ZERO 构建规则可能形成额外域标记。
4. **绝对统计角色具有数据库构建依赖。** degree、hyperedge size、rarity 和二跳规模均受数据库收录范围、疾病粒度和证据密度影响。当前 probe 学到的主要是数据库特异尺度，而不是稳定生物学角色。

### 8.2 允许的一次协议校正版

V1 的 No-Go 必须永久保留，不能通过删除困难数据库或只报告 TCM-Suite 来改写。若继续，只允许将下列方案登记为独立的 **RG-SHADG-V2 协议校正审计**：

```text
1. 四库统一从完整未观测 C-P 候选空间按同一 seed 均匀采样 1:1 负例；不读取 ZERO 文件。
2. 每个数据库、每个实体侧仅依据无标签 H-C/P-D 图，将连续角色特征转换为库内 empirical percentile。
3. supported/isolated 指示位保持二值，不删除无侧关系实体。
4. 沿用相同 leave-one-dataset-out、degree-only、degree-matched 和标签置换对照。
5. 沿用 V1 的全部 Go/No-Go 门槛，不搜索 percentile 公式、采样比例、分类器、bin 或阈值。
```

该调整只处理两个已识别的协议混杂：负例来源和绝对尺度。它不能加入域对齐损失或神经注意力来“救”冻结 probe。V2 若仍未通过，则永久关闭侧超图角色迁移路线；若通过，才允许实现一版固定的角色引导稀疏注意力。

### 8.3 V2 协议校正结果（2026-07-18）

固定命令：

```bash
python tools/audit_side_hypergraph_role_dg.py --audit-version v2
```

| 目标库 | Percentile-degree AUPR | Full-role AUPR | Delta | Full-role AUC | Degree-matched AUC |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.666771 | 0.660680 | -0.006091 | 0.651648 | 0.523217 |
| TCMSP | 0.755530 | 0.627745 | -0.127786 | 0.652898 | 0.501473 |
| SymMap2.0 | 0.452714 | 0.595415 | +0.142701 | 0.575885 | 0.451698 |
| ETCM2.0-mention10 | 0.474240 | 0.509148 | +0.034908 | 0.514183 | 0.504195 |

V2 证明统一负例和库内百分位确实消除了部分 V1 混杂：SymMap2.0 与 ETCM2.0 的 full-role 增量转为正值，四库 full-role AUC 也均不低于 `0.49`。但核心门槛仍失败：

* full-role AUPR 只有 2/4 目标库提高至少 `0.01`，未达到 3/4；
* degree-matched AUC 为 `0.523217/0.501473/0.451698/0.504195`，0/4 达到 `0.55`；
* TCMSP full-role 相对 percentile-degree 下降 `0.127786`；
* 标签置换仍未在所有目标库回到随机范围。

因此 V2 正式判定为 **No-Go**，侧超图统计角色跨库迁移路线永久关闭。后续不得搜索 percentile 公式、role 特征、bin 数、Top-K、域损失或神经编码器，也不得只选取 SymMap2.0/ETCM2.0 的阳性结果实现 RG-SHADG。原始稠密注意力的论文贡献与本轮角色迁移失败是两个不同结论：可以继续保留原注意力作为 Legacy 机制证据，但不能再用当前侧图统计角色构造其稀疏替代。

完整 V2 结果见 `results/side_hypergraph_role_dg/frozen_role_percentile_uniform_seed2026/`。

### 8.4 为什么 SymMap2.0 的表面增量较大

SymMap2.0 的 `+0.142701` 不能单独解释为角色迁移成功，原因包括：

1. **基线较低放大了增量。** percentile-degree AUPR 只有 `0.452714`，full-role 虽提高到 `0.595415`，绝对性能仍只是中等；大 Delta 同时来自较弱的 degree-only 基线。
2. **侧关系覆盖较完整。** 正例 pair 的 compound/protein/both support 为 `99.37%/95.39%/94.78%`，明显高于 TCM-Suite 和 TCMSP，因此 hyperedge size、rarity、二跳邻居等完整角色特征能覆盖更多评价样本。
3. **百分位变换适合其相对结构。** SymMap2.0 的受支持 protein degree 中位数为 `6`，不像 ETCM2.0 的 `4325` 那样超出其他源库数量级。V2 把绝对规模转换为相对秩后，源库学习到的部分“高/低连接角色”可以迁移到标准随机未观测对。
4. **增益没有通过困难对照。** full-role 的标准 AUC 为 `0.575885`，但 degree-matched AUC 降到 `0.451698`。一旦正负 pair 的 compound/protein degree 分布被匹配，角色优势完全消失并出现反向排序。
5. **V1/V2 方向反转说明协议敏感。** SymMap2.0 在 V1 的 AUPR Delta 为 `-0.106571`，统一负例和百分位后变成 `+0.142701`。这证明结果高度依赖负例来源和角色尺度，而不是一个对协议稳定的生物学机制。

因此，SymMap2.0 可记录为“标准随机未观测候选上的数据集特异阳性信号”，不能作为继续实现 RG-SHADG、修改 Go/No-Go 门槛或只在 SymMap2.0 上报告模型提升的依据。

## 9. 未启用的条件实现方向

以下方案仅在冻结 probe 通过时成立；V1/V2 均 No-Go 后不再实施，保留在此只用于追溯原预注册设计：

```text
H-C/P-D side-hypergraph role encoder
        ↓
local + role Top-K sparse attention
        ↓
role-conditioned Hctx-P fusion
        ↓
source-domain classification + one fixed domain-alignment loss
```

域对齐损失只选择一种，例如 CORAL 或 MMD；不同时堆叠域对抗、对比学习和多种分布距离。最终必须报告：

```text
四库 leave-one-dataset-out
四库内部 Strict 随机折
compound cold-start
w/o role bias / local-only attention / w/o alignment 消融
degree-matched negative 压力测试
```

该路线的价值不取决于 ETCM 单库提升，而取决于跨库迁移是否在至少三个目标数据库上成立。
