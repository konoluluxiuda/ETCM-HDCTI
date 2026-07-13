# HDCTI 基线论文笔记

本文档集中记录 HDCTI 基线论文中的研究问题、数据、模型、实验协议、主要结果和消融设计，供后续复现、模型修改和论文写作查阅。

> 信息来源以论文正式版本为准；“复现审查”部分是结合当前仓库代码得到的检查结论，不属于论文作者的原始表述。

## 1. 论文信息

| 字段 | 内容 |
|---|---|
| 英文题目 | Identifying novel therapeutic targets of natural compounds in traditional Chinese medicine herbs with hypergraph representation learning |
| 中文译名 | 基于超图表示学习识别中药天然化合物的新型治疗靶点 |
| 作者 | Yantong Qiao, Lun Hu, Jun Zhang, Pengwei Hu, Xin Luo |
| 期刊 | Briefings in Bioinformatics |
| 年份 | 2025 |
| 卷期 | Volume 26, Issue 4 |
| 文章编号 | bbaf399 |
| DOI | [10.1093/bib/bbaf399](https://doi.org/10.1093/bib/bbaf399) |
| 正式全文 | [Oxford Academic](https://academic.oup.com/bib/article/26/4/bbaf399/8229711) |
| 本地 PDF | [HDCTI.pdf](HDCTI.pdf) |
| 原始代码 | [tong87-bio/HDCTI](https://github.com/tong87-bio/HDCTI) |

## 2. 研究问题与主要贡献

### 2.1 研究任务

论文研究中药天然成分与蛋白靶点之间的关联预测：

$$
(c,t) \rightarrow y_{ct}, \qquad y_{ct}\in\{0,1\}
$$

模型输入包括三类已知关系：

```text
Herb-Compound：药材—成分
Compound-Target：成分—靶点
Target-Disease：靶点—疾病
```

其中 H-C 和 T-D 用于构建两侧超图，C-T 作为监督标签连接两个表示空间。

### 2.2 论文动机

论文认为普通 DTI 方法难以表达中药的多成分、多靶点机制，即 MCMT（multi-component, multi-target）关系。HDCTI 通过超边同时连接多个节点，以建模共享药材的成分高阶关系和共享疾病的靶点高阶关系。

### 2.3 论文强调的贡献

1. 构建药材—成分和靶点—疾病两个独立超图。
2. 使用节点—超边—节点传播学习成分和靶点的高阶结构表示。
3. 使用 self-gating 调节初始嵌入不同维度的信息流。
4. 融合 PageRank，向节点表示注入全局结构重要性。
5. 使用多头注意力建模节点间局部、自适应依赖。
6. 通过点积解码器端到端预测 C-T 关联。

## 3. 论文数据集

### 3.1 表 1：数据规模

| 实体或关系 | TCM-Suite | TCMSP | SymMap2.0 |
|---|---:|---:|---:|
| Herb | 1,009 | 502 | 697 |
| Compound | 1,193 | 13,716 | 27,277 |
| Protein target | 7,258 | 1,749 | 18,192 |
| Disease | 11,071 | 322 | 12,690 |
| Herb-Compound | 6,496 | 33,933 | 85,172 |
| Compound-Protein | 43,669 | 56,169 | 38,043 |
| Protein-Disease | 44,170 | 173,639 | 196,110 |

论文说明三个数据集之间的 herb/disease 直接重叠约为 6%，C-T pair 重复率低于 3%。

### 3.2 正负样本

- 数据库已记录的 C-T 关系作为正样本。
- 从数据库未记录的 C-T pair 中随机抽取负样本。
- 正负样本数量保持 `1:1`。
- 未记录关系实际属于 unlabeled，并不等价于经过实验确认的生物学负关联。

### 3.3 与本地数据统计的关系

论文表 1 是复现目标，但本地文件经过格式转换、去重或版本变化后，部分计数与论文不完全一致。复现实验时应同时记录：

```text
论文声明的实体/关系数
本地原始行数
去重后的唯一边数
实际进入模型的实体/关系数
```

本地详细统计见 [DATASET_STATISTICS.md](DATASET_STATISTICS.md)。

## 4. 模型结构

### 4.1 双超图

药材—成分超图：

```text
节点：Compound
超边：Herb
关联矩阵：A_hc
```

靶点—疾病超图：

```text
节点：Protein target
超边：Disease
关联矩阵：A_td
```

记为：

$$
G_{hc}=(V_c,\epsilon_h,A_{hc}), \qquad
G_{td}=(V_t,\epsilon_d,A_{td})
$$

### 4.2 Self-gating

成分和靶点初始嵌入分别经过门控：

$$
E_c^0=E_c^{ini}\odot\sigma(E_c^{ini}W_{gc}+b_{gc})
$$

$$
E_t^0=E_t^{ini}\odot\sigma(E_t^{ini}W_{gt}+b_{gt})
$$

作用是为不同节点和特征维度自适应控制信息保留比例。

### 4.3 超图传播

以成分侧为例，传播遵循：

```text
Compound node -> Herb hyperedge -> Compound node
```

对应的核心传播算子为：

$$
D_c^{-1}A_{hc}D_h^{-1}A_{hc}^{T}
$$

靶点侧通过 T-D 超图执行同类传播。该操作使同属一个药材的多个成分、同关联一种疾病的多个靶点交换信息。

### 4.4 PageRank 融合

论文使用 PageRank 表示节点在整体结构中的重要性：

$$
PR(v)=\frac{1-\alpha}{N}+\alpha\sum_{u\in M(v)}\frac{PR(u)}{L(u)}
$$

随后对节点嵌入进行标量加权：

$$
E_i \leftarrow E_i\cdot PR(v_i)
$$

论文将 PageRank 描述为全局结构重要性信息，与多头注意力提供的局部、自适应依赖互补。

### 4.5 多头自注意力

每个注意力头计算：

$$
Q=EW_Q,\qquad K=EW_K,\qquad V=EW_V
$$

$$
S^{(h)}=softmax\left(\frac{Q^{(h)}(K^{(h)})^T}{\sqrt{d_k}}\right)V^{(h)}
$$

多个 head 的输出在特征维拼接。论文公式采用无图结构 mask 的全节点注意力，因此时间和空间复杂度包含 $O(N^2)$ 项。

### 4.6 超图卷积与残差

论文最终超图卷积写为：

$$
E_c^l=\sigma\left(
(D_c^{-1}A_{hc}D_h^{-1}A_{hc}^{T}S_c^l)\theta^l
+E_c^{l-1}
\right)
$$

残差连接用于缓解多层传播时的信息稀释和过平滑。

### 4.7 层聚合、解码与损失

论文公式将多层表示写为平均：

$$
E_c=\frac{1}{l_{max}}\sum_{l=1}^{l_{max}}E_c^l,\qquad
E_t=\frac{1}{l_{max}}\sum_{l=1}^{l_{max}}E_t^l
$$

成分—靶点预测采用点积和 Sigmoid：

$$
\hat p_{ij}=\sigma(E_{ci}E_{tj}^{T})
$$

训练目标为交叉熵加 L2 正则：

$$
L=\sum_{i,j}\left[-A_{ct}^{ij}\log\hat p_{ij}
-(1-A_{ct}^{ij})\log(1-\hat p_{ij})\right]
+\lambda\|\Theta\|_2^2
$$

## 5. 论文实验协议

### 5.1 随机五折交叉验证

1. 将全部正样本和等量随机负样本随机划分为五折。
2. 每次使用一折测试，其余四折训练。
3. 论文称五折交叉验证重复五次，再报告平均性能。
4. 随机边划分允许测试集中出现训练阶段没有 C-T 关系的靶点。

这里的“重复五次五折”不等于只运行一次五折并计算 fold 标准差。复现时需要确认论文表中的 `mean +/- SD` 究竟来自重复运行、fold，还是两者的组合。

### 5.2 Disease-aware validation

论文从疾病集合中随机选择 10 个疾病，将这些疾病关联靶点涉及的全部 CTI 作为测试集，其余关系作为训练集。目的是避免与同一疾病关联的靶点同时进入训练和测试。

该协议比随机边划分更接近 target/disease context 冷启动，但仍需要固定所选疾病、随机种子和负样本，才能严格复现。

### 5.3 Disease cluster-based validation

论文还在 TCM-Suite 上根据疾病文本名称进行 K-Means 聚类，将疾病划分为五个互不重叠的组，再进行五折验证。作者强调使用疾病名称而不是 CTI profile 聚类，以降低标签泄漏风险。

### 5.4 Cross-dataset validation

在一个数据集训练，在另外两个数据集测试，共形成六种有向组合，用于评估跨数据库分布泛化。

### 5.5 指标

```text
AUC
AUPR
Recall
Precision
F1-score
```

论文给出了 Recall、Precision 和 F1 的定义，但正文没有明确说明分类阈值选择方法。当前本地代码采用固定阈值 `0.5`，该设置应标为本地实现约定。

### 5.6 对比方法

论文使用以下基线：

```text
HyperAttentionDTI
CoaDTI
HGNNLDA
DrugBAN
MCL-DTI
PerceiverCPI
BINDTI
HGHDA
```

其中 HGHDA 是作者此前面向 herb-disease 超边关联预测的模型；HDCTI 则直接优化 compound 和 target 节点表示以预测 CTI。

## 6. 主要结果

### 6.1 表 2：随机划分下的 HDCTI 结果

| 数据集 | AUC | AUPR | Recall | Precision | F1-score |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.9917 (+/-0.0005) | 0.9934 (+/-0.0002) | 0.9475 (+/-0.0014) | 0.9887 (+/-0.0014) | 0.9677 (+/-0.0008) |
| TCMSP | 0.9890 (+/-0.0005) | 0.9867 (+/-0.0007) | 0.9781 (+/-0.0011) | 0.9439 (+/-0.0016) | 0.9607 (+/-0.0011) |
| SymMap2.0 | 0.9632 (+/-0.0010) | 0.9610 (+/-0.0015) | 0.9180 (+/-0.0019) | 0.8979 (+/-0.0028) | 0.9078 (+/-0.0017) |

### 6.2 表 3：Disease-aware 结果

| 数据集 | AUC | AUPR | Recall | Precision | F1-score |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.9734 (+/-0.0023) | 0.9784 (+/-0.0019) | 0.9045 (+/-0.0077) | 0.9776 (+/-0.0035) | 0.9396 (+/-0.0029) |
| TCMSP | 0.9670 (+/-0.0001) | 0.9649 (+/-0.0004) | 0.9416 (+/-0.0011) | 0.9113 (+/-0.0030) | 0.9262 (+/-0.0010) |
| SymMap2.0 | 0.9048 (+/-0.0020) | 0.9161 (+/-0.0014) | 0.8558 (+/-0.0011) | 0.8077 (+/-0.0078) | 0.8310 (+/-0.0040) |

与随机边划分相比，三个数据集性能均下降，SymMap2.0 的下降最明显。这说明随机边五折可能高估模型在新靶点或新疾病上下文上的泛化能力。

### 6.3 表 6：跨数据集 AUC

| 训练集 | 测试集 | HDCTI AUC |
|---|---|---:|
| TCM-Suite | TCMSP | 0.9239 |
| TCM-Suite | SymMap2.0 | 0.9147 |
| TCMSP | TCM-Suite | 0.9305 |
| TCMSP | SymMap2.0 | 0.9382 |
| SymMap2.0 | TCM-Suite | 0.9247 |
| SymMap2.0 | TCMSP | 0.9195 |

论文报告 HDCTI 在六种跨数据集组合中均优于对比模型和自身消融变体。

## 7. 消融实验

### 7.1 论文定义的变体

| 变体 | 改动 | 检验内容 |
|---|---|---|
| `HDCTI-n` | 将原层表示聚合替换为平均池化 | 求和层聚合是否有效 |
| `HDCTI-x` | 将原层表示聚合替换为最大池化 | 求和是否优于 max |
| `HDCTI-a` | 删除多头注意力 | 多头注意力贡献 |
| `HDCTI-g` | 删除 self-gating | 门控贡献 |
| `HDCTI-p` | 删除 PageRank | 全局重要性加权贡献 |
| `HDCTI` | 完整模型 | 对照组 |

### 7.2 表 5：完整消融结果

| 数据集 | 变体 | AUC | AUPR | Recall | Precision | F1-score |
|---|---|---:|---:|---:|---:|---:|
| TCM-Suite | HDCTI-n | 0.9884 | 0.9907 | 0.9568 | 0.9582 | 0.9575 |
| TCM-Suite | HDCTI-x | 0.9765 | 0.9843 | 0.9558 | 0.9286 | 0.9420 |
| TCM-Suite | HDCTI-a | 0.9901 | 0.9923 | 0.9474 | 0.9863 | 0.9665 |
| TCM-Suite | HDCTI-g | 0.9862 | 0.9894 | 0.9465 | 0.9437 | 0.9451 |
| TCM-Suite | HDCTI-p | 0.9914 | 0.9929 | 0.9505 | 0.9826 | 0.9663 |
| TCM-Suite | HDCTI | 0.9917 | 0.9934 | 0.9475 | 0.9887 | 0.9677 |
| TCMSP | HDCTI-n | 0.9874 | 0.9843 | 0.9799 | 0.9382 | 0.9586 |
| TCMSP | HDCTI-x | 0.9801 | 0.9787 | 0.9791 | 0.8848 | 0.9296 |
| TCMSP | HDCTI-a | 0.9788 | 0.9749 | 0.9636 | 0.9315 | 0.9473 |
| TCMSP | HDCTI-g | 0.9886 | 0.9860 | 0.9793 | 0.9406 | 0.9596 |
| TCMSP | HDCTI-p | 0.9885 | 0.9862 | 0.9758 | 0.9435 | 0.9594 |
| TCMSP | HDCTI | 0.9890 | 0.9867 | 0.9781 | 0.9439 | 0.9607 |
| SymMap2.0 | HDCTI-n | 0.9613 | 0.9602 | 0.9192 | 0.8929 | 0.9059 |
| SymMap2.0 | HDCTI-x | 0.9519 | 0.9512 | 0.9578 | 0.8118 | 0.8788 |
| SymMap2.0 | HDCTI-a | 0.9604 | 0.9587 | 0.9136 | 0.8975 | 0.9055 |
| SymMap2.0 | HDCTI-g | 0.9624 | 0.9610 | 0.9184 | 0.8969 | 0.9075 |
| SymMap2.0 | HDCTI-p | 0.9611 | 0.9583 | 0.9041 | 0.9000 | 0.9020 |
| SymMap2.0 | HDCTI | 0.9632 | 0.9610 | 0.9180 | 0.8979 | 0.9078 |

为便于横向比较，表中省略了各指标的标准差；完整均值和标准差见论文表 5。

### 7.3 对消融结果的直接观察

- `HDCTI-x` 在三个数据集上均明显降低 F1，最大池化不适合作为层聚合方式。
- 删除多头注意力在 TCMSP 上影响最大：AUC 从 `0.9890` 降至 `0.9788`，F1 从 `0.9607` 降至 `0.9473`。
- self-gating 在 TCM-Suite 上贡献明显，但在 SymMap2.0 上增益很小，影响具有数据集依赖性。
- PageRank 在随机五折中的提升整体较小，但论文跨数据集实验中删除 PageRank 后下降更明显。
- 论文没有直接消融双超图、单侧超图、超图卷积、残差连接或点积解码器，这些是本地研究可以补充的基线分析。

## 8. 网络层数实验

论文报告：

- 层数不超过 3 时，整体性能相对稳定。
- TCM-Suite 综合表现最好时使用 2 层。
- TCMSP 和 SymMap2.0 在 3 层时达到最好表现。
- 超过 3 层后，三个数据集性能均下降，作者将其归因于过平滑。

这意味着固定使用 2 层并不一定是全部数据集的论文最优配置。对 ETCM2.0 应独立测试 `1/2/3` 层，而不能直接根据论文基准数据选择。

## 9. 案例研究

论文选择 Coumarin 和 Progesterone：

1. 从训练数据中排除涉及该成分的已知 CTI。
2. 训练 HDCTI 后对候选靶点排序。
3. 检查 Top-10 结果的外部数据库或文献证据。

| 成分 | Top-10 中有证据的靶点数 |
|---|---:|
| Coumarin | 7/10 |
| Progesterone | 8/10 |

该案例属于 compound hold-out 场景，比随机 C-T 边划分更接近新成分靶点发现。复现时还应明确候选靶点全集、已知边过滤规则、外部证据检索日期和证据等级。

## 10. 公开代码中的基础配置

论文正文没有完整列出训练超参数。当前仓库保留的原始配置和代码主要使用：

| 参数 | 值或实现 |
|---|---|
| 嵌入维度 | 64 |
| 最大 epoch | 50 |
| batch size | 2,000 |
| 初始学习率 | 0.005 |
| 优化器 | Adam |
| 超图层数 | 2，代码内硬编码 |
| 注意力头数 | 2，代码内硬编码 |
| 正则参数 | `u=0.001, i=0.001, b=0.2, s=0.2` |
| 主评估 | 随机五折交叉验证 |

这些数值应标为“公开代码配置”，不能自动视为论文正文明确声明的超参数。

## 11. 复现审查与待确认问题

### 11.1 论文内部不一致

论文公式 (12)-(13) 写的是多层表示平均，即乘以 `1/l_max`；但消融部分称完整 HDCTI 使用求和，`HDCTI-n` 才改为平均池化。当前公开代码采用求和。后续应同时保留：

```text
paper-formula：按公式平均
paper-ablation/code：按消融说明和代码求和
```

### 11.2 PageRank 图来源需要核对

论文方法部分将 compound PageRank 描述为基于 H-C 超图；当前代码却使用 C-P 矩阵计算 compound PageRank，而 protein PageRank 使用 P-D 矩阵。这会造成两类问题：

1. 实现与论文描述不一致。
2. 如果 C-P PageRank 使用完整关系文件，测试正边会间接进入训练表示。

因此必须区分：

```text
Legacy PageRank：复现公开代码行为
Paper-faithful PageRank：按论文描述基于 H-C/P-D 侧信息
Strict C-P PageRank：若保留 C-P PageRank，每折仅使用训练正边
```

### 11.3 二部图节点 ID 冲突

当前 NetworkX PageRank 构图直接使用矩阵行号和列号。左侧实体 ID `0` 与右侧实体 ID `0` 会被视为同一个图节点。修复时应使用 `(type, id)` 节点键或给右侧实体增加偏移。

### 11.4 注意力的规模问题

论文多头注意力是无 mask 的全节点注意力。对于 ETCM2.0 的大量 compound，注意力矩阵会达到 $N\times N$，显存需求与磁盘数据集大小无直接关系。

本地新增的 `attention.max.nodes` 不是论文参数：

- 未设置时执行 full self-attention。
- 设置阈值后可能只关闭节点较多的一侧，形成部分注意力模型。
- 这种状态不能直接等同于论文的 `HDCTI-a`，因为 `HDCTI-a` 应同时删除成分侧和靶点侧多头注意力。

### 11.5 当前代码存在额外计算

当前 `HDCTI.py` 在多头注意力之后还有一层特征维度 softmax 加权，论文公式没有清楚描述该模块。进行论文忠实复现前，应确认它来自原始作者代码还是后续本地修改，并将其设置为独立开关。

### 11.6 随机性与统计量

论文称五折交叉验证重复五次，当前本地结果多数是一次五折的 `fold mean +/- fold SD`。二者不能直接视为相同统计协议。严格复现至少要固定并记录：

```text
负样本 seed
fold 划分 seed
模型初始化 seed
batch shuffle seed
每个 fold 的原始指标
每次重复运行的均值
跨 seed 的标准差
```

## 12. 推荐的基线复现实验矩阵

### 12.1 协议审计，不属于模型创新消融

| 实验 | 目的 |
|---|---|
| Repo-Legacy | 保存当前公开代码式结果 |
| Deterministic | 固定负样本、fold 和随机种子 |
| No-Leak | 每折仅使用训练标签构造 C-P 派生统计 |
| PageRank-Fix | 修复图来源和二部图 ID 冲突 |
| Regularization-Fix | 修复正则项重复累加等实现问题 |

这些实验回答“结果是否可信、为何变化”，不回答“模型创新点是否有效”。

### 12.2 论文模块消融

| 实验 | 模块变化 |
|---|---|
| Strict-HDCTI | 固定协议下的完整基线 |
| Strict-HDCTI-avg | 平均层聚合 |
| Strict-HDCTI-max | 最大层聚合 |
| Strict-HDCTI-no-attention | 两侧都删除多头注意力 |
| Strict-HDCTI-no-gating | 删除 self-gating |
| Strict-HDCTI-no-pagerank | 删除 PageRank |
| Strict-HDCTI-no-HC | 删除 H-C 超图传播 |
| Strict-HDCTI-no-PD | 删除 P-D 超图传播 |
| Strict-HDCTI-no-hypergraph | 两侧均不执行超图传播 |
| Strict-HDCTI-no-residual | 删除残差连接 |

前五个变体对应论文表 5；后四个变体用于补充论文未充分验证的结构贡献。所有变体必须共用相同的数据、负样本、fold、seed、训练设置和指标实现。

## 13. 与项目其他文档的分工

| 文档 | 用途 |
|---|---|
| 本文档 | 查询基线论文事实、公式、协议和消融结果 |
| [PAPER_BASELINES.md](PAPER_BASELINES.md) | 保存本地复现实验结果及其与论文结果的差值 |
| [DATASET_STATISTICS.md](DATASET_STATISTICS.md) | 保存本地数据文件的真实统计 |
| [ETCM2_CORE_NOTES.md](ETCM2_CORE_NOTES.md) | 保存 ETCM2.0 构建、剪枝和实验记录 |
| [修改计划.md](%E4%BF%AE%E6%94%B9%E8%AE%A1%E5%88%92.md) | 保存后续模型改进和实验路线 |

## 14. 查阅时的简短结论

1. HDCTI 的核心不是普通二部图预测，而是 H-C/P-D 双超图表示学习。
2. 论文明确消融了层聚合、多头注意力、自门控和 PageRank。
3. 论文没有充分消融双超图、超图卷积、残差和解码器，应在本地补充。
4. 随机五折结果明显高于 disease-aware 结果，不能只报告随机边划分。
5. 当前公开代码行为不完全等于论文文字描述，必须区分 Legacy、paper-faithful 和 Strict 三类实现。
6. ETCM2.0 的 full self-attention 主要受 $O(N^2)$ 节点规模限制，而不是数据目录文件大小限制。

