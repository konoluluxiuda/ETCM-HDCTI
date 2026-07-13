# DTI 前沿方法文献核验

本文档核验 15 篇 DTI/CPI 论文的题名、发表信息和主要方法，并修正此前总结中可能引起误解的表述。核验日期为 2026-07-13，优先采用期刊官网、DOI、PubMed 和 PubMed Central 页面。

> “可解释”“跨域”“全局”等词在不同论文中的定义并不一致。本文只记录论文明确实现的技术，不把注意力权重自动视为生物学解释，也不把随机边划分下的性能自动视为冷启动泛化能力。

## 1. 核验结论概览

15 篇论文均能确认存在，但原总结的准确程度不同：

| 序号 | 方法 | 核验结果 | 原总结评价 | 主要修正 |
|---:|---|---|---|---|
| 1 | MMDG-DTI | 已确认 | 基本准确 | 应明确为文本特征、图结构特征与域泛化分类器，而不是泛指任意多模态 |
| 2 | ColdstartCPI | 已确认 | 需修正 | 诱导契合是成对依赖表示的建模思想，并非直接模拟三维构象变化 |
| 3 | SP-DTI | 已确认 | 基本准确 | 亚口袋来自 AlphaFold 结构和 CAVIAR，并结合 ESM-2、ChemBERTa 与 GCN |
| 4 | RSGCL-DTI | 已确认 | 基本准确 | 关系相似图由已知 DTI 派生，交叉验证时必须逐折重建 |
| 5 | DACMF-DTI | 已确认 | 基本准确 | 核心是药物 SMILES 与蛋白序列的双向注意力和跨模态融合 |
| 6 | MML-DTI | 已确认 | 基本准确 | 双曲 GNN 主要用于小分子图，不宜泛化为整个药物—蛋白异构网络 |
| 7 | NASNet-DTI | 已确认 | 明显错误 | `NAS` 指 Node Adaptation and Similarity，不是 Neural Architecture Search |
| 8 | CDI-DTI | 已确认 | 需修正 | 可解释性不能仅由注意力或融合模块推断，摘要重点是跨域多策略融合 |
| 9 | TriCvT-DTI | 已确认 | 需修正 | 三模态主要指药物的图像、序列和图表示，不是药物与蛋白都转为三模态图像 |
| 10 | DHGT-DTI | 已确认 | 准确 | 全局视图是基于元路径的 Graph Transformer，并非无约束全节点注意力 |
| 11 | DTI-MPFM | 已确认 | 过于笼统 | 具体由 BBDKG、RESCAL、CNN 与 Transformer 组成 |
| 12 | HMT-DTI | 已确认 | 需修正 | “层级”主要指多轮高阶元路径知识提取，不等于元路径之间存在形式化包含层级 |
| 13 | MVCL-DTI | 已确认 | 明显错误 | 三个视图是邻居、元路径和扩散视图，不是以随机删边和特征掩码为主 |
| 14 | MVR-DTI | 已确认 | 明显错误 | 分子视觉表示用于药物侧，蛋白仍使用序列特征；并非把两者都转成二维图像 |
| 15 | LDM-DTI | 已确认 | 需修正 | 几何图网络用于药物 2D/3D 结构，不是直接建模药物—蛋白复合物三维结构 |

## 2. 逐篇核验

### 2.1 MMDG-DTI

- 题名：*Drug-target interaction prediction via multimodal feature fusion and domain generalization*
- 期刊：Pattern Recognition, 157, 110887 (2025)
- DOI：[10.1016/j.patcog.2024.110887](https://doi.org/10.1016/j.patcog.2024.110887)
- 核心方法：使用大语言模型相关文本编码器提取文本语义，使用混合 GNN 提取结构特征，再通过域泛化分类器降低训练域过拟合。
- 修正：原总结关于“跨域泛化”的方向正确，但“多模态”应具体化为论文实际使用的文本与图结构信息；不能在没有方法细节支持时笼统写成序列、指纹和三维结构的任意组合。
- 对 HDCTI 的意义：适合支撑“跨数据库泛化”扩展，但需要多个训练域和严格的留域测试，不适合作为第一阶段模型修改。

### 2.2 ColdstartCPI

- 题名：*ColdstartCPI: an induced-fit theory-guided CPI predictive model with improved generalization performance*
- 期刊：Nature Communications, 16, 6436 (2025)
- DOI：[10.1038/s41467-025-61745-7](https://doi.org/10.1038/s41467-025-61745-7)
- 核心方法：使用 Mol2Vec 和 ProtTrans 表示化合物与蛋白，通过解耦 MLP 和 Transformer 建模分子内与分子间作用，使同一实体的表示能够随交互对象变化。
- 修正：论文受诱导契合理论启发，但没有直接执行蛋白构象动力学模拟或显式预测结合前后的三维构象。更准确的说法是“用交互对象依赖的表示近似表达诱导契合思想”。
- 评估：明确覆盖 warm、compound-cold、protein-cold 和双端 blind-start 场景。
- 对 HDCTI 的意义：提示当前纯 ID embedding 无法真正处理新成分或新靶点；若要做冷启动，需要引入可迁移的分子或蛋白属性。

### 2.3 SP-DTI

- 题名：*SP-DTI: subpocket-informed transformer for drug-target interaction prediction*
- 期刊：Bioinformatics, 41(3), btaf011 (2025)
- DOI：[10.1093/bioinformatics/btaf011](https://doi.org/10.1093/bioinformatics/btaf011)
- 核心方法：从 AlphaFold 蛋白结构中使用 CAVIAR 识别亚口袋，结合 ESM-2、ChemBERTa 和 GCN 特征，再用 Transformer 建模药物与蛋白亚口袋之间的细粒度交互。
- 修正：原总结基本准确，但应明确亚口袋不是由普通序列注意力自动得到，而是依赖蛋白结构和口袋识别流程。
- 对 HDCTI 的意义：属于结构多模态路线，需要蛋白结构和化合物结构映射，无法直接应用于只有匿名 ID 的 ETCM2.0 图数据。

### 2.4 RSGCL-DTI

- 题名：*Relational similarity-based graph contrastive learning for drug-target interaction prediction*
- 方法名：RSGCL-DTI
- 期刊：Briefings in Bioinformatics, 26(2), bbaf122 (2025)
- DOI：[10.1093/bib/bbaf122](https://doi.org/10.1093/bib/bbaf122)
- 核心方法：由已知 DTI 构造药物—药物和蛋白—蛋白关系相似网络，通过图对比学习提取关系特征，并融合 D-MPNN 药物结构特征与 CNN 蛋白序列特征。
- 修正：它不是只在原始药物—靶点异构图上做通用对比增强。关系相似网络本身依赖 DTI 标签，因此每个交叉验证 fold 必须只用训练正边构造。
- 对 HDCTI 的意义：与计划中的 `C-P-C`、`P-C-P` 视图高度相关，也是必须建立 Strict-HDCTI 无泄漏流程的直接依据。

### 2.5 DACMF-DTI

- 题名：*DACMF-DTI: Dual attention embedded cross-modality fusion for drug-target interaction prediction*
- 期刊：Knowledge-Based Systems, 326, 114063 (2025)
- DOI：[10.1016/j.knosys.2025.114063](https://doi.org/10.1016/j.knosys.2025.114063)
- 核心方法：以药物 SMILES 和蛋白氨基酸序列为输入，使用一维 CNN 编码，再通过双向注意力显式建模药物子结构与蛋白局部片段之间的作用，并融合交互上下文与独立特征。
- 修正：原总结方向正确，但这里的“多模态”主要是药物序列模态与蛋白序列模态之间的跨模态交互，不应泛化成大量外部组学信息。
- 对 HDCTI 的意义：可借鉴 pair-wise decoder 的双向交互思想；只有单个节点向量时，直接套用 cross-attention 的意义有限。

### 2.6 MML-DTI

- 题名：*MML-DTI: Multimanifold Learning with Hyperbolic Graph Neural Networks for Enhanced Drug-Target Interaction Prediction*
- 期刊：Journal of Chemical Information and Modeling (2026)
- DOI：[10.1021/acs.jcim.5c02826](https://doi.org/10.1021/acs.jcim.5c02826)
- 核心方法：在双曲空间中使用 GNN 编码小分子图，并将双曲结构特征、化学指纹和预训练语言模型语义表示进行多流形融合。
- 修正：双曲 GNN 的明确对象是药物分子图。现有公开摘要不足以支持“药物和靶点的全部层级关系都在统一双曲异构图中学习”这一更强表述。
- 对 HDCTI 的意义：属于高复杂度几何表示路线，当前数据缺少分子结构属性，不是近期最优先方向。

### 2.7 NASNet-DTI

- 题名：*NASNet-DTI: accurate drug-target interaction prediction using heterogeneous graphs and node adaptation*
- 期刊：Briefings in Bioinformatics, 26(4), bbaf342 (2025)
- DOI：[10.1093/bib/bbaf342](https://doi.org/10.1093/bib/bbaf342)
- 核心方法：构建包含药物—药物、靶点—靶点和药物—靶点关系的异构图，用 GCN 提取表示；通过 node-dependent local smoothing 为不同节点动态选择聚合深度，缓解过平滑，最后使用 GBDT 预测。
- 关键修正：`NASNet` 中的 `NAS` 是 **Node Adaptation and Similarity**，不是 Neural Architecture Search。论文没有以自动搜索 GNN 架构作为核心贡献。
- 对 HDCTI 的意义：与修改计划中的“节点自适应传播深度”直接重叠，该模块不能再被当作未经已有工作覆盖的新颖想法。

### 2.8 CDI-DTI

- 题名：*CDI-DTI: A Strong Cross-Domain Interpretable Drug-Target Interaction Prediction Framework Based on Multi-Strategy Fusion*
- 期刊：Journal of Chemical Information and Modeling, 66(5), 2627-2639 (2026)
- DOI：[10.1021/acs.jcim.5c02908](https://doi.org/10.1021/acs.jcim.5c02908)
- 核心方法：融合文本、结构和功能特征，使用多源 cross-attention 做早期对齐，使用双向 cross-attention 建模细粒度药物—靶点交互，并通过 Gram Loss 与深度正交融合减少冗余。
- 修正：论文题名包含“interpretable”，但不能仅凭 attention、Gram Loss 或正交融合就声称已经定位了具有因果意义的关键子结构。具体解释形式需要根据正文实验单独核实。
- 对 HDCTI 的意义：与跨域、多模态和交互解码相关，但实现依赖丰富实体属性，适合后续扩展。

### 2.9 TriCvT-DTI

- 题名：*TriCvT-DTI: Predicting Drug-Target Interactions Using Trimodal Representations and Convolutional Vision Transformers*
- 期刊：IEEE Journal of Biomedical and Health Informatics, 29(6), 4585-4592 (2025)
- DOI：[10.1109/JBHI.2025.3536476](https://doi.org/10.1109/JBHI.2025.3536476)
- 核心方法：构建药物分子图像、化学序列和分子图三种药物表示，以 Convolutional Vision Transformer 提取视觉结构特征，并用双向多头注意力建模药物—靶点交互。
- 修正：三模态主要描述药物侧表示。不能写成“药物和蛋白序列都被转换为 1D、2D 和图三种模态”，也不能在没有正文依据时自行指定距离矩阵等图像生成方式。
- 对 HDCTI 的意义：证明视觉分子表示是一条可行的属性增强路线，但与当前 TCM 关系网络主线距离较远。

### 2.10 DHGT-DTI

- 题名：*DHGT-DTI: advancing drug-target interaction prediction through a dual-view heterogeneous network with GraphSAGE and graph transformer*
- 期刊：Journal of Pharmaceutical Analysis (2025), 101336
- DOI：[10.1016/j.jpha.2025.101336](https://doi.org/10.1016/j.jpha.2025.101336)
- 核心方法：GraphSAGE 编码局部异构邻域，带残差的 Graph Transformer 编码元路径全局/高阶信息，使用注意力融合不同元路径，并通过矩阵分解重构 DTI 与辅助网络。
- 修正：原总结基本准确，但“全局”是基于元路径语义视图的全局信息，不代表对所有节点执行无 mask 的稠密自注意力。
- 对 HDCTI 的意义：这是当前“局部编码器 + 全局语义编码器 + 融合”设想最接近的已有工作，必须作为核心对比文献。仅把 GraphSAGE 换成超图卷积不足以证明强创新性。

### 2.11 DTI-MPFM

- 题名：*DTI-MPFM: a multi-perspective fusion model for predicting potential drug-target interactions*
- 期刊：Expert Systems with Applications, 264, 125740 (2025)
- DOI：[10.1016/j.eswa.2024.125740](https://doi.org/10.1016/j.eswa.2024.125740)
- 核心方法：构建多模态药物—靶点知识图谱 BBDKG，使用 RESCAL 张量分解提取拓扑特征，同时重新编码 SMILES 和蛋白序列，以 CNN 获取局部信息、Transformer 获取全局信息，再进行线性和高阶交互融合。
- 修正：原总结过于宽泛，容易与任何多模态 DTI 模型混淆。论文的可辨识组合是“知识图谱/RESCAL + CNN 局部特征 + Transformer 全局特征 + 多层次融合”。
- 对 HDCTI 的意义：与局部—全局融合概念重叠，但其输入和编码对象不同；比较时应明确 HDCTI 的双超图高阶关系优势。

### 2.12 HMT-DTI

- 题名：*HMT-DTI: hierarchical meta-path learning with transformer for drug-target interaction prediction*
- 期刊：Neural Networks, 194, 108093 (2026)
- DOI：[10.1016/j.neunet.2025.108093](https://doi.org/10.1016/j.neunet.2025.108093)
- 核心方法：采用预计算式层级元路径框架，在预收集阶段使用 Transformer 消息传递评估邻居重要性，多轮扩展高阶语义，并使用局部与全局知识提取器建模多跳邻居和元路径模式。
- 修正：“层级”不宜解释为元路径之间具有明确的包含树结构。更稳妥的表述是分层、逐轮地收集和融合多跳元路径知识。
- 对 HDCTI 的意义：与计划中的元路径 Transformer、局部—全局知识和高阶语义明显重叠，应在设计全局语义模块前阅读全文。

### 2.13 MVCL-DTI

- 题名：*MVCL-DTI: predicting drug-target interactions using a multiview contrastive learning model on a heterogeneous graph*
- 期刊：Journal of Chemical Information and Modeling, 65, 1009-1026 (2025)
- DOI：[10.1021/acs.jcim.4c02073](https://doi.org/10.1021/acs.jcim.4c02073)
- 核心方法：在异构图上构建邻居视图、元路径视图和扩散视图，通过注意力式图对比学习和多视图加权融合学习表示，并额外评估正负比例、困难负样本、已知 DTI 掩蔽和冗余关系等因素。
- 关键修正：论文的主要多视图不是“随机删边 + 特征掩码”生成的两个增强图，而是三个具有不同语义的结构视图。
- 对 HDCTI 的意义：与修改计划中的“局部视图 + 稀疏多跳扩散视图 + 对比学习 + 门控/注意力融合”几乎逐项重叠，是目前最重要的近邻工作之一。

### 2.14 MVR-DTI

- 题名：*MVR-DTI: A Multimodal Molecular Visual Representation Learning for Drug-Target Interaction Prediction*
- 期刊：Journal of Chemical Information and Modeling (2026)
- DOI：[10.1021/acs.jcim.6c01212](https://doi.org/10.1021/acs.jcim.6c01212)
- 核心方法：使用 Vision Transformer 从药物分子视觉表示中提取结构感知特征，并与传统分子描述符、蛋白序列和知识图谱嵌入对齐，结合对比学习和注意力完成跨模态融合。
- 关键修正：分子视觉表示是药物侧模态；公开摘要不支持“将蛋白质序列也转换成二维图像”的说法。模型也不是单纯 CNN 图像分类器。
- 对 HDCTI 的意义：适用于实体属性补全后的多模态扩展，不应替代当前先完成无泄漏结构基线的工作。

### 2.15 LDM-DTI

- 题名：*LDM-DTI: A multimodal framework integrating pretrained language models and geometric graph networks for interpretable drug-target interaction prediction*
- 期刊：Expert Systems with Applications, 313, 131485 (2026)
- DOI：[10.1016/j.eswa.2026.131485](https://doi.org/10.1016/j.eswa.2026.131485)
- 核心方法：使用 ChemBERTa 编码药物 SMILES、ProtBERT 编码蛋白序列，使用三层 GCN 和 EGNN 提取药物二维拓扑与三维几何特征，以动态卷积和多头注意力细化蛋白特征，再通过动态交互注意力融合；其解释实验包括蛋白注意力图可视化。
- 修正：几何图网络明确用于药物分子，不是基于药物—蛋白复合物三维结构进行端到端几何建模。
- 对 HDCTI 的意义：多模态与解释性较完整，但数据要求远高于当前项目；可作为未来 SMILES/序列扩展对照。

## 3. 对原“领域趋势”总结的修正

原趋势判断总体成立，但建议调整为以下六类：

1. **实体属性多模态化**：SMILES、分子图、分子图像、三维几何、蛋白序列、文本与知识图谱联合使用。
2. **交互建模细粒度化**：从独立编码后点积，转向子结构—残基、药物—亚口袋和双向 cross-attention。
3. **结构语义多视图化**：邻居、元路径、扩散、局部与全局视图联合建模。
4. **评估场景严格化**：compound-cold、protein-cold、双端 blind-start、跨域和困难负样本逐渐成为重要评估。
5. **表示空间多样化**：双曲空间和几何等变网络开始进入 DTI，但仍依赖适合的层级或三维数据。
6. **解释目标具体化**：可信解释应落到原子、残基、亚口袋、路径或反事实证据；仅展示 attention 权重不足以证明机制解释。

“神经架构搜索是当前这 15 篇论文体现的主要趋势”不成立，因为 NASNet-DTI 并未使用 Neural Architecture Search。

## 4. 与当前 HDCTI 修改计划的关系

### 4.1 已被近邻工作充分覆盖的组合

当前计划中的以下组合已经存在高度相似的先行工作：

```text
局部图编码
+ 全局/扩散/元路径编码
+ 多视图注意力或门控融合
+ 图对比学习
```

主要近邻论文：

| 当前设想 | 最接近论文 | 重叠内容 |
|---|---|---|
| 局部 GraphSAGE/超图 + 全局 Transformer | DHGT-DTI | 局部 GraphSAGE、元路径 Graph Transformer、注意力融合 |
| 邻居/全局扩散 + 多视图对比 | MVCL-DTI | 邻居、元路径、扩散三视图与对比融合 |
| 层级元路径与局部—全局知识 | HMT-DTI | 多跳元路径、Transformer、局部和全局提取器 |
| C-P-C/P-C-P 关系对比 | RSGCL-DTI | 从训练 DTI 派生关系相似网络并做图对比 |
| 节点自适应传播深度 | NASNet-DTI | node-dependent local smoothing |

因此，“局部—全局 + 门控 + 对比学习 + MLP”本身不足以作为清晰的新颖性主张。模型可以做，但论文贡献必须进一步收窄并形成 DTI 近邻工作没有覆盖的技术差异。

### 4.2 更适合当前项目的差异化方向

建议将研究主线从通用模块堆叠调整为：

```text
面向中药多成分—多靶点机制的双超图建模
+ 严格逐折、无标签泄漏的稀疏全局传播
+ 来源独立的 Herb-Disease 机制路径
+ 面向未观测 C-P 对的困难负样本或 PU 学习
```

其中需要满足：

- `C-P-C`、`P-C-P`、PageRank 和关系相似图只由当前 fold 的训练正边生成。
- H-D 必须先审计来源；若由完整 C-P/P-D 间接推导，则不能作为独立侧信息。
- ETCM2.0 主实验继续优先使用 `mention10`，并以 `cpdeg3` 做剪枝鲁棒性检查。
- 当前数据只有可靠 ID 和关系时，优先完成结构方法；SMILES、蛋白序列和三维模态应在映射覆盖率达到要求后再加入。

### 4.3 下一步文献工作

在开始实现新模型前，建议优先全文精读以下四篇，而不是继续扩展文献数量：

1. **MVCL-DTI**：核对三视图构造、对比目标、数据划分和困难负样本。
2. **DHGT-DTI**：核对局部/全局定义、元路径 Transformer 和辅助网络重构。
3. **HMT-DTI**：核对层级元路径预计算、复杂度和消融。
4. **RSGCL-DTI**：核对关系相似网络是否逐折构造，以及冷启动协议。

建议形成一张统一证据矩阵：

| 论文 | 输入模态 | 图/超图 | 局部视图 | 全局视图 | 是否依赖 DTI 标签构图 | 对比目标 | 划分协议 | 冷启动 | 复杂度 | 代码 |
|---|---|---|---|---|---|---|---|---|---|---|

完成这一步后，再冻结本项目的研究问题、技术差异和必要消融，能显著降低“实现完成后才发现与近期论文高度重叠”的风险。

## 5. 当前结论

1. 这 15 篇论文都是真实可检索的工作，列表本身可信。
2. 原总结在宏观趋势上基本正确，但多处方法细节由题名推断而来，不能直接用于论文相关工作。
3. 最大的事实错误是把 NASNet-DTI 解释为神经架构搜索。
4. 对当前项目最重要的不是立即照搬多模态或 Transformer，而是先处理 MVCL-DTI、DHGT-DTI、HMT-DTI 与拟议主模型之间的新颖性重叠。
5. 当前最合理的研究动作是完成近邻工作证据矩阵，然后重新收敛 `docs/修改计划.md` 中的主贡献表述。
