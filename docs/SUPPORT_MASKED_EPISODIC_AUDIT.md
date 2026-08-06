# 支持掩码 Episodic 训练近邻工作审计

> **历史状态说明**：本审计关于支持掩码候选的 No-Go 仍有效；文中“最终方法
> 暂时保留 Hctx-P + SDIS”是 2026-07-29 的阶段性记录。SCHPT 后的最终模型为
> `Hctx-P + SDIS + SCHPT`，见
> [最终方法冻结规格](FINAL_METHOD_SPECIFICATION.md)。

## 1. 审计结论

截至 2026-07-29，原计划中的“共享 Hctx-P head + 支持掩码 episodic
training”不能作为第三个核心模型创新直接实施和命名。

原候选的关键操作为：

1. 从训练折确定性选择一部分已有 C-P 支持的 compound；
2. 从训练 C-P 图中移除这些 compound 的全部正边，制造 pseudo-cold 实体；
3. 保留这些 pair 的监督标签；
4. 关闭 compound-ID 基础评分，仅用 H-C 侧信息和共享 Hctx-P 评分；
5. 用同一规则处理真实 compound cold-start。

该思想与冷启动推荐中的训练期协同信息 dropout、模拟 cold-start episode 和
masked interaction reconstruction 高度重合。即使把用户/物品替换为
compound/protein、把内容特征替换为 H-C 超图上下文，也主要属于领域适配，
不足以单独承担期刊论文的第三项方法创新。

因此本候选判定为：

```text
创新性门槛：No-Go
工程可用性：可作为后续训练技巧或消融项
当前动作：不实现，不进入单折 Pilot，不写入 Ours-full
```

## 2. 检索范围

### 2.1 研究问题

> 是否已有方法通过在训练期人为删除 warm 实体的交互信息，利用侧信息重建其
> 表示或预测关系，以模拟零交互 cold-start？

### 2.2 检索主题

检索覆盖三个相邻领域：

* drug-target / compound-protein cold-start；
* user/item cold-start recommendation；
* inductive graph representation 和 masked link reconstruction。

关键词组合包括：

```text
drug target interaction cold-start meta-learning
cold-start recommendation interaction dropout side information
simulate cold-start episodic training graph
masked interaction reconstruction cold item
inductive link prediction node masking
```

本次是创新边界审计，不是 PRISMA 系统综述。优先核验正式出版社、会议论文页、
DOI 和作者机构页面；预印本只用于补充方法细节。

## 3. 关键近邻工作证据矩阵

| 工作 | 核心做法 | 与原候选的重合 | 差异 | 判断 |
|---|---|---|---|---|
| DropoutNet, NeurIPS 2017 | 训练期丢弃用户或物品的协同表示，使单一网络显式学习 warm、user-cold 和 item-cold | “删除可迁移性差的 ID/协同输入，再依赖内容侧信息”与 SDIS + pseudo-cold 训练同构 | 不使用中医药超图，也不按 C-P 支持度定义门控 | 高度近邻 |
| PT-GNN, WSDM 2021 | 从交互丰富实体构造模拟 cold-start episode，只用采样邻居重建 warm 实体的 ground-truth embedding | 已明确提出“用 warm 实体模拟 cold entity 并 episodic 训练” | 目标是 embedding reconstruction，并带 meta aggregator 和 sampler | 高度近邻 |
| CLCRec, ACM MM 2021 | 对齐内容特征与协同表示，使用对比目标提高完全 cold item 表示 | 覆盖“侧信息—协同信息一致性训练” | 使用信息论对比目标，不是 H-C/P-D 超图 | 阻断简单对比式重命名 |
| ALDI, SIGIR 2023 | warm item 作为 teacher，通过 rating、ranking 和 identification alignment 将协同知识传给 cold item | 覆盖 warm-to-cold 蒸馏与共享预测行为对齐 | 多种教师资格权重和蒸馏目标 | 阻断简单蒸馏式重命名 |
| CGRC, SIGIR 2024 | 随机选择物品并掩蔽其全部 user-item 边，使用多模态内容重建交互图，直接模拟新物品 | 与“移除 pseudo-cold compound 的全部 C-P 图边但保留重建监督”最接近 | 使用 masked graph autoencoder 和多模态内容 | 直接重合 |
| GNP, ADC 2024 | warm GNN 加 patching network，模拟并修补 cold-start 推荐，同时保护 warm 性能 | 与已失败的 SCHE warm/cold 专家动机接近 | 采用专门 patching 网络 | 支持停止双专家路线 |
| KGE_NFM, Nature Communications 2021 | 在支持知识图中保留 drug/protein 侧信息，但从训练 DTI 中完全移除 cold 实体关系 | 与本项目 side-information-assisted cold-start 定义一致 | 主要是评估和 KG+NFM 架构，不做 pseudo-cold episode | 证明问题设置不是新概念 |
| C2P2, Briefings in Bioinformatics 2022 | 从 CCI/PPI 辅助任务迁移交互知识，提高 cold-drug/cold-target 泛化 | 同样利用独立侧关系补足缺失 DTI | 依赖分子/蛋白外部任务和预训练 | 说明“侧关系辅助 cold-start”已有成熟路线 |
| ColdstartCPI, Nature Communications 2025 | 预训练分子/蛋白特征加 pair-conditioned Transformer，在 warm/cold 设置使用同一框架 | 强调唯一模型同时覆盖 warm 和 cold | 依赖分子子结构与蛋白序列，不使用 TCM 超图 | 当前统一性的重要强基线 |
| CrossLinker, JCIM 2026 | 链路级序列—关系对比与 cross-attention，面向 cold-start 和 few-shot | 覆盖 link-conditioned relation/side-feature alignment | 依赖序列模态和关系模态 | 阻断泛化的 link-level contrastive 主张 |

## 4. 已核验来源

1. Volkovs M, Yu G, Poutanen T. DropoutNet: Addressing Cold Start in
   Recommender Systems. NeurIPS 2017.
   [NeurIPS 正式论文页](https://papers.neurips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)
2. Hao B, Zhang J, Yin H, Li C, Chen H. Pre-Training Graph Neural Networks
   for Cold-Start Users and Items Representation. WSDM 2021.
   [DOI: 10.1145/3437963.3441738](https://doi.org/10.1145/3437963.3441738)
3. Wei Y, Wang X, Li Q, et al. Contrastive Learning for Cold-Start
   Recommendation. ACM Multimedia 2021.
   [DOI: 10.1145/3474085.3475665](https://doi.org/10.1145/3474085.3475665)
4. Huang F, Wang Z, Huang X, et al. Aligning Distillation For Cold-start
   Item Recommendation. SIGIR 2023.
   [DOI: 10.1145/3539618.3591732](https://doi.org/10.1145/3539618.3591732)
5. Kim J, Kim E, Yeo K, et al. Content-based Graph Reconstruction for
   Cold-start Item Recommendation. SIGIR 2024.
   [DOI: 10.1145/3626772.3657801](https://doi.org/10.1145/3626772.3657801)
6. Chen H, Yang Y, Bei Y, et al. Graph Neural Patching for Cold-Start
   Recommendations. ADC 2024.
   [DOI: 10.1007/978-981-96-1242-0_25](https://doi.org/10.1007/978-981-96-1242-0_25)
7. Ye Q, Hsieh C-Y, Yang Z, et al. A unified drug-target interaction
   prediction framework based on knowledge graph and recommendation system.
   Nature Communications 2021.
   [DOI: 10.1038/s41467-021-27137-3](https://doi.org/10.1038/s41467-021-27137-3)
8. Nguyen TM, Nguyen T, Tran T. Mitigating cold-start problems in
   drug-target affinity prediction with interaction knowledge transferring.
   Briefings in Bioinformatics 2022.
   [DOI: 10.1093/bib/bbac269](https://doi.org/10.1093/bib/bbac269)
9. Zhao Q, Zhao H, Guo L, et al. ColdstartCPI: Induced-fit theory-guided
   DTI predictive model with improved generalization performance. Nature
   Communications 2025.
   [DOI: 10.1038/s41467-025-61745-7](https://doi.org/10.1038/s41467-025-61745-7)
10. Xu Z, Que J, Hong Y, et al. CrossLinker: Aligning Relational and
    Sequential Contexts for Drug-Target Interaction Prediction in Cold-Start
    and Few-Shot Scenarios. Journal of Chemical Information and Modeling 2026.
    [DOI: 10.1021/acs.jcim.5c03216](https://doi.org/10.1021/acs.jcim.5c03216)

## 5. 与当前项目的可保留差异

当前项目仍有两个真实区别：

1. cold compound 不是完全无信息实体，而是无训练 C-P 支持、仍保留 H-C
   中医药侧信息的实体；
2. SDIS 显式识别 ID 基础评分与归纳上下文评分的支持失配，并以训练折支持状态
   确定性关闭不可靠分支。

这些差异能够支撑 SDIS 的任务化定义和实验价值，但不足以让“再做一次
pseudo-cold masking”自动变成独立创新。若使用该训练技巧，论文必须引用
DropoutNet、PT-GNN 和 CGRC，并表述为已有 cold-start simulation 思想在当前
任务中的实现。

## 6. Devil's Advocate 审查

### 最强质疑

> 作者只是把 item cold-start 的 interaction dropout / masked reconstruction
> 换成 compound–protein 边掩码，再把 item content 换成 H-C 上下文。

该质疑目前成立。仅强调“首次用于中草药 CTI”不能消除方法同构性。

### 判定

```text
Critical：将支持掩码 episodic training 单列为第三个核心创新
Major：不引用冷启动推荐中的直接近邻工作
Minor：将固定 10% pseudo-cold 称为 meta-learning，但没有 task-level
       adaptation 或 bi-level optimization
```

## 7. 后续决策

1. 不实现原定的简单支持掩码 episodic candidate；
2. 不继续搜索 ratio、mask seed、蒸馏权重或对比损失；
3. 不把已有 SCHE pseudo-cold 代码改名后写入论文；
4. 最终方法暂时保留 `Hctx-P + SDIS`；
5. 第三项贡献优先从“问题/数据/证据闭环”而不是同族训练技巧中寻找：
   * Strict 无泄漏四库 cold-start benchmark；
   * ETCM2.0 mention10 数据构建、实体映射与外部证据；
   * Top-K 冻结候选的正向与冲突案例。

若仍坚持增加第三个模型机制，必须先提出一个不等价于以下类别的明确公式：

```text
interaction dropout / pseudo-cold masking
warm-to-cold distillation
content-collaborative contrastive alignment
warm/cold dual experts or support gate
masked graph reconstruction
```

该条件随后由 HPLGA 候选满足到“允许审计”的最低程度：它研究超图 PageRank
调制的线性全局节点读取，不再模拟 cold entity，也不使用蒸馏、对比或双专家。
这不代表 HPLGA 已成为创新；它只获准进入预注册 Gate 0，详见
[HYPERGRAPH_PAGERANK_LINEAR_ATTENTION.md](HYPERGRAPH_PAGERANK_LINEAR_ATTENTION.md)。
