# HDCTI 近邻工作证据矩阵

本文档比较 HDCTI 与四篇最接近当前修改计划的 DTI 方法，目的是确定已有工作覆盖范围、当前方案的创新边界和实现前必须解决的评估问题。核验日期：2026-07-13。

## 1. 证据等级

| 等级 | 含义 |
|---|---|
| A | 已核对正式全文或开放全文，并有方法/实验段落支撑 |
| B | 已核对期刊正式摘要、引言和可访问的补充材料说明，但部分实现细节不可见 |
| C | 仅由二手页面或题名推断，不用于形成确定结论 |

本矩阵中的“未确认”表示当前可访问的一手材料没有给出足够信息，不代表论文一定没有实现该内容。

## 2. 文献信息

| 方法 | 年份 | 期刊 | DOI/正式来源 | 证据等级 | 代码状态 |
|---|---:|---|---|---|---|
| HDCTI | 2025 | Briefings in Bioinformatics | [10.1093/bib/bbaf399](https://doi.org/10.1093/bib/bbaf399) | A：本地 PDF、正式页面、公开代码 | [tong87-bio/HDCTI](https://github.com/tong87-bio/HDCTI) |
| MVCL-DTI | 2025 | Journal of Chemical Information and Modeling | [10.1021/acs.jcim.4c02073](https://doi.org/10.1021/acs.jcim.4c02073) | B：正式摘要和补充材料目录 | 正式页面未给出代码链接，未确认 |
| DHGT-DTI | 2025 | Journal of Pharmaceutical Analysis | [10.1016/j.jpha.2025.101336](https://doi.org/10.1016/j.jpha.2025.101336) | A：开放全文 | 正式页面未给出代码链接，未确认 |
| HMT-DTI | 2026 | Neural Networks | [10.1016/j.neunet.2025.108093](https://doi.org/10.1016/j.neunet.2025.108093) | B：正式摘要和可访问的引言/方法概述 | 未确认 |
| RSGCL-DTI | 2025 | Briefings in Bioinformatics | [10.1093/bib/bbaf122](https://doi.org/10.1093/bib/bbaf122) | A：[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11932091/) | [tangjlh/RSGCL-DTI](https://github.com/tangjlh/RSGCL-DTI) |

## 3. 方法证据矩阵

| 维度 | HDCTI | MVCL-DTI | DHGT-DTI | HMT-DTI | RSGCL-DTI |
|---|---|---|---|---|---|
| 预测任务 | 中药天然成分—蛋白靶点 | 药物—靶点 | 药物—靶点 | 药物—靶点 | 药物—靶点 |
| 主要输入 | H-C、C-P 标签、P-D | 多类生物子网络构成的异构图 | DTI 与辅助生物网络 | 三个异构生物数据集 | 已知 DTI、药物分子图、蛋白序列 |
| 核心图结构 | H-C 与 P-D 双超图 | 邻居、元路径、扩散三视图异构图 | 邻域与元路径双视图异构图 | 层级元路径异构图 | DTI 派生的药物关系图和蛋白关系图 |
| 局部信息 | 节点—超边—节点传播 | 邻居视图 | 异构 GraphSAGE | 多跳邻居的局部知识提取 | D-MPNN/CNN 结构局部模式与关系图 GCN |
| 全局/高阶信息 | PageRank；全节点自注意力 | 扩散视图、元路径视图 | 元路径 Graph Transformer | 多轮元路径预收集、全局元路径模式 | DTI 关系相似网络；不等同于独立全局视图 |
| 视图融合 | 多层表示求和；节点注意力 | 多视图注意力加权融合 | 局部与元路径表示融合 | 局部与全局知识融合 | 关系特征与结构特征融合 |
| 对比学习 | 无 | attention-based multiview contrastive learning | 无 | 无 | 关系相似图上的图对比学习 |
| 解码/预测 | 点积 + Sigmoid | 当前可访问材料未确认 | 矩阵分解，同时重构 DTI 与辅助网络 | 融合表示送入 predictor，细节未完全确认 | 分类模块融合关系与结构特征 |
| 主要复杂度风险 | 无 mask 全节点注意力为 $O(N^2)$ | 扩散与多视图存储；精确复杂度未确认 | 元路径实例扩展和 Transformer 计算 | 通过预计算与 even-relation propagation 降低反复消息传递 | 药物/蛋白关系相似矩阵与对比训练 |
| 明确解释对象 | PageRank、注意力权重 | 节点/视图注意力和特征可视化 | 元路径注意力与案例研究 | 邻居和元路径重要性 | 关系与结构特征贡献消融 |

## 4. 标签依赖与泄漏风险矩阵

| 方法 | 是否使用已知 DTI 构图 | 论文明确逐折重建标签依赖图吗 | 当前风险判断 |
|---|---|---|---|
| HDCTI 论文描述 | C-P 是监督标签；论文将 compound PageRank 描述为 H-C 侧信息 | 不适用/表述不完整 | 论文结构与代码存在差异，需要分别记录 |
| HDCTI 公开代码 | compound PageRank 实际读取完整 C-P 文件 | 否 | 高风险：测试正边可能通过 PageRank 进入表示 |
| MVCL-DTI | 异构网络包含已知 DTI，并进行已知 DTI masking 实验 | 当前可访问材料未确认 | 中高风险：每折是否重建三视图必须查全文/代码 |
| DHGT-DTI | 预测并重构 DTI 与辅助网络 | 开放正文的随机 5-CV 描述未明确逐折图重建 | 中高风险：随机划分不能自动保证图无泄漏 |
| HMT-DTI | 异构图和元路径包含 DTI 语义关系的可能性高 | 当前可访问材料未确认 | 待核验：预计算若早于数据划分，风险尤其高 |
| RSGCL-DTI | **明确由已知 DTI 构造药物和蛋白关系相似网络** | 正文未明确说明每折重新构造关系网络 | 高风险：关系相似图必须只使用训练正边 |

这里的“风险”表示论文材料没有证明泄漏被排除，不等于断言论文一定发生泄漏。对本项目而言，所有标签依赖图统一执行以下规则：

```text
split pairs first
→ obtain training positive C-P edges
→ rebuild PageRank / C-P-C / P-C-P / similarity / meta-path views
→ train model
→ evaluate untouched test pairs
```

## 5. 评估证据矩阵

| 方法 | 主要划分 | 负样本/困难设置 | 冷启动或跨域 | 统计报告 |
|---|---|---|---|---|
| HDCTI | 随机 5 折 | 随机 1:1 未观测 C-P | disease-aware、跨数据集 | 五折均值与标准差 |
| MVCL-DTI | 正式摘要未给出完整主划分细节 | 1:1/1:10、困难负样本、已知 DTI 不同比例 masking、冗余 DTI | 摘要称鲁棒泛化，但严格实体冷启动未确认 | 补充材料含均值、标准差、95% CI 和统计比较 |
| DHGT-DTI | Luo 与 Zeng 数据集随机 5-CV | 具体负采样规则需结合全文数据段复核 | 未报告严格 compound/target cold-start | 各折 AUC/AUPR 曲线及均值 |
| HMT-DTI | 三个异构生物数据集 | 当前可访问材料未确认 | 正式摘要/引言明确声称进行了 cold-start/cross-domain robustness 实验 | 细节待全文核验 |
| RSGCL-DTI | 固定 10% 测试集；其余 90% 做 10 折训练/验证 | 随机负样本；1:1、1:5、1:10；聚类设置 | 使用聚类方式检查泛化，但并非本文已确认的标准双端 cold-start | 报告均值和标准差 |

不同论文的结果不能直接按数值横向排序，因为它们的数据、负样本比例、候选集合和划分协议不同。

## 6. 与当前修改计划的逐项重叠

| 当前候选模块 | 最近已有工作 | 重叠强度 | 还能否作为主创新 |
|---|---|---:|---|
| 局部编码 + 全局编码 | DHGT-DTI、HMT-DTI | 高 | 单独不能 |
| 邻居 + 元路径 + 扩散视图 | MVCL-DTI | 很高 | 不能照搬 |
| 多视图对比学习 | MVCL-DTI、RSGCL-DTI | 很高 | 单独不能 |
| 门控/注意力融合 | MVCL-DTI、DHGT-DTI | 高 | 只能作为实现组件 |
| 节点自适应传播 | NASNet-DTI，见文献核验文档 | 高 | 不能作为主要新意 |
| MLP/Bilinear decoder | 大量 DTI 方法 | 高 | 可作为低风险改进，不宜作为主要贡献 |
| 逐折无泄漏构图 | 近邻论文常未清楚报告 | 中 | 可作为严谨协议贡献，不是模型结构创新 |
| TCM H-C/P-D 双超图 | HDCTI | 已有 | 是领域基础，不是本项目新增 |
| C-H-D-P 跨超图机制桥接 | 当前五篇中未见相同实现 | 潜在差异 | 仅在 H-D 来源独立时成立 |
| 未观测 C-P 的 PU/困难负样本 | MVCL-DTI 已做困难负样本，PU 未见 | 部分重叠 | 需突出 PU 或 TCM 特定采样依据 |

## 7. 新颖性结论

当前计划中的以下表述不能直接作为论文核心贡献：

> 提出局部—全局多视图图网络，并通过门控融合和对比学习增强 DTI 表示。

原因是 MVCL-DTI、DHGT-DTI 和 HMT-DTI 已分别覆盖了该组合的大部分技术元素。

更可辩护的研究方向原本是：

> 在严格逐折构图下，使用来源独立的 H-D 关系形成 C-H-D-P 机制桥接，连接 HDCTI 的 H-C 与 P-D 双超图，并在疾病感知和困难候选场景中验证其作用。

但 [H_D_SOURCE_AUDIT.md](H_D_SOURCE_AUDIT.md) 表明当前 TCM-Suite、TCMSP 和 ETCM2.0 的 H-D 不能视为与 C-P 标签独立的外部证据。因此，在获得独立 H-D 数据前，这一方向也不能直接成立。

## 8. 建议形成的研究主线

### 当前可立即实施

```text
Strict-HDCTI
+ fold-safe 标签依赖视图
+ candidate-level H-C / P-D 上下文交互
+ 未观测关系的 PU 或结构困难训练
```

候选级上下文交互只使用来源相对独立的 H-C 和 P-D，不将现有 H-D 当作固定先验；C-P 派生视图必须逐折重建。

### 获得独立 H-D 后再实施

```text
Strict-HDCTI
+ independently curated H-D
+ C-H-D-P cross-hypergraph bridge
+ path-level explanation
```

独立 H-D 应具有数据库/文献来源、证据类型、获取日期和置信等级，不能只由完整 C-P 与 P-D 链路推导。

## 9. 下一步

1. 修改 [修改计划.md](修改计划.md)，删除“局部—全局 + 对比 + 门控”作为默认主创新的表述。
2. 将 H-D 路径降为“需独立数据后启用”的扩展模块。
3. 把第一项代码工作固定为 Strict 数据划分和逐折构图基础设施。
4. 在 Strict 基础上只选择一个结构创新：优先候选级 H-C/P-D 上下文交互。
5. 把 PU/困难负样本作为独立训练增强，不与结构模块同时首次加入。

