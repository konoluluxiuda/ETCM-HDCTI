# 方法—证据—主张矩阵

## 1. 文档目的

本文档冻结当前论文的方法边界，防止后续写作混淆以下内容：

1. 数据与评估协议工作；
2. 真正的模型设计；
3. 普通随机边与 compound cold-start 两种不同任务；
4. 预测性能证据与机制解释证据；
5. 已通过结果、描述性观察与明确 No-Go。

本文档不新增模型结论。所有数值均来自当前仓库已经冻结的配置、checkpoint、结果日志或审计文档。

## 2. 证据等级

| 等级 | 定义 | 可用于何处 |
|---|---|---|
| A | 四库完整五折、匹配配置、冻结判据通过 | 主结果与核心贡献 |
| B | 单库完整五折或四库单折预注册 Pilot | 消融、可行性和补充证据 |
| C | 冻结 checkpoint 纯推理机制审计 | 机制解释，不能替代性能实验 |
| D | 描述性结果或历史非统一配置 | 讨论与背景，不进入最终主比较 |
| No-Go | 未通过预注册门槛 | 必须披露，禁止包装为有效模块 |
| Gap | 当前证据缺失或口径不匹配 | 投稿前需要补齐或明确降级主张 |

五个 cross-validation folds 反映划分差异，不等同于五次独立训练重复。除已完成的 ETCM CHCR 三 seed 结果外，不将 fold 标准差写成初始化稳定性证据。

## 3. 冻结方法结构

### 3.1 共享骨干

```text
Strict 逐折无泄漏构图
+ H-C / P-D 双超图编码
+ 候选级 Hctx-P
+ Dot decoder
+ attention.max.nodes=0
```

候选分数中的显式药材上下文项为：

$$
s_{HP}(c,p)=(h_c\odot w_{HP})^Tz_p
$$

其中 $h_c$ 只由候选 compound 的 H-C 药材上下文构造，$z_p$ 来自 P-D 侧蛋白表示；不读取 H-D 或测试 C-P 标签。

### 3.2 普通 Strict 随机边配置

```text
共享骨干 + CHCR
```

CHCR 只改变训练目标：对已知训练正样本构造同 H-C degree 的反事实上下文，并约束事实上下文得分高于反事实上下文。部署推理结构仍为共享 Hctx-P 骨干。

### 3.3 Strict compound cold-start 配置

```text
共享骨干 + SDIS
```

当训练折中 compound 的 C-P 正边支持度为 0，且存在可用 H-C 上下文时，SDIS 确定性关闭不可靠的 compound-ID 基础分。该规则只依赖训练支持状态，不按数据库或测试结果切换。

### 3.4 禁止的伪统一配置

`Hctx-P + CHCR + SDIS` 不是最终 `Ours-full`。冻结 cold-start 组合实验中，TCM-Suite AUPR 下降 `0.019451`，超过预注册最大退化 `0.005`。为解决共享上下文参数的 warm/cold 负迁移，唯一重新开放的 SCHE 双专家 Pilot 也未通过：TCM-Suite fold 1 GPU inner-validation AUPR 为 `0.650499`，低于冻结 SDIS 的 `0.669984`。因此论文必须按任务协议分别报告 CHCR 和 SDIS，不再搜索统一门控或专家结构。

## 4. 核心主张证据矩阵

| ID | 方法或工作 | 允许主张 | 主要证据 | 等级 | 关键边界 | 来源 |
|---|---|---|---|---|---|---|
| P1 | Strict 数据与评估协议 | 每折 C-P 图统计仅使用训练正边，负样本、fold、seed 和实体 ID 固定且可审计 | 固定 manifest、训练 C-P PageRank、二部节点类型隔离、H-D 关闭及泄漏测试 | A（协议） | 属于可信评估基础，不是模型结构创新 | [修改计划](修改计划.md)、[HDCTI 论文笔记](HDCTI_PAPER_NOTES.md) |
| M1 | 候选级 Hctx-P | 显式药材上下文—靶点交互能够补充独立双超图编码，并为 side-information-assisted compound cold-start 提供可迁移分数 | 随机边四库 AUPR macro `+0.009250`；cold-start 相对 NoContext 四库增量 `+0.203622/+0.584248/+0.407087/+0.556348`，macro `+0.437826`，20/20 folds 提高 | A | 随机边 TCM-Suite 轻微下降；cold-start 使用完整 H-C 侧信息，且 NoContext 是非归纳原模型，不代表现代属性式 cold-start 基线 | [HCTX_NO_DENSE_ABLATION](HCTX_NO_DENSE_ABLATION.md)、[cold-start 递进消融](COLD_START_HCTX_ABLATION.md) |
| M2-P | CHCR 性能贡献 | CHCR 是普通随机边协议下跨库非劣、在 ETCM 上更明显的训练期上下文正则 | 无稠密注意力四库五折 AUPR 增量 `+0.000408/+0.001107/+0.000039/+0.006329`，macro `+0.001971` | A | SymMap 基本持平；不能声称四库大幅提高或所有分类指标改善 | [UNIFIED_NO_DENSE_CHCR](UNIFIED_NO_DENSE_CHCR.md) |
| M2-M | CHCR 机制证据 | 在具备稳定支持的环境中，冻结 Hctx-P 使用了超出 H-C degree 的上下文信息 | 同 degree donor 四库 20-fold 纯推理：TCM-Suite、TCMSP、ETCM 为 5/5 folds 支持 | C | SymMap 仅 1/5 folds 通过；四库普适机制判定为 No-Go | [CHCR_DONOR_CONTROLS](CHCR_DONOR_CONTROLS.md) |
| M2-S | CHCR 支持度边界 | CHCR/Hctx-P 的上下文可靠性受 H-C 与训练 C-P 支持度调节 | SymMap 在 `H-C degree=1`、训练 `C-P degree=0/1-2` 方向不一致；TCM-Suite 在训练 `C-P degree=1-2/3-5` 也不稳定 | C | 这是失败模式定位，不等于已经实现或验证自适应路由 | [CHCR_DONOR_CONTROLS](CHCR_DONOR_CONTROLS.md) |
| M3-P | SDIS 排序贡献 | SDIS 改善 compound cold-start 下零训练 C-P 支持实体的归纳排序 | 四库五折 AUPR 增量 `+0.059305/+0.022891/+0.012215/+0.017686`，macro `+0.028024`，20/20 folds 提高 | A | 只适用于 compound cold-start；不是普通随机边默认模块 | [SELF_EXCLUDED_HERB_CONTEXT_AUDIT](SELF_EXCLUDED_HERB_CONTEXT_AUDIT.md) |
| M3-C | SDIS 校准分类 | 固定 0.5 阈值下的 F1 下降主要来自分数尺度变化，inner-validation 阈值可恢复分类表现 | 纯推理阈值校准后四库 F1 均提高，macro `+0.029535`，20/20 folds 提高 | A/C | 校准阈值必须逐折仅由 inner-validation 选择；固定 0.5 结果仍需披露 | [SELF_EXCLUDED_HERB_CONTEXT_AUDIT](SELF_EXCLUDED_HERB_CONTEXT_AUDIT.md) |
| F1 | 场景化双配置证据 | CHCR 与 SDIS 分别说明随机边上下文正则和 cold-start 支持度失配 | 两套四库冻结协议；组合实验显式 No-Go | 补充 | 作者已决定双配置不构成最终统一方法，不能写为 `Ours-full` 或核心框架贡献 | [最终方法统一性决策](UNIFIED_METHOD_DIRECTION.md)、[SELF_EXCLUDED_HERB_CONTEXT_AUDIT](SELF_EXCLUDED_HERB_CONTEXT_AUDIT.md) |
| C1 | SCHE 统一候选 | 独立 warm/cold Hctx-P 参数与逐样本支持度路由未能消除共享参数冲突 | TCM-Suite cold-start fold 1 GPU inner-validation AUPR `0.650499`，低于冻结 SDIS `0.669984`，差值 `-0.019485`；CPU 复现为 `0.650017` | No-Go | 不进入四库，不搜索 ratio、seed、margin、weight、soft gate 或数据库特定参数；不能写入最终贡献 | [SUPPORT_CONDITIONED_DUAL_EXPERT](SUPPORT_CONDITIONED_DUAL_EXPERT.md) |
| C2 | 支持掩码 episodic training | 用 warm compound 制造零 C-P 支持的 pseudo-cold episode 具有工程合理性，但不能作为独立新机制 | DropoutNet、PT-GNN 和 CGRC 已分别覆盖协同输入 dropout、warm-to-cold episode 与全交互边掩码重建；CLCRec/ALDI 覆盖对比和蒸馏变体 | No-Go（创新性） | 不实现、不进入 Pilot；若后续作为训练技巧使用，必须引用近邻工作并降级表述 | [SUPPORT_MASKED_EPISODIC_AUDIT](SUPPORT_MASKED_EPISODIC_AUDIT.md) |
| C3 | HPLGA 线性全局注意力 | 用 H-C/P-D 超图 PageRank 调制核化线性全局读取，以线性复杂度补回被删除的全节点感受野 | Gate 0 通过；四库 fold 1 validation AUPR 增量 `+0.000448/-0.001326/-0.004159/-0.000067`，macro `-0.001276`，仅 1/4 提高 | No-Go | 分支残差尺度均已激活；停止完整五折、cold-start Gate 和所有结构/参数搜索，不能列为第三机制 | [HYPERGRAPH_PAGERANK_LINEAR_ATTENTION](HYPERGRAPH_PAGERANK_LINEAR_ATTENTION.md) |
| C4 | H-C/P-D 侧关系辅助重构 | 用侧关系拓扑保持约束 compound、protein 与超边表示 | NeoDTI 已对全部异构关系使用关系特定投影和联合边重构，并报告只重构 DTI 时 AUPR 下降 `5.5%`；后续 DTI 图自编码器继续沿用该范式 | No-Go（创新性） | 串联门槛第一关失败，不执行冻结表示 probe、训练实现或损失权重搜索；只能作为有出处的普通正则 | [侧关系重构审计](SIDE_RELATION_RECONSTRUCTION_AUDIT.md) |
| D1 | ETCM2.0 数据工作 | 构建具有实体映射、关系审计和剪枝依据的 ETCM2.0 CTI 数据集，用于外部验证和案例研究 | mention10/core 构建、数据统计、关系交集与映射审查 | A（数据） | mention10 是证据频次过滤；不能声称覆盖 ETCM2.0 全部实体 | [DATASET_STATISTICS](DATASET_STATISTICS.md)、[ETCM2_CORE_NOTES](ETCM2_CORE_NOTES.md) |
| D2 | ETCM2.0 Top-K 独立核验 | 冻结候选中存在可由外部实验支持的预测，同时高排名候选也可能与直接实验冲突 | 检索前冻结 15 个 pair，完成 45 个 BindingDB/ChEMBL/PubMed 查询；B1 2 条、E 12 条、Conflict 1 条 | B（案例） | `2/15` 不是总体 precision；E 不是确认负例；页面路径不是独立 C-P 证据 | [ETCM Top-K 核验](ETCM_TOPK_MANUAL_VALIDATION.md)、[代表案例](ETCM_REPRESENTATIVE_CASES.md) |

## 5. 允许与禁止表述

| 主题 | 允许表述 | 禁止表述 |
|---|---|---|
| Hctx-P | 随机边上四库 macro AUPR 提高；side-information-assisted compound cold-start 上四库 20/20 folds 提高 | 随机边四库全部提高；不依赖侧信息的 de novo cold-start；优于所有现代归纳模型 |
| CHCR 性能 | 四库随机边 AUPR 均值非下降，主要增益集中于 ETCM | 四库均显著提升；对任意数据库都有效 |
| CHCR 机制 | 三个数据库支持超出 H-C degree 的上下文特异性 | 已在四库排除全部度数与研究热度偏倚 |
| 反事实 donor | 合成上下文扰动与训练正样本的 ranking regularization | 生物学真实负药材上下文；因果干预证据 |
| SDIS | compound cold-start 下关闭零训练支持的不可迁移 ID 基础分 | 通用冷启动；无侧信息新实体归纳；所有指标均提高 |
| 场景切换 | 由 pair-stratified 或 compound cold-start 协议预先触发 | 根据每个数据库结果挑选最优模块 |
| ETCM2.0 | mention10 核心子集上的外部验证 | ETCM2.0 全库无偏代表性结果 |

## 6. No-Go 与负结果矩阵

| 分支 | 冻结结论 | 论文用途 |
|---|---|---|
| Bilinear / MLP decoder | 均未优于 Dot Pilot | Decoder 消融；保留 Dot |
| C-Dctx / Hctx-Dctx | 未提供 Hctx-P 之外的稳定收益 | 证明最终上下文设计的简约性 |
| Target-conditioned Herb Attention V1/V2 | 动态注意力已激活，但未优于静态 Hctx-P | 讨论候选级注意力并非越复杂越好 |
| Mixed hard negatives / PU | Pilot 非劣性失败，Top 未标注候选缺少可信阳性 | 不进入最终方法 |
| CMIT / CCD | 辅助或蒸馏目标改善部分 masked 能力，但损害/未改善主任务 | 补充材料中的失败分析 |
| SACR / support router | 四库 Pilot macro 不增益，SymMap 明显退化 | 禁止重新包装为已解决的 support-aware 路由 |
| SP-FBHA / HILGA / RG-SHADG | 跨库方向不稳定或冻结角色审计失败 | 不进入主模型 |
| Top-K 全局扩散 / 超边 IDF 重加权 | 缺乏结构新颖度或表示变化过小 | 说明停止同族图结构修补的依据 |
| 独立 H-D 路径 | H-D 来源审计未满足独立先验要求 | 仅作 post-hoc 假设生成 |
| Direct self-exclusion | 相对 SD-only 为 0/4 提高，macro `-0.025989` | 作为 SDIS 消融 No-Go |
| SDIS + CHCR | TCM-Suite `-0.019451`，违反单库退化上限 | 必须披露，支持场景化配置而非插件堆叠 |
| SCHE warm/cold 双专家 | TCM-Suite fold 1 GPU validation AUPR `0.650499`，相对冻结 SDIS `-0.019485`；CPU 复现差异仅 `0.000482` | 预注册首门槛即失败，停止四库和统一模型主张 |
| 支持掩码 episodic training | 与 DropoutNet、PT-GNN、CGRC 的 cold-start simulation 核心动作高度同构 | 在创新性门槛停止，不以改名方式进入新 Pilot |
| H-C/P-D 侧关系辅助重构 | NeoDTI 已联合重构 DTI 与其他异构关系，核心训练动作高度同构 | 在创新性门槛停止；不运行 frozen probe 或训练 Pilot |
| CHCR 四库普适机制 | SymMap 仅 1/5 folds 通过 donor-control | 限制机制主张，不否定其已观察的性能结果 |

## 7. 论文表格映射

### 表 1：数据集与协议

```text
四库实体/关系统计
随机边与 compound cold-start 定义
正负样本比例
Strict 逐折构图和 H-D 使用状态
```

### 表 2：普通 Strict 随机边主结果

```text
Strict-HDCTI
Strict-HDCTI + Hctx-P
Strict-HDCTI + Hctx-P + CHCR
```

主要指标为 AUC/AUPR；固定阈值 Precision/Recall/F1 同时保留。CHCR 使用现有四库五折冻结结果。

### 表 3：Compound cold-start 主结果

```text
NoContext
Hctx-P
Hctx-P + SDIS
```

同时报告固定 0.5 阈值和 inner-validation 校准阈值结果，不能只展示校准后的 F1。

### 表 4：核心消融

```text
w/o Hctx-P
w/o CHCR（随机边）
w/o SDIS（cold-start）
SDIS + self-exclusion
SDIS + CHCR（No-Go）
```

不同协议的消融分区展示，不计算跨协议的统一 macro。

### 表 5：机制与支持度分析

```text
CHCR donor-control 四库结果
H-C degree / training C-P degree 分层
SDIS zero-support eligible/ineligible 分组
```

### 补充说明：效率与复杂度

```text
新增参数的理论数量
CHCR 仅增加训练期反事实打分
SDIS 为无参数确定性门控
稠密注意力移除后的复杂度变化
```

当前不为该部分重新运行四库硬件 benchmark。若目标期刊或审稿人明确要求，再在固定单机环境下补参数量、单 epoch 时间、推理时间和峰值显存；这些指标不作为当前投稿前阻塞项。

### 表 7：ETCM2.0 案例研究

```text
Top-K compound-target 候选
已知关系留出命中
外部数据库/文献证据
Herb context 与可解释路径
```

## 8. 投稿前证据缺口

| 优先级 | 缺口 | 为什么重要 | 下一动作 |
|---|---|---|---|
| 已完成 | 最终 `attention.max.nodes=0` 下四库匹配的 `Strict-HDCTI vs Hctx-P` 普通随机边五折直接消融 | M1 是共享骨干创新，必须有最终统一口径直接证据 | 冻结判定 PASS：macro AUPR `+0.009250`，3/4 数据库不下降且达到逐折方向门槛 |
| 已完成 | 最终主结果、消融和场景表统一生成 | 避免混用历史 attention、epoch 或 split 口径 | 已通过冻结来源与配置哈希生成随机边 Strict/Hctx-P/CHCR、cold-start Hctx-P/SDIS 固定阈值及校准阈值表，见 `FINAL_RESULTS_TABLES.md` |
| 已完成 | 四库统一 cold-start NoContext 完整五折 | 同一协议已形成 `NoContext -> Hctx-P -> Hctx-P + SDIS` 递进证据 | Hctx-P 相对 NoContext macro AUPR `+0.437826`、20/20 folds 提高；SDIS 再提高 `+0.028024`，见 [cold-start 递进消融](COLD_START_HCTX_ABLATION.md) |
| 可选 | 未形成统一硬件复杂度 benchmark | 可能用于回应 CHCR 训练成本与 SDIS 部署代价，但不影响主要有效性结论 | 正文仅报告理论增量：Hctx-P 少量参数、CHCR 仅训练期开销、SDIS 无参数；审稿明确要求时再补硬件实测 |
| 已完成 | ETCM Top-K 外部证据闭环 | 数据贡献和中医药解释需要独立于训练数据的证据边界 | 15 个冻结 pair 的 45 个查询已完成；已冻结 2 个 B1 正向案例和 1 个 Conflict 失败案例，待制作论文图 |
| 已完成 | 同一 Strict 协议下的外部同输入对比表 | Dual-HGNN-CTI、LightGCN-CTI、R-GCN-CTI 与稀疏 HGT-CTI 的四库 16 个外层五折任务全部完成；R-GCN 是四个基线中最强者。最终随机边模型相对 R-GCN 的 macro AUPR 为 `+0.001408`，在 SymMap/ETCM 较高、TCMSP 基本持平、TCM-Suite 略低 | 两个冻结结果源及逐行配置哈希已接入 `FINAL_RESULTS_TABLES.md`；HGT 使用统一 64 入邻居上限，其 ETCM 结果不代表无采样完整 HGT。停止库特定调参；作者材料若到达，再追加可原样复现的属性模型 |
| 加强 | 除 ETCM CHCR 外，其他最终配置主要为单训练 seed | fold 方差不能代表初始化稳定性 | 在主表冻结后选择一个代表库补 3 seed，或在局限性中明确披露 |
| 可选 | disease-aware / target cold-start 未形成四库最终结果 | 可增强对原论文和困难泛化场景的覆盖 | 仅在主表完成且计算预算允许时追加，不阻塞当前模型冻结 |
| 已完成，No-Go | HPLGA 模型门槛 | Gate 0 工程与复杂度通过，但四库 Gate 1 macro AUPR `-0.001276`，且 SymMap2.0 下降超过预注册上限 | 停止完整五折和统一 cold-start；保留实现作为负结果，不再调参 |

## 9. 当前决策

既有 No-Go 路线继续冻结，不重新开启 SACR、donor、margin、pseudo-cold 或
数据集特定路由调参。HPLGA 已在预注册 Gate 1 判定 No-Go；H-C/P-D 侧关系
辅助重构也因与 NeoDTI 的多关系拓扑重构高度同构，在创新性门槛停止。当前没有
开放的第三模型候选。

四库无稠密注意力随机边 Hctx-P 直接消融已经完成：

```text
Strict-HDCTI, Hctx-P off
vs
Strict-HDCTI + Hctx-P, Hctx-P on
```

两组复用了现有 `no_dense_chcr_full` 批次中的 Hctx-P 配置、split manifest、seed、inner-validation、早停和 Dot decoder，只补跑缺失的 NoContext 一侧。最终 AUPR 增量为 TCM-Suite `-0.000255`、TCMSP `+0.011325`、SymMap2.0 `+0.014082`、ETCM2.0 mention10 `+0.011847`，macro `+0.009250`，冻结判定为 **PASS**。下一项任务是生成两种协议的最终统一结果表，而不是继续增加新模型模块。

两种协议的统一结果表现已生成，见 [FINAL_RESULTS_TABLES.md](FINAL_RESULTS_TABLES.md)。生成器 `tools/build_paper_results_tables.py` 校验冻结结果 SHA-256、逐行配置 SHA-256、Strict 协议、split、seed、Dot decoder、`attention.max.nodes=0` 以及校准前后 AUC/AUPR 一致性。cold-start 主表现已包含匹配的 NoContext、Hctx-P 和 Hctx-P+SDIS 三阶段结果。

ETCM Top-K 证据闭环也已完成，见
[ETCM_TOPK_MANUAL_VALIDATION.md](ETCM_TOPK_MANUAL_VALIDATION.md)。两个 B1
正向案例和一个 Conflict 失败案例已经按证据等级自动冻结，不根据模型分数或
路径数量替换。同一 Strict 协议下的四种外部拓扑基线和代表案例图也均已完成。
已有方法实验和外部同输入基线结果均冻结，但最终方法尚未冻结。唯一重新开放的
SCHE 统一模型 Pilot 已按预注册门槛判定 No-Go；作者同时否决将 CHCR 与 SDIS
按两个场景拼接为最终框架。研究问题已经收窄为 compound cold-start。共享
Hctx-P + SDIS 的支持掩码 episodic training 也已完成近邻工作审计，并因与
DropoutNet、PT-GNN、CGRC 等方法高度同构而在创新性门槛判定 No-Go，不再进入
单折或四库实验。当前已冻结主模型仍为 `Hctx-P + SDIS`。为补足第三项模型
机制而实现的 HPLGA 已完成 Gate 0/1：工程与复杂度检查通过，但四库 macro
AUPR 下降 `0.001276`，仅 TCM-Suite 提高，SymMap2.0 下降 `0.004159`。因此
不进入统一 cold-start Gate，最终模型仍冻结为 `Hctx-P + SDIS`。

四库 cold-start NoContext 完整五折也已补齐。Hctx-P 相对 NoContext 的 AUPR
增量为 `+0.203622/+0.584248/+0.407087/+0.556348`，macro `+0.437826`，
20/20 folds 同向；SDIS 在同一骨干上继续提高 macro `+0.028024`。因此当前
两个模型机制已经在同一 side-information-assisted compound cold-start 问题中
形成明确递进，不再依赖把 CHCR 和 SDIS 拼接为双场景框架。

执行协议、预注册门槛和输出文件见 [HCTX_NO_DENSE_ABLATION.md](HCTX_NO_DENSE_ABLATION.md)。当前实现会在运行前校验四库配置 SHA-256，并只允许 `model.variant`、`context.interaction` 与 `context.herb_protein` 三项不同；完成后自动输出逐折配对结果和 `PASS/NO-GO` 判定。

## 10. 2026-08-04 V3 冻结上下文专家更新

本节覆盖本文档中“当前没有开放的第三模型候选”和“最终方法仅为
`Hctx-P + SDIS`”的旧状态描述，但不改变 SCHE、V2 四状态联合重训或
Hctx-Dctx residual 的 No-Go 结论。

新 V3 与上述失败方案的关键区别是：先冻结 NoContext base checkpoint，再只用
inner unit 训练隔离的 Hctx-P linear head；推理时由训练支持状态执行固定路由：

```text
WW = frozen base + frozen Hctx-P head
CW = frozen Hctx-P head
WC = frozen base
CC = frozen base
```

四库 inner Gate 全部通过后，模型、head、epoch 与路由在查看 outer 指标前完成
冻结。独立 outer-unit 评价未执行训练或参数选择，结果如下：

| 数据集 | NoContext Macro-AUPR | V3 Macro-AUPR | 差值 | WW | CW | WC | CC | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite | 0.571853 | 0.617732 | +0.045880 | +0.018818 | +0.164701 | +0.000000 | +0.000000 | PASS |
| TCMSP | 0.591204 | 0.709745 | +0.118541 | +0.005624 | +0.468538 | +0.000000 | +0.000000 | PASS |
| SymMap2.0 | 0.559741 | 0.658774 | +0.099033 | +0.024080 | +0.372052 | +0.000000 | +0.000000 | PASS |
| ETCM2.0-mention10 | 0.555118 | 0.685350 | +0.130232 | +0.010399 | +0.510530 | +0.000000 | +0.000000 | PASS |

四库 Macro-AUPR 平均提升 `+0.098421`，WC/CC 与 NoContext 精确一致；base 与
head 哈希在评价前后保持不变。当前将其记为 **B 级外层四库单 unit 证据**：
可以冻结为“支持状态感知的冻结上下文专家”候选机制，并支持其专门改善 CW
状态的主张；尚不能按本文 A 级定义替代四库完整多折/重复 outer 证据，也不能
宣称改善 target-cold 或 double-cold。

完整协议、哈希和边界见 [SUPPORT_STATE_ROUTING](SUPPORT_STATE_ROUTING.md)。
后续禁止针对本次 outer unit 修改路由或 head；若升级为最终核心贡献，只能在
预先生成的新 outer units 上进行固定方法复验。
