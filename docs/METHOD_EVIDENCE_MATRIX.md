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

`Hctx-P + CHCR + SDIS` 不是最终 `Ours-full`。冻结 cold-start 组合实验中，TCM-Suite AUPR 下降 `0.019451`，超过预注册最大退化 `0.005`。论文必须按任务协议分别报告 CHCR 和 SDIS。

## 4. 核心主张证据矩阵

| ID | 方法或工作 | 允许主张 | 主要证据 | 等级 | 关键边界 | 来源 |
|---|---|---|---|---|---|---|
| P1 | Strict 数据与评估协议 | 每折 C-P 图统计仅使用训练正边，负样本、fold、seed 和实体 ID 固定且可审计 | 固定 manifest、训练 C-P PageRank、二部节点类型隔离、H-D 关闭及泄漏测试 | A（协议） | 属于可信评估基础，不是模型结构创新 | [修改计划](修改计划.md)、[HDCTI 论文笔记](HDCTI_PAPER_NOTES.md) |
| M1 | 候选级 Hctx-P | 显式药材上下文—靶点交互能够补充独立双超图编码，并在多数数据库及 compound cold-start 下改善排序 | 无稠密注意力四库随机边五折 AUPR 增量 `-0.000255/+0.011325/+0.014082/+0.011847`，macro `+0.009250`，TCMSP/SymMap/ETCM 均为 5/5 folds 提高 | A | TCM-Suite 仅 1/5 folds 提高且均值轻微下降；不能声称四库全部改善 | [HCTX_NO_DENSE_ABLATION](HCTX_NO_DENSE_ABLATION.md)、[CONTEXT_INTERACTION](CONTEXT_INTERACTION.md) |
| M2-P | CHCR 性能贡献 | CHCR 是普通随机边协议下跨库非劣、在 ETCM 上更明显的训练期上下文正则 | 无稠密注意力四库五折 AUPR 增量 `+0.000408/+0.001107/+0.000039/+0.006329`，macro `+0.001971` | A | SymMap 基本持平；不能声称四库大幅提高或所有分类指标改善 | [UNIFIED_NO_DENSE_CHCR](UNIFIED_NO_DENSE_CHCR.md) |
| M2-M | CHCR 机制证据 | 在具备稳定支持的环境中，冻结 Hctx-P 使用了超出 H-C degree 的上下文信息 | 同 degree donor 四库 20-fold 纯推理：TCM-Suite、TCMSP、ETCM 为 5/5 folds 支持 | C | SymMap 仅 1/5 folds 通过；四库普适机制判定为 No-Go | [CHCR_DONOR_CONTROLS](CHCR_DONOR_CONTROLS.md) |
| M2-S | CHCR 支持度边界 | CHCR/Hctx-P 的上下文可靠性受 H-C 与训练 C-P 支持度调节 | SymMap 在 `H-C degree=1`、训练 `C-P degree=0/1-2` 方向不一致；TCM-Suite 在训练 `C-P degree=1-2/3-5` 也不稳定 | C | 这是失败模式定位，不等于已经实现或验证自适应路由 | [CHCR_DONOR_CONTROLS](CHCR_DONOR_CONTROLS.md) |
| M3-P | SDIS 排序贡献 | SDIS 改善 compound cold-start 下零训练 C-P 支持实体的归纳排序 | 四库五折 AUPR 增量 `+0.059305/+0.022891/+0.012215/+0.017686`，macro `+0.028024`，20/20 folds 提高 | A | 只适用于 compound cold-start；不是普通随机边默认模块 | [SELF_EXCLUDED_HERB_CONTEXT_AUDIT](SELF_EXCLUDED_HERB_CONTEXT_AUDIT.md) |
| M3-C | SDIS 校准分类 | 固定 0.5 阈值下的 F1 下降主要来自分数尺度变化，inner-validation 阈值可恢复分类表现 | 纯推理阈值校准后四库 F1 均提高，macro `+0.029535`，20/20 folds 提高 | A/C | 校准阈值必须逐折仅由 inner-validation 选择；固定 0.5 结果仍需披露 | [SELF_EXCLUDED_HERB_CONTEXT_AUDIT](SELF_EXCLUDED_HERB_CONTEXT_AUDIT.md) |
| F1 | 场景化统一框架 | CHCR 与 SDIS 的切换由监督可用性和预定义评估协议决定，不由数据库结果决定 | 两套四库冻结协议；组合实验显式 No-Go | A | 不能把两个场景最优结果拼成同一 `Ours-full` 行 | [修改计划](修改计划.md)、[SELF_EXCLUDED_HERB_CONTEXT_AUDIT](SELF_EXCLUDED_HERB_CONTEXT_AUDIT.md) |
| D1 | ETCM2.0 数据工作 | 构建具有实体映射、关系审计和剪枝依据的 ETCM2.0 CTI 数据集，用于外部验证和案例研究 | mention10/core 构建、数据统计、关系交集与映射审查 | A（数据） | mention10 是证据频次过滤；不能声称覆盖 ETCM2.0 全部实体 | [DATASET_STATISTICS](DATASET_STATISTICS.md)、[ETCM2_CORE_NOTES](ETCM2_CORE_NOTES.md) |

## 5. 允许与禁止表述

| 主题 | 允许表述 | 禁止表述 |
|---|---|---|
| Hctx-P | 建立候选 compound 药材上下文与候选 target 的显式交互；四库 macro AUPR 提高且 3/4 数据库稳定受益 | 四库全部提高或显著提高；首次使用药材信息；完整模拟了生物结合机制 |
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

### 表 6：效率与复杂度

```text
参数量
训练时间
推理时间
峰值显存
稠密注意力移除后的复杂度变化
```

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
| 阻塞，部分完成 | 最终主结果、消融和场景表尚未统一生成 | 容易混用历史 attention、epoch 或 split 口径 | Hctx-P 直接消融配对表已完成；下一步统一 Strict/Hctx-P/CHCR 与 cold-start Hctx-P/SDIS 主表 |
| 阻塞 | 参数量、单 epoch 时间、推理时间和峰值显存未形成统一表 | 期刊审稿会质疑 CHCR 训练成本与 SDIS 部署代价 | 冻结硬件和 batch，执行轻量复杂度审计 |
| 阻塞 | ETCM Top-K 案例仍缺最终外部证据闭环 | 数据贡献和中医药解释目前弱于方法实验 | 使用实体映射选取少量高置信候选，记录检索日期与证据等级 |
| 加强 | 除 ETCM CHCR 外，其他最终配置主要为单训练 seed | fold 方差不能代表初始化稳定性 | 在主表冻结后选择一个代表库补 3 seed，或在局限性中明确披露 |
| 可选 | disease-aware / target cold-start 未形成四库最终结果 | 可增强对原论文和困难泛化场景的覆盖 | 仅在主表完成且计算预算允许时追加，不阻塞当前模型冻结 |

## 9. 当前决策

模型搜索继续冻结，不重新开启 SACR、同族注意力、donor、margin 或数据集特定路由调参。

四库无稠密注意力随机边 Hctx-P 直接消融已经完成：

```text
Strict-HDCTI, Hctx-P off
vs
Strict-HDCTI + Hctx-P, Hctx-P on
```

两组复用了现有 `no_dense_chcr_full` 批次中的 Hctx-P 配置、split manifest、seed、inner-validation、早停和 Dot decoder，只补跑缺失的 NoContext 一侧。最终 AUPR 增量为 TCM-Suite `-0.000255`、TCMSP `+0.011325`、SymMap2.0 `+0.014082`、ETCM2.0 mention10 `+0.011847`，macro `+0.009250`，冻结判定为 **PASS**。下一项任务是生成两种协议的最终统一结果表，而不是继续增加新模型模块。

执行协议、预注册门槛和输出文件见 [HCTX_NO_DENSE_ABLATION.md](HCTX_NO_DENSE_ABLATION.md)。当前实现会在运行前校验四库配置 SHA-256，并只允许 `model.variant`、`context.interaction` 与 `context.herb_protein` 三项不同；完成后自动输出逐折配对结果和 `PASS/NO-GO` 判定。
