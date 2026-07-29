# 最终方法统一性决策

## 1. 决策

截至 2026-07-29，不能将

```text
普通随机边：Hctx-P + CHCR
Compound cold-start：Hctx-P + SDIS
```

包装成一个统一最终模型。两套配置分别在不同协议下取得较好结果，属于场景化
实验发现，不是同一套训练和推理算法的联合证据。即使场景切换由协议预先定义，
审稿人仍可合理质疑：

1. 最终方法没有唯一配置和唯一算法；
2. CHCR 与 SDIS 无法同时稳定工作；
3. 模块选择与评价场景绑定，容易被解读为结果驱动的配置选择；
4. 每个分支单独承担的创新和实验深度不足。

因此，场景化双配置不再作为最终论文方案。已有实验不会删除，但应降为方法边界、
补充实验和失败分析。

## 2. 证据依据

| 证据 | 结果 | 决策含义 |
|---|---|---|
| Hctx-P 随机边四库消融 | macro AUPR `+0.009250`，3/4 数据库稳定提高 | 可保留为共享结构方法 |
| CHCR 随机边四库消融 | macro AUPR `+0.001971`，增益主要来自 ETCM | 只保留为辅助训练正则，不单列核心创新 |
| CHCR donor-control | 3/4 数据库完整支持，SymMap 仅 1/5 folds 通过 | 机制主张存在明显适用边界 |
| SDIS compound cold-start | macro AUPR `+0.028024`，20/20 folds 提高 | 是当前最强、最稳定的方法证据 |
| SDIS + CHCR | TCM-Suite AUPR `-0.019451` | 两个机制不能直接组成最终配置 |
| SCHE 统一双专家 | GPU fold 1 AUPR `0.650499`，低于 SDIS `0.669984` | 独立 warm/cold head 也未解决统一问题 |

SCHE 的 cold 参数已经有效更新，CPU/GPU 结果一致，因此失败不能解释为模块未
激活或硬件偶然性。

## 3. 贡献层级重新划分

当前允许保留的工作分为：

| 层级 | 内容 | 当前定位 |
|---|---|---|
| 实验基础 | Strict 划分、逐折构图、固定 manifest、统一指标 | 可信评估和复现贡献，不冒充模型创新 |
| 共享结构 | 候选级 Hctx-P | 已验证的主要结构方法 |
| 冷启动机制 | SDIS | 当前最强方法候选，只适用于 side-information-assisted compound cold-start |
| 辅助实验 | CHCR | 训练期正则和机制分析，不再作为独立核心创新 |
| 数据与解释 | ETCM2.0 mention10、外部 evidence、Top-K 案例 | 数据与应用贡献 |
| 否证结果 | SACR、SCHE、SDIS+CHCR 及其他 No-Go | 证明方法边界，不进入最终模型 |

因此当前尚未形成适合按“三个联合模型创新”表述的最终方法。

## 4. 首选研究主线

从现有证据强度看，首选路线是把论文研究问题收窄为：

> 如何利用药材—成分和靶点—疾病侧信息，提高中医药成分—靶点预测中的
> side-information-assisted compound cold-start 泛化能力？

该路线只使用一套主协议和一套最终配置：

```text
H-C / P-D 双超图编码
+ HPLGA 线性全局注意力候选
+ 候选级 Hctx-P
+ SDIS
+ Dot 解码器
```

CHCR 不进入最终主配置，只在补充材料中报告为普通随机边条件下的辅助实验。
普通随机边结果继续用于与原 HDCTI 和外部基线比较，但不再决定最终模块。

如果论文必须保持“通用随机边 CTI”定位，则应反向处理：SDIS 只能作为额外
cold-start 实验，最终模型不能把它列为核心通用创新。两条定位不能同时作为主线。

## 5. 支持掩码候选审计结果

原定“共享上下文 head 的支持掩码 episodic training”已完成近邻工作审计，
判定为**创新性 No-Go**。DropoutNet 已通过训练期协同输入 dropout 显式模拟
cold-start；PT-GNN 已从 warm 实体构造 cold episode；CGRC 更直接地掩蔽随机
物品的全部交互边并重建关系。CLCRec 与 ALDI 还分别覆盖了内容—协同对比和
warm-to-cold 蒸馏。

因此：

1. 不实现该候选，不进入单折 Pilot；
2. 不搜索 pseudo-cold ratio、mask seed、蒸馏或对比权重；
3. 现有 SCHE masking 代码只保留为失败实验基础设施；
4. `Hctx-P + SDIS` 仍是唯一 cold-start 最终模型候选；
5. 若仍增加第三个模型机制，必须先证明其不等价于 interaction dropout、
   masked graph reconstruction、warm-to-cold distillation、内容—协同对比或
   warm/cold 双专家。

完整证据见
[支持掩码 Episodic 训练近邻工作审计](SUPPORT_MASKED_EPISODIC_AUDIT.md)。

## 6. 投稿状态

在投稿定位冻结前：

* 不绘制最终 `Ours-full` 方法图；
* 不撰写“三项模型创新已经闭环”；
* 不把 CHCR 与 SDIS 分别放进两个主场景后合称统一框架；
* 可以继续整理数据、基线、案例和复现材料，但正式 Methods 只能写已冻结事实；
* 当前支持掩码候选已在创新性门槛停止，不能通过更名重新开放；
* 当前可行的统一模型只有 `Hctx-P + SDIS`，第三项贡献优先由 Strict 四库
  cold-start benchmark 与 ETCM2.0 数据/证据闭环承担；
* 第三项模型候选现限定为 HPLGA，它补回被删除的全局依赖能力，不再搜索同族门控；
* HPLGA 只有通过无二次张量测试、四库 validation-only Gate 和统一 cold-start
  Gate 后，才能进入最终 `Ours-full`。

HPLGA 的公式、近邻工作边界和预注册停止条件见
[HYPERGRAPH_PAGERANK_LINEAR_ATTENTION.md](HYPERGRAPH_PAGERANK_LINEAR_ATTENTION.md)。
