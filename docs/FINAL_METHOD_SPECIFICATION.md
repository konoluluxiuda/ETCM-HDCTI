# 最终方法冻结规格

> 状态：2026-08-05 冻结。本文档是最终模型、公式、配置和论文表述的唯一权威入口。
> 早期场景化方案与失败候选仍保留在其他文档中，但不得覆盖本文档的最终口径。

> **2026-08-06 复审**：以下结构和 checkpoint 继续冻结，不回看 outer-test
> 调参；但它在四库全候选首折的 MRR/Recall@20 均低于简单折内启发式。因此
> “最终”目前仅表示已有 sampled-pair 模型的冻结规格，不表示候选靶点检索方法
> 已通过投稿 Gate。后续决策以 `FULL_CANDIDATE_RANKING_GATE.md` 为准。

## 1. 研究问题

本文的主要任务是中医药成分—靶点预测中的：

```text
side-information-assisted compound cold-start
```

测试 compound 的全部 C-P 关系均不进入训练或内层验证，但其 H-C 药材侧信息
仍然可用；测试 protein 可以在训练 C-P 图中出现，P-D 侧信息同样保留。因此该
任务回答的是“没有已知靶点、但具有药材来源信息的新成分如何预测候选靶点”，
不是没有 H-C、分子结构或其他属性的完全 de novo 预测，也不是 target-cold 或
double-cold。

## 2. 最终模型

最终模型使用一套固定训练和推理配置：

```text
Strict 逐折无泄漏数据协议
        ↓
H-C / P-D 双超图编码器
        ↓
候选级药材上下文—靶点交互 Hctx-P
        ↓
支持度解耦归纳评分 SDIS
        ↓
支持度校准药材原型迁移 SCHPT
        ↓
Sigmoid 输出 C-P 关联概率
```

模型保留原 HDCTI 的 H-C/P-D 双超图与蛋白侧 P-D PageRank，关闭全节点稠密
自注意力，并以 SCHPT 替换成分侧 C-P PageRank。基础 pair decoder 固定为 Dot。

## 3. 共享表示

令 $z_c,z_p\in\mathbb R^d$ 分别为双超图编码器生成的 compound 和 protein
表示。候选 compound 的药材上下文由其关联 H-C 超边表示聚合并 L2 归一化：

$$
h_c=\operatorname{Norm}\left(\sum_{h\in\mathcal H(c)}e_h\right).
$$

这里 $e_h$ 是药材超边表示，$\mathcal H(c)$ 是 compound $c$ 所属药材集合。
该上下文只依赖 H-C，不读取 H-D 或测试 C-P 标签。

## 4. 三个算法贡献

### 4.1 Hctx-P：候选级药材上下文交互

原模型对两个超图独立编码后主要依赖 compound/protein 表示的点积，候选
compound 的药材归属语义没有在 pair 级显式保留。Hctx-P 增加：

$$
s_{HP}(c,p)=(h_c\odot w_{HP})^Tz_p,
$$

其中 $w_{HP}\in\mathbb R^d$ 为可训练的因子化交互权重。它把“该成分来自哪些
药材”直接与候选靶点表示匹配，同时避免构造全节点 $N\times N$ 注意力矩阵。

### 4.2 SDIS：支持度解耦归纳评分

基础 ID 分支为：

$$
s_{ID}(c,p)=z_c^Tz_p.
$$

在 compound cold-start 中，测试 compound 的 $z_c$ 未获得 C-P 监督，该分支
可能产生不可迁移的高置信分数。SDIS 只根据当前训练折定义门控：

$$
g_c=
\begin{cases}
0,&\deg_{CP}^{train}(c)=0\ \text{且}\ H\text{-}C\text{上下文可用},\\
1,&\text{其他情况}.
\end{cases}
$$

SDIS 不增加可训练参数，也不根据数据库名称或测试结果切换。其基础评分为：

$$
s_{SD}(c,p)=g_c s_{ID}(c,p)+s_{HP}(c,p).
$$

### 4.3 SCHPT：支持度校准药材原型迁移

原 compound C-P PageRank 是候选无关的节点标量，且在 cold-start compound 上
缺少直接 C-P 支撑。SCHPT 使用当前训练折 C-P 正边，为每个药材构建靶点原型。

令 $S_h$ 为药材 $h$ 下具有训练 C-P 支撑的 compound 集合，$n_{h,p}$ 为其中与
靶点 $p$ 相连的 compound 数，$m_h=|S_h|$。训练折靶点先验为：

$$
\pi_p=\frac{\sum_c\mathbb I[(c,p)\in E_{CP}^{train}]}
{|\{c:\deg_{CP}^{train}(c)>0\}|}.
$$

评价 $(c,p)$ 时执行 leave-one-compound-out，删除 $c$ 的支撑状态和其
$(c,p)$ 训练边贡献：

$$
q_{h,p}^{(-c)}=
\frac{n_{h,p}^{(-c)}+\kappa\pi_p}
{m_h^{(-c)}+\kappa},\qquad \kappa=1.
$$

对候选 compound 的有效药材取平均残差：

$$
r_{SCHPT}(c,p)=
\frac{1}{|\mathcal H_{valid}(c)|}
\sum_{h\in\mathcal H_{valid}(c)}
\left(q_{h,p}^{(-c)}-\pi_p\right).
$$

若没有其他受支持药材成员，残差严格回退为 $0$。最终评分为：

$$
s(c,p)=g_c z_c^Tz_p+s_{HP}(c,p)+\alpha r_{SCHPT}(c,p),
$$

其中 $\alpha$ 是唯一新增标量参数，从 $0$ 初始化。最终概率为：

$$
\hat y_{cp}=\sigma(s(c,p)).
$$

SCHPT 启用时删除 compound 侧 C-P PageRank，保留 protein 侧 P-D PageRank。

## 5. 三个贡献为何能够联合

三个模块位于同一模型、同一 checkpoint 和同一 cold-start 协议中，分别处理
不同失败环节：

| 模块 | 作用层次 | 解决的问题 | 是否依赖测试标签 |
|---|---|---|---|
| Hctx-P | pair 级表示交互 | 双超图独立编码导致药材语义进入预测过晚 | 否 |
| SDIS | 基础分可靠性控制 | 零训练支持 compound 的 ID 点积分支不可迁移 | 否 |
| SCHPT | 折内监督迁移 | compound PageRank 候选无关且 cold-start 支撑不足 | 否，只使用训练折 C-P |

它们不是针对三个数据库分别挑选的配置，也不是互斥的双场景方案。CHCR 仅保留为
普通随机边协议的补充训练正则，不进入最终 `Ours-full`。

## 6. 训练目标与模型选择

最终模型使用二元交叉熵与既有正则项：

$$
\mathcal L=\mathcal L_{BCE}+\lambda\mathcal L_{reg}.
$$

每个 outer fold 内再划分 inner-validation，按 AUPR 早停并恢复最佳 checkpoint。
outer test 只在模型和 checkpoint 冻结后评价，不参与超参数或模块选择。

## 7. 冻结配置

最终确认实验统一使用：

| 配置项 | 冻结值 |
|---|---|
| Split | compound cold-start, 5 folds |
| Seed | `52026` |
| Inner-validation metric | AUPR |
| Embedding dimension | 64 |
| Pair decoder | Dot |
| Hctx-P | enabled, static |
| SDIS | enabled |
| SCHPT | enabled, prior `1.0` |
| Compound C-P PageRank | disabled/replaced |
| Protein P-D PageRank | enabled |
| Dense full self-attention | disabled, `attention.max.nodes=0` |
| H-D | disabled |

配置清单及哈希位于 `configs/schpt_full_manifest.json`，统一入口为：

```bash
./run_schpt_full.sh
```

实现冻结于提交 `afda595`；后续文档提交不得改变模型公式、seed、prior 或 Gate。

## 8. 递进消融与证据

最终 cold-start 消融链固定为：

```text
Strict-HDCTI / NoContext
→ + Hctx-P
→ + SDIS
→ + SCHPT（Ours-full）
```

| 贡献 | 四库主要证据 | 证据等级 |
|---|---|---|
| Hctx-P | 相对 NoContext macro AUPR `+0.437826`，20/20 folds 提高 | A |
| SDIS | 相对 Hctx-P macro AUPR `+0.028024`，20/20 folds 提高 | A |
| SCHPT | 相对 Hctx-P+SDIS macro AUPR `+0.015931`，4/4 数据库、17/20 folds 提高 | A |

SCHPT 的分库 AUPR 增量为 TCM-Suite `+0.003758`、TCMSP `+0.017766`、
SymMap2.0 `+0.030323`、ETCM2.0-mention10 `+0.011878`。

## 9. 论文主张边界

允许主张：

1. 三个模块在同一 compound cold-start 最终配置中形成递进改进；
2. H-C 侧信息可为无训练 C-P 支撑 compound 提供可迁移上下文；
3. SCHPT 在四库整体上优于匹配的 Hctx-P+SDIS 基线；
4. Strict 协议保证所有 C-P 派生统计仅来自当前训练折。

禁止主张：

1. 解决无 H-C 或无任何属性的完全 de novo compound；
2. 解决 target-cold、double-cold 或 disease-aware 泛化；
3. SCHPT 在每个 fold 均提高，TCM-Suite 实际仅 2/5 folds 为正；
4. CHCR 是最终模型的第四模块，或 `Hctx-P+CHCR+SDIS` 已联合通过；
5. 使用了分子结构、蛋白序列或统一多模态输入；
6. fold 标准差等同于多随机初始化稳定性。

## 10. 论文材料映射

| 论文部分 | 权威来源 |
|---|---|
| 方法定义与公式 | 本文档 |
| SCHPT 预注册与完整结果 | `SCHPT_PILOT_PROTOCOL.md` |
| 证据等级与边界 | `METHOD_EVIDENCE_MATRIX.md` |
| 主结果和消融数值 | `FINAL_RESULTS_TABLES.md` |
| 数据统计 | `DATASET_STATISTICS.md` |
| ETCM 案例解释 | `ETCM_REPRESENTATIVE_CASES.md` |
| 历史失败分支 | `修改计划.md` 与各 No-Go 专题文档 |

下一阶段不再新增模型模块，工作转为：更新最终结果表中的 SCHPT 递进消融、绘制
最终方法图、撰写 Methods/Experiments，并对照本规格执行投稿前一致性审计。
