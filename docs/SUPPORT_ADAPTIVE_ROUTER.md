# 结构支持度自适应上下文路由

## 1. 研究动机

统一的一折 compound cold-start 审计中，Hctx-P 相对 NoContext 的 AUPR 增益为：

```text
TCM-Suite  +0.167222
TCMSP      +0.599694
SymMap2.0  +0.388923
ETCM2.0    +0.533880
Macro mean +0.422430
```

四个数据集均同向提高，说明药材上下文是未见 compound 的稳定信息来源。但在随机边划分中，已有 C-P 支撑的 compound 仍可从协同结构中获益，不应对所有节点固定使用相同强度的上下文残差。

因此新增“结构支持度自适应上下文路由”（Support-Adaptive Context Routing, SACR），在协同结构专家与 Hctx-P 上下文增强专家之间进行可解释的单调路由，并通过整 compound 伪冷启动图掩码训练 degree=0 场景。

## 2. 模型定义

基础协同结构分数为：

$$
b_{cp}=z_c^Tz_p
$$

Hctx-P 残差为：

$$
r_{cp}=(h_c\odot w_{HP})^Tz_p
$$

其中 $h_c$ 只由候选 compound 的 H-C 药材上下文聚合得到。令 $d_c$ 为当前 fold 训练 C-P 图中 compound $c$ 的正边度数，$a_c\in\{0,1\}$ 表示是否具有 H-C 上下文。路由权重定义为：

$$
g_c=a_c(1+d_c)^{-\operatorname{softplus}(\theta)}
$$

最终分数为：

$$
s_{cp}=b_{cp}+g_cr_{cp}
$$

该形式满足：

* $d_c=0$ 且 H-C 可用时，$g_c=1$，完整启用 Hctx-P；
* $g_c$ 随 C-P 支持度严格单调不增；
* H-C 缺失时 $g_c=0$，回退到基础专家；
* 只学习一个全局正斜率，不使用数据集特定阈值或黑箱 MLP 门控。

## 3. 伪冷启动训练

每个 inner-training fold 中，使用固定 seed 从具有 H-C 支撑的训练正例 compound 中确定性选择 10%。对每个被选 compound：

1. 从 PageRank 使用的训练 C-P 图中移除其全部 C-P 正边；
2. 保留原始正负监督 pair，用于 BCE 训练；
3. 将其图支持度设为 0，使上下文 gate 严格等于 1；
4. 不修改 H-C、P-D、外层测试集或 inner-validation。

这不是随机逐边 DropEdge，而是整 compound 的标签图隔离。选择结果记录 compound 数、移除边数、seed 与 assignment SHA-256，并随 checkpoint 写入 `support_router.json`。

## 4. 无泄漏边界

路由器只能读取：

```text
当前 inner-training C-P 正边度数
当前数据集固定 H-C 关系及其可用性
```

禁止读取：

```text
outer-test C-P 边
inner-validation C-P 边
完整数据集 C-P degree
由测试边计算的 PageRank、相似度或统计特征
```

伪冷启动 compound 的监督标签仍参与 BCE，但其 C-P 边不得进入 PageRank 图。这一设置模拟“可获得训练标签用于学习上下文专家，但节点协同图支持被整体隐藏”的训练 episode。

## 5. Pilot 冻结配置

```text
support.router=True
support.router.mode=monotonic_residual
support.router.pseudo.cold.ratio=0.1
support.router.seed=62026
support.router.initial.slope=1.0
counterfactual.context=False
attention.max.nodes=2000
```

CHCR 在本轮关闭，避免把第二项创新的增益混入 SACR 筛选。其余 split、训练 seed、早停、Dot decoder、维度和 batch size 与 Hctx-P 一折对照完全一致。

## 6. 一折运行命令

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_cold_start_support_router_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_cold_start_support_router_pilot.conf
./run_hdcti.sh configs/HDCTI_symmap_cold_start_support_router_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_support_router_pilot.conf
```

## 7. Go/No-Go 条件

Cold-start Router Pilot 进入下一阶段必须同时满足：

1. Router 相对 Hctx-P 的 macro AUPR 增量不小于 0；
2. 至少 3/4 数据集的 AUPR 不低于 Hctx-P；
3. 任一数据集 AUPR 下降不得超过 0.005；
4. 学习后的 slope 有限且大于 0；
5. degree=0 且 H-C 可用的 gate 均为 1，seen compound 的平均 gate 位于 $(0,1)$。

若 cold-start 条件通过，再在 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0 的标准 Strict 随机折上做一折 degree 分层审计。随机折要求总体 AUPR 相对 Hctx-P 不劣于 0.001，并验证 gate 随训练 C-P degree 单调下降。只有 cold-start 与随机折两类证据都通过，才进入完整五折；否则停止 SACR，不搜索其他伪冷比例、seed、斜率初值或门控网络。

## 8. 四库一折结果

所有结果使用与冻结 Hctx-P 对照相同的 Strict compound cold-start fold 1、训练 seed、早停、Dot decoder、注意力上限和外层测试协议。CHCR 保持关闭。

| 数据集 | Hctx-P AUPR | SACR AUPR | AUPR 增量 | learned slope | seen gate mean |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.626401 | 0.639716 | +0.013315 | 0.074198 | 0.847821 |
| TCMSP | 0.923662 | 0.920105 | -0.003557 | 0.064646 | 0.887766 |
| SymMap2.0 | 0.785736 | 0.739882 | -0.045854 | 0.804874 | 0.309981 |
| ETCM2.0 | 0.856796 | 0.886944 | +0.030148 | 0.028661 | 0.941477 |
| Macro mean | 0.798149 | 0.796662 | **-0.001487** | - | - |

四个 checkpoint 的 zero-support gate 均严格等于 `1.0`，learned slope 均为有限正数，因此实现和约束本身工作正常。但 SACR 只有 2/4 数据集不低于 Hctx-P，且 SymMap2.0 的 AUPR 下降 `0.045854`，超过预注册的最大允许下降 `0.005`；macro AUPR 也下降 `0.001487`。

## 9. 判定

SACR **未通过 Go/No-Go 条件，路线终止**：

* TCM-Suite 和 ETCM2.0 的提升说明按支持度衰减上下文在部分数据库可能有效；
* TCM-Suite、TCMSP 和 ETCM2.0 的 slope 接近 0、seen gate mean 接近 1，说明模型大体退回静态 Hctx-P；
* SymMap2.0 学到较强衰减，却出现明显排序退化，说明训练 C-P degree 不是跨数据库稳定的上下文可信度代理；
* 不继续搜索伪冷比例、seed、斜率初值、分段阈值或 MLP 门控，也不进入标准随机折与完整五折。

实现代码和配置保留为可追溯负结果。默认 `support.router=False`，不会影响冻结的 Hctx-P/CHCR 主线或历史 checkpoint。
