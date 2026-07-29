# 支持度条件化双专家药材上下文学习

## 1. 研究动机

当前冻结证据支持三个机制：

1. Hctx-P 能够显式保留候选成分的药材上下文；
2. CHCR 能够在普通随机边协议中约束事实药材上下文优于度数匹配反事实；
3. SDIS 能够在 compound cold-start 中关闭没有训练 C-P 支持的 compound-ID 基础分。

但三者目前没有形成一个可统一启用的最终模型。冻结的 `SDIS+CHCR`
组合在 TCM-Suite compound cold-start 上 AUPR 下降 `0.019451`，说明共享
Hctx-P 参数同时承担已观测实体判别和零支持归纳预测时存在负迁移。若继续以
两个协议分别选择 CHCR 或 SDIS，容易被质疑为事后选择配置。

因此只重新开放一次结构性修正：支持度条件化双专家药材上下文学习
（Support-Conditioned Herb Experts，SCHE）。本轮不是继续搜索 CHCR 权重、
margin、donor 或 SDIS 阈值，而是将 warm/cold 上下文参数显式解耦，并在同一个
checkpoint 中按训练支持状态逐样本路由。

## 2. 统一模型

基础协同分数为：

$$
b_{cp}=z_c^Tz_p
$$

warm 药材上下文专家为：

$$
r_{cp}^{warm}=(h_c\odot w_{warm})^Tz_p
$$

cold 药材上下文专家为：

$$
r_{cp}^{cold}=(h_c\odot w_{cold})^Tz_p
$$

其中 $h_c$ 由 H-C 超图聚合得到，$z_p$ 由 P-D 侧编码器得到。令
$d_c^{graph}$ 为当前训练模型实际可用的 C-P 图正边度数，$a_c$ 表示 H-C
上下文是否可用：

$$
g_c^{cold}=
\mathbb I[d_c^{graph}=0\land a_c=1]
$$

最终分数为：

$$
s_{cp}=
(1-g_c^{cold})(b_{cp}+r_{cp}^{warm})
+
g_c^{cold}r_{cp}^{cold}
$$

因此：

* 有训练 C-P 图支持时使用基础协同分和 warm Hctx-P；
* 零训练 C-P 图支持且 H-C 可用时，关闭 compound-ID 基础分并使用独立 cold
  Hctx-P；
* 零支持且 H-C 也不可用时回退到 warm 基础分，但该情形不属于当前
  side-information-assisted cold-start 的可靠适用范围；
* 路由只读取当前训练折图支持状态，不读取数据库名称、评估协议或测试指标。

## 3. 训练方式

每个 inner-training fold 使用固定 seed，从具有 H-C 支持的训练正例 compound
中确定性选择 10% 作为 pseudo-cold compound：

1. 从当前训练 C-P 图中删除这些 compound 的全部正边；
2. 其监督 pair 仍保留在训练批次中；
3. pseudo-cold pair 只通过 cold expert 计算 BCE；
4. cold expert 使用 `stop_gradient(h_c,z_p)`，避免其损失反向改变 warm 编码器；
5. 其他 pair 使用 warm expert；
6. CHCR 使用同一组冻结 donor、margin、weight 和 draw 设置，但根据相同
   support gate 分别更新 warm 或 cold 上下文权重。

总损失为：

$$
L=L_{BCE}^{routed}+\lambda_{cf}L_{CHCR}^{routed}+\lambda_{reg}L_{reg}
$$

本轮固定：

```text
support.experts=True
support.experts.mode=hard_zero_support
support.experts.pseudo.cold.ratio=0.1
support.experts.seed=72026
support.experts.detach.cold.features=True
counterfactual.context=True
counterfactual.weight=0.05
counterfactual.margin=0.2
counterfactual.draws=20
counterfactual.seed=42026
attention.max.nodes=0
pair.decoder=dot
```

不搜索 ratio、seed、loss weight、margin、draw、soft gate、专家数量或数据库特定
参数。

## 4. 与既有失败路线的区别

SCHE 不等于已经失败的 SACR：

| 项目 | SACR | SCHE |
|---|---|---|
| 上下文参数 | warm/cold 共用一个 Hctx-P | 独立 `w_warm` 与 `w_cold` |
| 路由对象 | 按 degree 连续缩放上下文残差 | 在完整 warm score 与 context-only cold score 间硬路由 |
| 零支持基础分 | 仍保留 | 有 H-C 时关闭 |
| 伪冷梯度 | 可改变共享上下文编码 | cold 特征停止梯度，只训练 cold head |
| CHCR | Pilot 中关闭 | 按相同 gate 分专家约束 |

SCHE 也不等于旧 `SDIS+CHCR`：旧组合仍使用同一个
`context_herb_protein` 权重，SCHE 为 cold 路径提供独立参数，目标是消除
warm/cold 负迁移，而不是修改已有 CHCR 或 SDIS 超参数。

## 5. Pilot 顺序与门槛

第一步只运行 TCM-Suite compound cold-start fold 1 inner-validation，因为该库
是旧组合退化最明显的失败点。outer test 保持关闭。

进入四库双协议 Pilot 的条件：

```text
SCHE validation AUPR >= 0.669984
```

`0.669984` 是同一 TCM-Suite cold-start fold 1 的冻结 SDIS validation AUPR。
若未达到，SCHE 立即终止，不修改结构或超参数。

若第一步通过，再执行四库 fold 1：

```text
普通 Strict 随机边：相对 Hctx-P+CHCR macro AUPR >= -0.001
Compound cold-start：相对 Hctx-P+SDIS macro AUPR >= 0
两种协议分别至少 3/4 数据库不下降
任一数据库下降不得超过 0.005
```

只有两种协议同时通过，才允许在新的冻结 split seed 上运行完整五折。新 split
产生前必须记录 manifest 和配置 SHA-256；结构冻结后才能读取 outer-test。

## 6. 论文决策

若完整验证通过，最终三项方法贡献统一为：

1. 候选级药材上下文交互 Hctx-P；
2. 度数匹配反事实上下文可靠性学习 CHCR；
3. 支持度条件化 warm/cold 双专家归纳评分 SCHE。

三者位于同一个模型、同一个 checkpoint 和同一配置中，不再按评估协议切换模块。

若 TCM-Suite 第一阶段或后续四库门槛失败：

* SCHE 判定 No-Go；
* SDIS 保留为 cold-start 机制证据；
* CHCR 降级为辅助训练正则，不再作为独立主创新；
* 不继续寻找第四种门控、注意力或数据集特定组合。

## 7. Pilot 结果与冻结结论

2026-07-29 按上述冻结配置完成 TCM-Suite compound cold-start fold 1
inner-validation。运行时未读取 outer test：

```text
配置：configs/HDCTI_tcmsuite_cold_start_sche_pilot.conf
配置 SHA-256：aa4fc8196e302959560247721071df9e002d48ffe3b3fbed19f050c9e3583d36
正式复核设备：GPU
伪冷成分：86 / 856
从训练 C-P 图移除的正边：2,160
最佳 epoch：6
停止 epoch：16
SCHE validation AUPR：0.650499
冻结 SDIS validation AUPR：0.669984
差值：-0.019485
运行时间：15.924 s
checkpoint：saved_model/2026-07-29 16-27-59/hdcti_model.ckpt
```

同配置的 CPU 首跑得到 AUPR `0.650017`，与 GPU 正式复核仅相差
`0.000482`，且最佳 epoch 和停止 epoch 完全一致。因此失败结论不能归因于
CPU/GPU 执行设备差异。

结果未达到预注册门槛，SCHE 判定为 **No-Go**，不进入四库双协议 Pilot，不搜索
pseudo-cold ratio、seed、margin、CHCR 权重、soft gate 或专家结构。

该结果说明“将训练样本拆成 warm/cold 两个独立线性上下文 head”没有解决当前
冲突。一个合理但尚未被单独证明的原因是：cold head 只从 10% pseudo-cold
成分获得监督，而且 context 特征停止梯度；它失去了 SDIS 中共享 Hctx-P 从全部
warm 成分学习上下文参数的优势。继续增加 cold 样本或解除停止梯度都属于新的
超参数/结构搜索，不在本轮预注册范围内。

因此最终口径保持：

1. Hctx-P 为共享结构方法；
2. CHCR 为普通随机边实验中的辅助训练正则和机制审计；
3. SDIS 为 compound cold-start 专用的无参数支持度规则；
4. 不声称三者已经组成一个联合增益的 `Ours-full`。

作者随后进一步否决将 CHCR 与 SDIS 分别放入两个主场景后合称统一框架。当前
首选方向是收窄到 compound cold-start 单一主任务，CHCR 降为补充材料，并在
实现前审计一个使用共享 Hctx-P 与 SDIS 的统一训练候选。详见
[最终方法统一性决策](UNIFIED_METHOD_DIRECTION.md)。
