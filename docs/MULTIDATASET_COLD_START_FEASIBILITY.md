# 多数据集 Compound Cold-Start 可行性审计

## 1. 审计目的

本审计用于判断下一项模型创新能否同时适用于 TCM-Suite、TCMSP、SymMap2.0 和 ETCM2.0，而不是只针对 ETCM2.0。候选研究问题限定为：

> 在训练 C-P 支持度不同、尤其是 compound 完全未见的条件下，如何在协同结构分数与药材上下文分数之间自适应选择，并保持可信的反事实上下文约束？

审计只读取 H-C、C-P、P-D 和现有未观测样本文件，不访问网络、不生成 split、不训练模型。运行命令：

```bash
python tools/audit_multidataset_cold_start.py
```

机器可读结果位于：

```text
results/multidataset_cold_start_feasibility/report.json
results/multidataset_cold_start_feasibility/report.md
```

## 2. 预注册门槛

统一 compound cold-start Pilot 需要满足：

* C-P compound 的 H-C 支撑率不低于 95%；
* 至少 500 个 C-P compound；
* 贪心平衡后每个测试 fold 至少包含 1,000 条 C-P 正边；
* 每个 compound 在全 protein 候选空间中都能构造 1:1 未观测 pair。

统一 CHCR 还需要满足：

* 至少 90% 的 C-P compound 存在 H-C 度数完全相同且药材集合不相交的供体；
* 这些可匹配 compound 覆盖至少 90% 的 C-P 正边。

固定 ZERO 文件覆盖不足不是否决条件。Strict split 可以从同一 protein 全集确定性补采，但必须排除全部已知 C-P 正边，并将 seed、候选范围和生成结果写入 manifest。

## 3. 审计结果

| 数据集 | C-P compound | C-P 正边 | H-C 支撑 | CHCR compound 覆盖 | CHCR 正边覆盖 | P-D protein 支撑 | 每折最少正边 | Cold-start | CHCR |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| TCM-Suite | 1,187 | 43,669 | 100.00% | 96.46% | 94.47% | 28.18% | 8,733 | 通过 | 统一可用 |
| TCMSP | 6,929 | 56,102 | 99.68% | 97.60% | 96.01% | 20.82% | 11,220 | 通过 | 统一可用 |
| SymMap2.0 | 1,618 | 37,991 | 98.64% | 81.58% | 33.40% | 92.70% | 7,598 | 通过 | 仅选择性可用 |
| ETCM2.0_core_mention10 | 9,519 | 88,431 | 100.00% | 99.26% | 99.48% | 100.00% | 17,686 | 通过 | 统一可用 |

总体判定：

```text
supports_multidataset_compound_cold_start_with_selective_CHCR
```

四个数据集都具备统一 compound cold-start 实验的结构条件。CHCR 在 TCM-Suite、TCMSP 和 ETCM2.0 上覆盖充分，但 SymMap2.0 只有 81.58% compound、33.40% C-P 正边具备合格供体。因此：

1. compound cold-start 可以作为四库共同主协议；
2. CHCR 必须使用 eligibility mask，只作用于有可信供体的训练样本；
3. 不能把 CHCR 写成对所有数据集、所有节点都同等适用的统一模块；
4. SymMap2.0 必须单独报告 CHCR 有效样本覆盖率和 eligible/ineligible 分层结果。

## 4. 对统一研究方向的约束

当前不采用 ETCM2.0 专用的 SMILES/蛋白序列分支作为共享主创新。四库共同拥有且覆盖充分的是 H-C/C-P/P-D 结构，因此第三项候选创新调整为“结构支持度自适应双专家冷启动框架”：

```text
协同结构专家：Strict-HDCTI 的 compound-protein 基础分数
药材上下文专家：候选级 Hctx-P 分数
支持度路由：根据当前 fold 训练 C-P degree、H-C degree 和上下文可用性分配专家权重
伪冷启动训练：按 compound 整体隐藏训练 C-P 边，显式训练低支持度场景
选择性 CHCR：仅对存在合格供体的 compound 施加反事实 margin 约束
```

路由特征只能从当前 fold 的训练数据计算。测试 C-P 边、完整 C-P degree 和完整图 PageRank 均不得进入路由器。为保证解释性，协同专家权重应随训练 C-P degree 单调不减，避免黑箱门控利用数据集偏差。

## 5. 为什么暂不统一做 Target Cold-Start

TCM-Suite 和 TCMSP 中，C-P protein 获得 P-D 支撑的比例分别只有 28.18% 和 20.82%。这意味着许多未见 target 缺少模型所需的疾病侧上下文。若直接在四库统一报告 target cold-start，会把“模型能力”与“侧信息缺失”混合在一起。

因此当前主协议只采用 compound cold-start。Target cold-start 可作为补充实验，但必须限制到 P-D-supported target 子集并明确报告覆盖率，不能与四库主结果直接混合。

## 6. 下一实验闸门

先使用完全相同的一折 compound cold-start 协议，在四个数据集上比较：

```text
NoContext
Hctx-P
```

只有当 Hctx-P 的 outer-test AUPR 在至少 3/4 个数据集上提高，且没有数据集出现超过 0.005 的 AUPR 下降，才实现支持度路由与伪冷启动训练。CHCR 不参与这一轮专家互补筛选，以免把上下文专家和正则增益混为一谈。

对应配置与运行命令为：

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_cold_start_no_context_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsuite_cold_start_herb_only_pilot.conf

./run_hdcti.sh configs/HDCTI_tcmsp_cold_start_no_context_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_cold_start_herb_only_pilot.conf

./run_hdcti.sh configs/HDCTI_symmap_cold_start_no_context_pilot.conf
./run_hdcti.sh configs/HDCTI_symmap_cold_start_herb_only_pilot.conf

./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_no_context_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_herb_only_pilot.conf
```

同一数据集的两项实验必须复用同一个 split manifest。四库配置统一固定 `attention.max.nodes=2000`，避免 SymMap2.0 的全节点稠密注意力导致 OOM；这一设置必须随结果报告，不得按数据集单独修改。由于先前 ETCM cold-start 使用的是不同 attention 口径，本轮也要重跑 ETCM 一折，不直接混用历史结果。建议按数据集成对运行并立即记录结果；不需要在这一筛选阶段增加 seed 或扩展为五折。

| 数据集 | NoContext AUPR | Hctx-P AUPR | Delta | 是否通过 |
|---|---:|---:|---:|---|
| TCM-Suite | 0.459180 | 0.626401 | +0.167222 | 通过 |
| TCMSP | 0.323968 | 0.923662 | +0.599694 | 通过 |
| SymMap2.0 | 0.396813 | 0.785736 | +0.388923 | 通过 |
| ETCM2.0_core_mention10 | 0.322916 | 0.856796 | +0.533880 | 通过 |

### 6.1 TCM-Suite 一折结果

TCM-Suite 已按统一配置完成 fold 1：

| 方法 | Best epoch | Validation AUPR | AUC | AUPR | Recall | Precision | F1 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoContext | 10 | 0.489302 | 0.449113 | 0.459180 | 0.002862 | 0.403226 | 0.005684 | 21.10 s |
| Hctx-P | 30 | 0.604804 | 0.630067 | 0.626401 | 0.567094 | 0.601676 | 0.583874 | 25.72 s |
| Delta | - | +0.115502 | +0.180954 | +0.167222 | +0.564232 | +0.198451 | +0.578189 | +4.62 s |

该数据集满足预注册 AUPR 闸门。NoContext 的 AUC 低于 0.5 且 0.5 阈值下 Recall 接近 0，说明无上下文模型在该 cold-start fold 上几乎失效；Hctx-P 同时改善排名与固定阈值分类。但这仍是单折筛选结果，不报告方差，也不用于调整下一数据集的超参数。

### 6.2 TCMSP 一折结果

| 方法 | Best epoch | Validation AUPR | AUC | AUPR | Recall | Precision | F1 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoContext | 4 | 0.324864 | 0.078616 | 0.323968 | 0.012833 | 0.250000 | 0.024413 | 20.57 s |
| Hctx-P | 48 | 0.948989 | 0.920794 | 0.923662 | 0.374833 | 0.966452 | 0.540166 | 40.96 s |
| Delta | - | +0.624125 | +0.842178 | +0.599694 | +0.362000 | +0.716452 | +0.515753 | +20.38 s |

TCMSP 同样满足 AUPR 闸门。NoContext 在完全未见 compound 上出现明显的排序反向（AUC `0.078616`），Hctx-P 恢复了有效排序。Hctx-P 的最佳 epoch 为 48，epoch 50 时 validation AUPR 已从 `0.948989` 降至 `0.947697`，因此本 Pilot 不延长 epoch 上限。

### 6.3 SymMap2.0 一折结果

| 方法 | Best epoch | Validation AUPR | AUC | AUPR | Recall | Precision | F1 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoContext | 10 | 0.406273 | 0.311577 | 0.396813 | 0.001184 | 0.529412 | 0.002363 | 30.45 s |
| Hctx-P | 18 | 0.806072 | 0.769000 | 0.785736 | 0.723648 | 0.710832 | 0.717183 | 27.05 s |
| Delta | - | +0.399799 | +0.457423 | +0.388923 | +0.722463 | +0.181421 | +0.714819 | -3.40 s |

SymMap2.0 满足 AUPR 闸门，且 Hctx-P 没有引入额外时间负担。这一结果只验证静态药材上下文专家；由于 SymMap2.0 的 CHCR 供体仅覆盖 `33.40%` C-P 正边，不得将本结果延伸为 CHCR 在 SymMap2.0 上同样有效。

### 6.4 ETCM2.0_core_mention10 一折结果

| 方法 | Best epoch | Validation AUPR | AUC | AUPR | Recall | Precision | F1 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoContext | 2 | 0.323198 | 0.111171 | 0.322916 | 0.057500 | 0.090974 | 0.070464 | 53.81 s |
| Hctx-P | 12 | 0.861716 | 0.874765 | 0.856796 | 0.482897 | 0.894720 | 0.627254 | 75.16 s |
| Delta | - | +0.538518 | +0.763594 | +0.533880 | +0.425397 | +0.803746 | +0.556790 | +21.35 s |

ETCM2.0 在与其他三库一致的 `attention.max.nodes=2000` 口径下同样通过。因此不再混用先前 full-attention cold-start 结果作为本轮证据。

### 6.5 四库闸门结论

Hctx-P 相对 NoContext 的一折 AUPR 增益为：

```text
TCM-Suite  +0.167222
TCMSP      +0.599694
SymMap2.0  +0.388923
ETCM2.0    +0.533880
Macro mean +0.422430
```

四个数据集均提高，没有出现超过 `0.005` 的退化，因此通过预注册闸门。这一结论支持进入“结构支持度自适应双专家 + 伪冷启动训练”实现阶段，但四组结果仍是单折方法筛选，不代替最终五折和多 seed 统计。

后续 SACR 四库一折 Pilot 已完成。其相对 Hctx-P 的 AUPR 增量为 `+0.013315/-0.003557/-0.045854/+0.030148`，macro 增量为 `-0.001487`。由于仅 2/4 数据集不下降，且 SymMap2.0 的退化超过 `0.005` 上限，SACR 未通过二次 Go/No-Go，不进入标准随机折或完整五折，也不在 outer-test 上搜索路由网络层数、阈值或损失权重。完整结果和判定见 [SUPPORT_ADAPTIVE_ROUTER.md](SUPPORT_ADAPTIVE_ROUTER.md)。
