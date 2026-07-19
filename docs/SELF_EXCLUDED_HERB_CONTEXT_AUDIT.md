# 自排除药材上下文归纳编码审计

## 1. 研究问题

当前冻结主模型的候选分数为：

$$
s(c,p)=z_c^Tz_p+(h_c\odot w_{HP})^Tz_p
$$

在 compound cold-start 中，测试 compound 不具有训练 C-P 边，但基础点积仍使用其 ID 表示。同时，当前药材上下文由候选 compound 的关联药材超边聚合，药材超边的节点到超边均值包含候选 compound 自身。因而现有 Hctx-P 虽然显著改善冷启动排序，却不是严格的自排除归纳表示。

本阶段只回答：

> 在冻结 Hctx-P checkpoint 上，直接删除候选 compound 对药材超边的自身贡献，并屏蔽未训练 compound ID 的基础点积后，是否存在跨数据库稳定的 inner-validation 排名增益？

审计不训练参数、不读取 outer-test，也不把冻结审计结果作为最终模型结果。

## 2. 冻结自排除表示

对第 $l$ 层药材 $h$ 的节点到超边输入 $x_c^{(l)}$，候选 compound $c$ 的直接自排除超边为：

$$
e_{h\setminus c}^{(l)}=
\operatorname{Norm}\left(
\frac{
\sum_{j\in C(h)}x_j^{(l)}-x_c^{(l)}
}{|C(h)|-1}
\right)
$$

跨层、跨药材聚合得到：

$$
\widetilde h_c=
\operatorname{Norm}\left(
\sum_{h\in H(c),|C(h)|>1}
\sum_l e_{h\setminus c}^{(l)}
\right)
$$

冻结候选分数固定为：

$$
s_{SE}(c,p)=(\widetilde h_c\odot w_{HP})^Tz_p
$$

即同时执行：

1. 删除候选 compound 对每层药材超边均值的直接贡献；
2. 对 cold-start compound 屏蔽基础 ID 点积；
3. 复用 checkpoint 已学习的 $w_{HP}$ 和蛋白表示，不拟合缩放系数或投影层。

若药材只包含一个已映射 compound，该药材无法构造自排除上下文并被跳过；某 compound 没有任何合格药材时使用零上下文，不通过删除记录提高覆盖率。

## 3. 协议

使用四个已经训练完成且配置统一为 `attention.max.nodes=2000` 的 compound-cold-start fold 1 HerbOnly checkpoint：

| 数据集 | 配置 | Checkpoint |
|---|---|---|
| TCM-Suite | `HDCTI_tcmsuite_cold_start_herb_only_pilot.conf` | `2026-07-16 22-25-13` |
| TCMSP | `HDCTI_tcmsp_cold_start_herb_only_pilot.conf` | `2026-07-16 22-51-03` |
| SymMap2.0 | `HDCTI_symmap_cold_start_herb_only_pilot.conf` | `2026-07-17 00-13-31` |
| ETCM2.0 | `HDCTI_etcm_mention10_cold_start_herb_only_pilot.conf` | `2026-07-17 10-51-44` |

该阈值会使不同规模的数据集实际保留不同侧的旧稠密注意力，因此本轮只能作为低成本结构可行性审计。若通过，后续训练 Pilot 必须统一为 `attention.max.nodes=0`，不能将本轮数值写入最终主表。

每个数据集只评价 Strict inner-validation，并验证 inner-train 与 validation compound 交集为零。输出以下冻结变体：

```text
base_only
current_total
current_context_only
rebuilt_inclusive_total
self_excluded_total
self_excluded_context_only
```

其中 `rebuilt_inclusive_total` 用于验证审计重建与模型原上下文一致；最终候选固定为 `self_excluded_context_only`。

## 4. 预注册判定

进入一版固定的单折训练 Pilot 必须同时满足：

1. `self_excluded_context_only` 相对 `current_total` 至少在 3/4 数据集 AUPR 不下降；
2. 四库 macro AUPR 增量不低于 `+0.005`；
3. 任一数据集 AUPR 下降不超过 `0.005`；
4. 四库最低 inner-validation 记录覆盖率不低于 `90%`；
5. 药材超边重建最大相对误差不超过 `1e-5`。

未通过时停止该路线，不根据结果搜索基础点积混合系数、数据集特定 gate、上下文缩放或自排除范围。通过后只允许实现一版固定模型，再在统一无稠密注意力协议下运行四库单折 Pilot。

## 5. 命令

```bash
python tools/audit_self_excluded_herb_context.py --dry-run
python tools/audit_self_excluded_herb_context.py
```

默认使用 CPU 完成冻结前向，结果写入：

```text
results/self_excluded_herb_context/frozen_four_dataset_seed2026/
```

## 6. 解释边界

本审计精确删除候选 compound 对当前层节点到药材超边均值的直接贡献，但不为每个候选重新运行整张 H-C 图。因此，同药材其他 compound 在更早传播层中可能仍间接包含该候选的信息。只有审计通过后，才值得实现训练期的真正归纳编码分支。

## 7. 审计结果

四库冻结审计已完成，全部预注册条件通过：

| 数据集 | 当前 total AUPR | 当前 context-only AUPR | Self-excluded context-only AUPR | 候选总增量 | 直接自排除增量 | 覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 0.604804 | 0.642874 | 0.645077 | +0.040273 | +0.002203 | 99.86% |
| TCMSP | 0.948989 | 0.957249 | 0.957909 | +0.008920 | +0.000660 | 99.95% |
| SymMap2.0 | 0.806072 | 0.801573 | 0.802279 | -0.003794 | +0.000706 | 99.97% |
| ETCM2.0 | 0.861716 | 0.890298 | 0.887213 | +0.025497 | -0.003085 | 99.43% |

汇总结果：

```text
候选相对当前模型 macro AUPR:        +0.017724
其中屏蔽基础 ID 点积 macro AUPR:     +0.017603
其中直接 self-exclusion macro AUPR: +0.000121
非下降数据库:                        3/4
最大单库下降:                        -0.003794
最低记录覆盖率:                      99.43%
```

正式判定为：

```text
go_to_single_fold_training_pilot
```

但该 Go 必须按机制拆分解释。复合候选通过主要来自 cold-start 时屏蔽未训练 compound ID 的基础点积，而不是直接自排除本身。直接自排除在 3/4 数据集为正，但 macro 仅 `+0.000121`，ETCM2.0 还下降 `0.003085`。因此：

1. 下一版应定位为**支持度解耦的归纳上下文打分**，核心是 degree=0 时确定性关闭基础 ID 分支；
2. self-excluded context 可以作为固定的归纳上下文构造进入单折训练 Pilot，但在训练结果证明其独立增益前，不能单独列为第三项创新；
3. Pilot 必须增加 `current context-only` 对照，分别验证“基础分支屏蔽”和“自排除构造”的贡献；
4. 后续统一使用 `attention.max.nodes=0`，本轮阈值 2000 的结果不进入最终主表。

机器可读结果位于：

```text
results/self_excluded_herb_context/frozen_four_dataset_seed2026/
```

## 8. 固定训练 Pilot

训练实现命名为**支持度解耦归纳评分**（Support-Decoupled Inductive Scoring, SDIS），包含两个可独立开关的操作：

```text
SD：当 compound 的当前训练折 C-P 正边支持度为 0，且至少存在一个 degree > 1 的关联药材时，关闭基础 ID pair score。
SE：逐层删除候选 compound 对关联药材超边均值的直接贡献，并用该上下文替换原 inclusive H-C context。
```

基础分门控固定为：

$$
g_c^{base}=
\begin{cases}
0,&deg_{CP}^{train}(c)=0\ \land\ q_{HC}(c)=1\\
1,&\text{otherwise}
\end{cases}
$$

其中 $q_{HC}(c)=1$ 表示至少有一个关联药材同时包含其他 compound。若不存在可用的归纳上下文，则保留基础分作为确定性回退，不删除记录、不拟合阈值或数据集特定 gate。

实现默认关闭，旧配置和 checkpoint 行为保持不变：

```ini
inductive.context=False
```

Pilot 使用以下两个训练变体：

```ini
# SD-only
inductive.context=True
inductive.context.suppress.base.zero.support=True
inductive.context.self.excluded=False

# SD+SE
inductive.context=True
inductive.context.suppress.base.zero.support=True
inductive.context.self.excluded=True
```

为隔离机制，Pilot 强制采用：

```text
Strict compound cold-start
HerbOnly static Hctx-P
Dot decoder
attention.max.nodes=0
CHCR / CMIT / support router / FBHA / HILGA 全部关闭
evaluation.fold.limit=1
evaluation.outer.test=False
```

四库配置位于 `configs/`，批处理命令为：

```bash
./run_sdis_pilot_batch.sh --dry-run
./run_sdis_pilot_batch.sh
```

现有匹配基线为四个 `*_cold_start_no_dense_herb_only_pilot.conf`。TCM-Suite 和 TCMSP 已记录的基线 validation AUPR 分别为 `0.606038` 和 `0.950699`；SymMap2.0 与 ETCM2.0 mention10 若尚无匹配结果，只补跑对应基线，不重复已完成实验。

### 8.1 预注册判定

SDIS 进入完整五折的主门槛：

1. `SD-only` 或 `SD+SE` 相对匹配 HerbOnly 基线至少 3/4 数据集 AUPR 不下降；
2. 四库 macro AUPR 增量至少 `+0.005`；
3. 任一数据集 AUPR 下降不超过 `0.005`；
4. 只能依据 inner-validation 选择变体，不读取 outer-test。

Self-exclusion 获得独立模型贡献资格还必须满足：

1. `SD+SE` 相对 `SD-only` 至少 3/4 数据集不下降；
2. 四库 macro AUPR 额外提高至少 `+0.002`；
3. 任一数据集额外下降不超过 `0.005`。

若主门槛通过但 self-exclusion 门槛未通过，则只保留 `SD-only`，第三项候选贡献表述为“支持度解耦归纳评分”；self-exclusion 记为训练消融 No-Go。若主门槛也未通过，则整条路线终止，不搜索软门控、支持度阈值、混合系数或数据集特定规则。

## 9. 单折训练结果

四库统一无稠密注意力单折 Pilot 已完成。匹配基线与两个候选均使用同一 Strict compound cold-start split、seed 2026 和 inner-validation：

| 数据集 | HerbOnly baseline | SD-only | SD+SE | SD-only - baseline | SD+SE - SD-only |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.606038 | 0.669984 | 0.593365 | +0.063946 | -0.076619 |
| TCMSP | 0.950699 | 0.963492 | 0.962525 | +0.012793 | -0.000967 |
| SymMap2.0 | 0.806066 | 0.808106 | 0.801841 | +0.002040 | -0.006265 |
| ETCM2.0 mention10 | 0.861319 | 0.896820 | 0.876714 | +0.035501 | -0.020106 |
| **Macro** | **0.806031** | **0.834601** | **0.808611** | **+0.028570** | **-0.025989** |

判定如下：

```text
SD-only：4/4 数据集提高，macro +0.028570，最小增量 +0.002040；通过主门槛。
SD+SE：相对 SD-only 为 0/4 提高，macro -0.025989；self-exclusion 门槛明确未通过。
```

因此冻结以下决策：

1. `SD-only` 进入四库完整五折 compound cold-start 验证；
2. 第三项候选贡献命名为**支持度解耦归纳评分**，核心是根据当前训练折 C-P 支持度确定性关闭不可靠的 compound-ID 基础分；
3. 直接 self-exclusion 作为消融 No-Go，不进入主模型、完整五折或后续调参；
4. 不搜索软门控、degree 阈值、基础分混合系数或数据集特定规则；
5. 本轮仅为 inner-validation 模型选择结果，不写入最终性能主表。

原始批处理结果位于：

```text
results/batch_runs/sdis_pilot_20260718_203452/
```

## 10. 完整五折确认协议

SD-only 单折通过后，下一阶段只比较以下两个模型：

```text
HerbOnly baseline：静态 Hctx-P，inductive.context=False
SDIS：静态 Hctx-P，零训练支持基础分抑制，self-exclusion=False
```

四库均复用既有 Strict compound cold-start 五折 manifest，统一设置：

```text
random.seed=2026
validation.seed=102026
early.stopping=True
evaluation.outer.test=True
attention.max.nodes=0
CHCR / CMIT / support router / FBHA / HILGA=False
```

批处理命令：

```bash
./run_sdis_full_batch.sh --dry-run
./run_sdis_full_batch.sh
```

完整五折 Go/No-Go 条件预注册为：

1. SDIS 相对匹配 HerbOnly 的 outer-test AUPR 均值至少在 3/4 数据库不下降；
2. 四库 AUPR 均值的 macro 增量不低于 `+0.005`；
3. 任一数据库的 AUPR 均值下降不超过 `0.005`；
4. 至少三个数据库各有不少于 3/5 个 fold 的 AUPR 配对增量为正；
5. 所有选择与判定只依据冻结配置，不增加支持度阈值、软 gate 或数据库特定参数。

通过后，SDIS 才正式成为第三项模型贡献，并允许进行一次 `SDIS + CHCR` 互补性实验；未通过则保留为冷启动机制观察，不进入最终主模型。

## 11. 完整五折结果

四库 HerbOnly 与 SDIS 的完整五折 outer-test 已完成：

| 数据集 | 模型 | AUC | AUPR | Recall@0.5 | Precision@0.5 | F1@0.5 | AUPR 增量 | AUPR 正向折数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TCM-Suite | HerbOnly | 0.639919 | 0.643662 | 0.476196 | 0.641446 | 0.541989 | - | - |
| TCM-Suite | SDIS | 0.721851 | 0.702967 | 0.178319 | 0.775422 | 0.289415 | +0.059305 | 5/5 |
| TCMSP | HerbOnly | 0.922903 | 0.918342 | 0.446080 | 0.953149 | 0.603776 | - | - |
| TCMSP | SDIS | 0.947252 | 0.941233 | 0.360789 | 0.964839 | 0.475087 | +0.022891 | 5/5 |
| SymMap2.0 | HerbOnly | 0.783910 | 0.797639 | 0.690557 | 0.739850 | 0.713885 | - | - |
| SymMap2.0 | SDIS | 0.807888 | 0.809854 | 0.406044 | 0.873108 | 0.553425 | +0.012215 | 5/5 |
| ETCM2.0 mention10 | HerbOnly | 0.891241 | 0.881279 | 0.431319 | 0.923382 | 0.587135 | - | - |
| ETCM2.0 mention10 | SDIS | 0.916521 | 0.898965 | 0.265234 | 0.944127 | 0.410331 | +0.017686 | 5/5 |

四库均值的 macro 变化为：

```text
AUC:       +0.038885
AUPR:      +0.028024
Recall@0.5:-0.208442
Precision: +0.074917
F1@0.5:    -0.179632
```

预注册门槛全部通过：

```text
数据库均值不下降：4/4
Macro AUPR 增量：+0.028024 >= +0.005
最小单库 AUPR 增量：+0.012215
逐折 AUPR 方向：TCM-Suite 5/5、TCMSP 5/5、SymMap2.0 5/5、ETCM2.0 5/5
总计：20/20 折提高
```

因此 SDIS 正式通过完整五折确认，可作为第三项模型贡献。但其证据边界是**compound cold-start 排序改进**：AUC/AUPR 和 Precision 提高，固定 `0.5` 阈值下 Recall/F1 明显下降，表明关闭基础 ID 分支后分数尺度发生变化。不能声称 SDIS 同时改善所有二分类指标。

下一步只允许执行两项固定工作：

1. 使用每折 inner-validation 选择 F1 阈值，对现有 checkpoint 做纯推理 outer-test 校准，不重新训练、不读取 outer-test 调阈值；
2. 完成校准审计后，运行一次固定的 `SDIS + CHCR` 互补性实验，不修改 SDIS gate 或 CHCR 超参数。

原始结果位于：

```text
results/batch_runs/sdis_full_20260718_212240/
```

## 12. Checkpoint 纯推理阈值校准

SDIS 完整五折已经依据预注册 AUPR 门槛通过，阈值校准不再承担模型选择功能，只审查固定 `0.5` 阈值下 Recall/F1 下降有多少来自分数尺度变化。校准严格遵守：

1. 复用 HerbOnly 与 SDIS 已保存的四库共 40 个 checkpoint，不重新训练；
2. 每个 fold 只在该 fold 的 inner-validation 上选择 F1 最优阈值；
3. 对应 outer-test 只评价一次，不参与阈值选择；
4. AUC/AUPR 在校准前后必须完全一致，否则视为 checkpoint 恢复或候选集不一致；
5. 同时保留固定 `0.5` 和校准阈值结果，不能用校准结果替换论文原协议对照；
6. 本步骤不新增 SDIS gate、支持度阈值、混合系数或数据库特定参数。

批处理会从完整五折日志自动恢复 checkpoint 路径，支持断点续跑：

```bash
./run_sdis_calibration_batch.sh --dry-run
./run_sdis_calibration_batch.sh
```

默认输入与输出分别为：

```text
results/batch_runs/sdis_full_20260718_212240/
results/batch_runs/sdis_full_20260718_212240/calibration/
```

统一汇总写入 `calibration/summary.md` 和 `calibration/results.tsv`，各模型逐折阈值、验证集 F1 与外层测试指标保存在相应子目录的 `report.md` 和 `report.json`。完成该审计后，再执行一次冻结设置的 `SDIS + CHCR` 互补性实验。

### 12.1 校准结果

首次离线恢复时，HerbOnly 的 AUC/AUPR 与训练日志一致，但 SDIS 不一致。审查发现通用 checkpoint 评分路径遗漏了 SDIS 的 `inductive_base_gate`。该轮报告在解释前即被训练日志一致性检查判为无效；修复评分路径并加入恢复指标硬校验后，重新计算得到以下可信结果：

| 数据集 | HerbOnly 校准 F1 | SDIS 校准 F1 | F1 增量 | Recall 增量 | Precision 增量 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.658462 | 0.688976 | +0.030515 | -0.050061 | +0.068116 |
| TCMSP | 0.871900 | 0.907638 | +0.035738 | +0.021763 | +0.050494 |
| SymMap2.0 | 0.728928 | 0.755071 | +0.026144 | -0.010977 | +0.054647 |
| ETCM2.0 mention10 | 0.825268 | 0.851010 | +0.025742 | +0.038799 | +0.014695 |
| **Macro** | **0.771140** | **0.800674** | **+0.029535** | **-0.000119** | **+0.046988** |

修复后的恢复 AUC/AUPR 与原训练日志在四库全部一致。SDIS 的校准 F1 在 4/4 数据库、合计 20/20 个 fold 均提高，macro Recall 基本不变而 Precision 提高 `0.046988`。因此固定 `0.5` 阈值下的 Recall/F1 下降主要来自 SDIS 改变了分数尺度，而不是排序改进以大量漏检为代价。论文应同时报告固定阈值和 inner-validation 校准阈值结果，不能仅保留较优口径。

## 13. SDIS 与 CHCR 冻结互补性实验

校准审计完成后，只运行一次四库 `SDIS + CHCR` 完整五折。候选配置相对已完成的 SDIS 只打开冻结 CHCR：

```text
counterfactual.context=True
counterfactual.match=exact_hc_degree_disjoint
counterfactual.weight=0.05
counterfactual.margin=0.2
counterfactual.draws=20
counterfactual.seed=42026
```

其他设置保持 `random.seed=2026`、Strict compound cold-start、`attention.max.nodes=0`、静态 Hctx-P、Dot decoder、SDIS gate 和相同五折 manifest。运行命令：

```bash
./run_sdis_chcr_full_batch.sh --dry-run
./run_sdis_chcr_full_batch.sh
```

组合进入最终冷启动主模型的冻结判据为：至少 3/4 数据库 outer-test AUPR 均值不下降、四库 macro AUPR 增量不低于 `0`，且任一单库下降不超过 `0.005`。若未通过，则 SDIS 与 CHCR 仍分别保留在各自已通过的证据范围内，但不声称二者联合互补；不再调 donor、margin、loss weight 或 SDIS gate。

首次启动在训练 epoch 开始前被 SDIS 单独 Pilot 遗留的组合保护条件终止，没有产生模型结果。该保护已收窄为只禁止 No-Go 的 self-excluded H-C readout 与 CHCR 叠加；标准 SDIS base gate 与冻结 CHCR 已通过 TCM-Suite 第一折零训练图初始化检查。批处理的 awk 列解析和失败记录替换也已修复，可在原目录续跑：

```bash
HDCTI_BATCH_DIR=./results/batch_runs/sdis_chcr_full_20260719_115934 \
  ./run_sdis_chcr_full_batch.sh
```

### 13.1 完整五折结果与冻结结论

| 数据集 | SDIS AUPR | SDIS + CHCR AUPR | 增量 | 正向 fold |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.702967 | 0.683516 | -0.019451 | 0/5 |
| TCMSP | 0.941233 | 0.949015 | +0.007782 | 5/5 |
| SymMap2.0 | 0.809854 | 0.810374 | +0.000520 | 3/5 |
| ETCM2.0 mention10 | 0.898965 | 0.924126 | +0.025161 | 5/5 |
| **Macro** | **0.838255** | **0.841758** | **+0.003503** | **13/20** |

冻结判定：

```text
不下降数据库：3/4，满足
Macro AUPR：+0.003503，满足
最差单库：TCM-Suite -0.019451 < -0.005，不满足
最终：No-Go
```

CHCR 与 SDIS 的组合在 TCMSP、SymMap2.0 和 ETCM2.0 上具有正向或近中性结果，但在 TCM-Suite 五折全部下降，且均值退化远超预注册上限。因此不将 `SDIS + CHCR` 作为统一最终模型，不声称二者具有稳定跨库互补性，也不继续搜索 CHCR 权重、margin、donor 规则或 SDIS gate。

两项贡献仍按各自通过的证据范围独立保留：CHCR 是普通 Strict 随机边协议下的训练期上下文鲁棒性增强；SDIS 是 compound cold-start 下的归纳排序与校准分类增强。最终实验表应分别报告，不把两者强制堆叠为单一配置。

## 14. 当前问题、论文定位与后续进度

### 14.1 第三项创新的判定

第三项模型创新最终冻结为**支持度解耦归纳评分（SDIS）**，而不是直接 self-exclusion，也不是 `SDIS + CHCR` 组合。SDIS 已满足预注册的四库完整五折门槛：AUPR macro 提高 `0.028024`，4/4 数据库和 20/20 个 fold 同向；逐折 inner-validation 阈值校准后，F1 macro 提高 `0.029535`，同样为 20/20 折提高。因此 SDIS 可以进入论文贡献，但结论必须限定为 compound cold-start 下的归纳排序和校准分类增强。

### 14.2 当前核心问题

三项机制没有形成一个在所有协议下同时最优的单一配置。若将 Hctx-P、CHCR 和 SDIS 描述成联合 `Ours-full`，会与 TCM-Suite cold-start 上 `SDIS + CHCR` 的退化结果冲突，并产生模块堆叠、结果挑选和事后切换配置的审稿风险。

该问题不通过继续调参解决，而通过预先定义的场景化方法结构解决：

```text
共享骨干：Strict-HDCTI + Hctx-P + Dot decoder
普通随机边：共享骨干 + CHCR
Compound cold-start：共享骨干 + SDIS
```

CHCR 与 SDIS 分别处理不同的监督条件：前者约束已观测实体条件下事实药材上下文相对反事实上下文的特异性，后者在训练 C-P 支持为零时抑制不可迁移的 compound-ID 基础分。配置切换只能由评估协议和训练支持状态触发，不能按数据库结果决定。

### 14.3 冻结结论与下一阶段

当前代码、四库 SDIS 配置、checkpoint 纯推理校准工具和 `SDIS + CHCR` 冻结组合脚本已经完成。模型搜索阶段到此结束，后续执行顺序为：

1. 提交并冻结本次 SDIS 实现、配置、测试和审计文档；
2. 按普通随机边与 compound cold-start 两个协议整理主结果表和消融表；
3. 绘制共享骨干与两条场景分支的方法图，并补充参数量、时间和空间复杂度；
4. 使用 ETCM 实体映射完成 Top-K 案例解释；
5. 在投稿前审查中专门检查“统一框架是否被误写为单一联合配置”和“结果是否超出协议证据边界”。

本阶段不再搜索 CHCR donor、margin、loss weight、SDIS gate、self-exclusion 或数据库特定组合规则。
