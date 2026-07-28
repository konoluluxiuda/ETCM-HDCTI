# HGT-CTI 同输入异构注意力基线

## 1. 基线定位

`HGT-CTI` 是针对当前任务实现的同输入 Heterogeneous Graph Transformer
适配基线，用于补充现代异构注意力模型对比。它不是某篇外部论文作者代码的
原样复现，论文中应准确命名为：

```text
HGT-CTI (same-input sparse attention adaptation)
```

模型只使用四库共同具备的拓扑信息，不读取 SMILES、蛋白序列、H-D 或 ETCM
外部核验数据：

```text
节点类型：Herb、Compound、Protein、Disease
关系：H→C、C→H、C→P、P→C、P→D、D→P
```

其中 C-P 边只来自当前 fold 的 inner-train 正样本；inner-validation 和
outer-test 正边均不进入消息传播图。

## 2. 模型结构

每层为四类节点分别学习 Query、Key 和 Value 投影，并为六类有向关系分别学习
注意力变换、消息变换和关系先验。关系 \(r:s\rightarrow t\) 上的第 \(h\) 个
注意力分数为：

$$
e_{ij}^{r,h}=
\frac{
Q_t^h(z_i)^T A_r^h K_s^h(z_j)
}{
\sqrt{d_h}
}\mu_r^h
$$

同一目标节点的全部入边在所有入关系上共同做 segment softmax：

$$
\alpha_{ij}^{r,h}=
\operatorname{softmax}_{(r',j')\rightarrow i}
(e_{ij}^{r,h})
$$

消息聚合后使用节点类型输出投影、可学习残差门和 GELU。两层传播后的 Compound
与 Protein 表示使用 Dot decoder，并在冻结正负 pair 上优化 BCE。

## 3. 稀疏计算与确定性采样

该实现只在真实关系边上计算注意力，不构造全节点 \(N\times N\) 矩阵。由于
ETCM2.0 的 P-D 边约为百万级，pilot 统一设置：

```text
hgt.max.neighbors=64
hgt.sampling.seed=2026
```

采样单位是“关系 × 目标节点”：每个目标节点在每种入关系中最多保留 64 个
邻居。选择由源 ID、目标 ID、关系名和固定 seed 的确定性哈希决定。训练元数据
记录每种关系的原始边数、保留边数、覆盖节点数和边集 SHA-256。

该上限是四库统一的可扩展性约束，不允许根据单个数据库的 pilot 结果单独调整。
设置为 `0` 可保留全部真实关系边，但不作为当前冻结 pilot 协议。

## 4. 冻结 Pilot 配置

```text
configs/HGTCTI_tcmsuite_pair_stratified_pilot.conf
configs/HGTCTI_tcmsp_pair_stratified_pilot.conf
configs/HGTCTI_symmap_pair_stratified_pilot.conf
configs/HGTCTI_etcm_mention10_pair_stratified_pilot.conf
```

共同设置为：

```text
experiment.protocol=strict
split.strategy=pair_stratified
split.seed=2026
evaluation.fold.limit=1
evaluation.outer.test=False
hgt.layers=2
hgt.heads=2
hgt.activation=gelu
hgt.objective=bce
hgt.max.neighbors=64
hgt.sampling.seed=2026
pair.decoder=dot
```

先检查任务：

```bash
./run_hgt_cti_pilot_batch.sh --dry-run
```

再由用户启动四库单折 pilot：

```bash
./run_hgt_cti_pilot_batch.sh
```

## 5. Pilot 结果与正式五折

四库单折 pilot 已于 2026-07-28 完成：

```text
results/batch_runs/hgt_cti_pilot_20260728_173740/
```

| 数据集 | Validation AUPR | 最佳 epoch | 用时 |
|---|---:|---:|---:|
| TCM-Suite | 0.989968 | 22 | 49s |
| TCMSP | 0.975035 | 8 | 37s |
| SymMap2.0 | 0.943607 | 8 | 63s |
| ETCM2.0 mention10 | 0.942948 | 18 | 208s |

四库训练损失、Validation AUPR 和 checkpoint 均为有限值，未出现 OOM、非法
显存访问或数据泄漏。工程可行性判定为 **Go**。这些数值低于对应 R-GCN pilot。
ETCM2.0 差距最大，同时该库 `disease_to_protein` 方向在统一 64 邻居上限下仅
保留 `1.25%` 边；二者存在关联，但单次 pilot 不能证明采样是性能差距的唯一
原因。因此 HGT-CTI 的论文角色是统一可扩展但偏弱的现代异构注意力对照，不能
写成最强基线。

正式五折配置已严格从 pilot 配置机械派生：

```text
configs/HGTCTI_tcmsuite_pair_stratified_full.conf
configs/HGTCTI_tcmsp_pair_stratified_full.conf
configs/HGTCTI_symmap_pair_stratified_full.conf
configs/HGTCTI_etcm_mention10_pair_stratified_full.conf
```

运行前检查：

```bash
./run_hgt_cti_full_batch.sh --dry-run
```

由用户启动正式五折：

```bash
./run_hgt_cti_full_batch.sh
```

正式五折已于 2026-07-28 完成：

```text
results/batch_runs/hgt_cti_full_20260728_181436/
```

结果文件 SHA-256：

```text
64cff4189c0a8b7166b4d27dadac9097a17eff320c96066581a0eb9b50776a64
```

| 数据集 | AUC | AUPR | Recall | Precision | F1-score |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.986387±0.000871 | 0.989482±0.000677 | 0.931302±0.014526 | 0.976298±0.005841 | 0.953199±0.006846 |
| TCMSP | 0.982474±0.001537 | 0.976212±0.002601 | 0.945225±0.008599 | 0.949606±0.002760 | 0.947389±0.003961 |
| SymMap2.0 | 0.956584±0.001581 | 0.951169±0.001379 | 0.902135±0.024450 | 0.889461±0.014481 | 0.895418±0.004830 |
| ETCM2.0 mention10 | 0.956237±0.004160 | 0.945994±0.005670 | 0.892255±0.013871 | 0.896265±0.008030 | 0.894157±0.004349 |

四库正式结果和逐行配置哈希已接入 `paper_results_manifest.json`，并由
`tools/build_paper_results_tables.py` 自动校验后写入
`FINAL_RESULTS_TABLES.md`。

## 6. Go/No-Go 与结果边界

四个 pilot 已满足下列预注册条件：

1. 训练损失、Validation AUPR 和 checkpoint 均为有限值；
2. 四个任务均正常结束，无显存溢出或非法显存访问；
3. 元数据确认 C-P 图仅来自 inner-train 正边；
4. 不因单库结果修改层数、head 数、采样上限、正则或学习率。

单折 pilot 只用于工程可行性筛选，不进入论文主表。正式配置只移除了
`evaluation.fold.limit`、启用 outer test，并修改版本标签；正式五折结果才可
作为第四个同输入外部基线。不得因 pilot 低于 R-GCN 而修改单库参数。

## 7. 解释边界

HGT-CTI 增加的是“现代异构关系注意力”比较覆盖，并不解决跨库属性缺失问题。
不应将其表述为多模态基线，也不能用它替代原 HDCTI 论文中基于其他输入和划分
得到的 Legacy 文献结果。
