# 统一无稠密注意力 CHCR 确认实验

## 1. 目的

原全节点自注意力在不同数据集上会因实体规模产生不同的启用状态，并在 ETCM2.0 上带来二次复杂度和 CUDA 稳定性问题。HILGA 四库 Pilot 未通过后，最终候选结构固定为：

```text
Strict H-C / P-D 双超图
+ Hctx-P
+ CHCR
+ Dot decoder
+ attention.max.nodes=0
```

既有 CHCR 配置没有显式固定 `attention.max.nodes=0`，因此旧结果只作为历史机制证据，不能直接作为删除全注意力后的最终主结果。本轮只确认 CHCR 在统一无稠密注意力协议下是否仍有跨库增益。

## 2. 配对基线

直接复用已经归档的 fold 1、validation-only Hctx-P 结果，不重新训练：

| 数据集 | Hctx-P validation AUPR |
|---|---:|
| TCM-Suite | 0.992845 |
| TCMSP | 0.984166 |
| SymMap2.0 | 0.951155 |
| ETCM2.0 mention10 | 0.975974 |
| 四库 macro | 0.976035 |

候选配置只允许改变 `model.variant` 和以下 CHCR 设置：

```ini
counterfactual.context=True
counterfactual.match=exact_hc_degree_disjoint
counterfactual.weight=0.05
counterfactual.margin=0.2
counterfactual.draws=20
counterfactual.seed=42026
```

## 3. 运行命令

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_chcr_no_dense_pilot.conf
./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_chcr_no_dense_pilot.conf
./run_hdcti.sh configs/HDCTI_symmap_pair_stratified_chcr_no_dense_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_pair_stratified_chcr_no_dense_pilot.conf
```

四项均只运行 fold 1、只报告 inner-validation AUPR，并跳过 outer-test。

## 4. Pilot 结果

| 数据集 | Hctx-P | Hctx-P + CHCR | 增量 | 最佳 epoch | 运行时间 |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.992845 | 0.993309 | +0.000464 | 16 | 23.80 s |
| TCMSP | 0.984166 | 0.985510 | +0.001344 | 46 | 65.98 s |
| SymMap2.0 | 0.951155 | 0.950442 | -0.000713 | 22 | 83.16 s |
| ETCM2.0 mention10 | 0.975974 | 0.979820 | +0.003846 | 38 | 172.62 s |
| 四库 macro | 0.976035 | 0.977270 | +0.001235 | - | - |

Pilot 同时满足四项预注册条件：macro 增量 `+0.001235`；3/4 数据集提升；最大单库下降仅 `0.000713`；四库均无运行故障。因此允许进入统一无稠密注意力的完整五折配对实验。

## 5. 完整五折配置

每个数据集必须分别运行 Hctx-P 和 Hctx-P + CHCR，不能使用旧注意力协议下的历史五折作为基线：

推荐使用批处理脚本：

```bash
./run_no_dense_chcr_full_batch.sh
```

脚本按数据集成对顺序运行八个任务，分别保存完整日志，并持续更新：

```text
results/batch_runs/no_dense_chcr_full_<time>/results.tsv
results/batch_runs/no_dense_chcr_full_<time>/summary.md
results/batch_runs/no_dense_chcr_full_<time>/environment.txt
```

如运行中断，使用终端最后显示的结果目录恢复；已经成功的任务会自动跳过：

```bash
HDCTI_BATCH_DIR=./results/batch_runs/no_dense_chcr_full_<time> \
  ./run_no_dense_chcr_full_batch.sh
```

也可以手动逐项运行：

```bash
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_herb_only_no_dense_full.conf
./run_hdcti.sh configs/HDCTI_tcmsuite_pair_stratified_chcr_no_dense_full.conf

./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_herb_only_no_dense_full.conf
./run_hdcti.sh configs/HDCTI_tcmsp_pair_stratified_chcr_no_dense_full.conf

./run_hdcti.sh configs/HDCTI_symmap_pair_stratified_herb_only_no_dense_full.conf
./run_hdcti.sh configs/HDCTI_symmap_pair_stratified_chcr_no_dense_full.conf

./run_hdcti.sh configs/HDCTI_etcm_mention10_pair_stratified_herb_only_no_dense_full.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_pair_stratified_chcr_no_dense_full.conf
```

完整结果同时记录 outer-test AUC、AUPR、Recall、Precision、F1，以及每折最佳 epoch。主要证据为同一数据集、同一 fold 的 AUPR 配对增量；五折标准差不能替代逐折配对结果。

## 6. 预注册判定

Pilot 只有同时满足以下条件才生成完整五折配置：

1. 四库 macro AUPR 增量不低于 `+0.001`；
2. 至少 3/4 数据集不低于对应 Hctx-P 基线；
3. 任一数据集下降不超过 `0.003`；
4. 无 OOM、NaN、非法地址或数据泄漏告警。

Pilot 已通过。完整五折仍沿用同一 macro、覆盖库数和最大退化门槛；若完整结果未通过，则保留 CHCR 的历史机制结果，但不能将其作为统一无稠密注意力最终模型的共享性能贡献，也不搜索新的 weight、margin、draw 或 donor 规则。

## 7. 完整五折结果

结果目录：

```text
results/batch_runs/no_dense_chcr_full_20260717_171403
```

### 7.1 汇总指标

| 数据集 | 模型 | AUC | AUPR | Recall | Precision | F1-score |
|---|---|---:|---:|---:|---:|---:|
| TCM-Suite | Hctx-P | 0.989824 | 0.991955 | 0.941537 | 0.979677 | 0.960228 |
| TCM-Suite | Hctx-P + CHCR | 0.990424 | 0.992363 | 0.945705 | 0.978977 | 0.962051 |
| TCMSP | Hctx-P | 0.987174 | 0.984146 | 0.957827 | 0.952063 | 0.954928 |
| TCMSP | Hctx-P + CHCR | 0.987570 | 0.985253 | 0.958344 | 0.951800 | 0.955057 |
| SymMap2.0 | Hctx-P | 0.956458 | 0.954348 | 0.907189 | 0.885244 | 0.895901 |
| SymMap2.0 | Hctx-P + CHCR | 0.956466 | 0.954387 | 0.905425 | 0.885215 | 0.895007 |
| ETCM2.0 mention10 | Hctx-P | 0.977754 | 0.973997 | 0.939173 | 0.924968 | 0.931999 |
| ETCM2.0 mention10 | Hctx-P + CHCR | 0.982039 | 0.980326 | 0.941231 | 0.933741 | 0.937466 |

四库 macro AUPR 从 `0.976112` 提高到 `0.978082`，增量为 `+0.001971`。各库 AUPR 增量分别为：

```text
TCM-Suite:       +0.000408
TCMSP:           +0.001107
SymMap2.0:       +0.000039
ETCM2.0 mention10: +0.006329
```

完整结果满足预注册门槛：macro 增量超过 `+0.001`；4/4 数据集平均 AUPR 不下降；没有单库退化。因此 CHCR 可以冻结为统一无稠密注意力主模型的训练增强。

### 7.2 逐折 AUPR 配对一致性

| 数据集 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 非下降 folds |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | +0.000366 | +0.000543 | +0.000563 | +0.000137 | +0.000431 | 5/5 |
| TCMSP | +0.001111 | -0.000056 | +0.001428 | +0.001969 | +0.001083 | 4/5 |
| SymMap2.0 | -0.000139 | +0.000251 | +0.000215 | +0.000120 | -0.000254 | 3/5 |
| ETCM2.0 mention10 | +0.006593 | +0.006103 | +0.006430 | +0.007644 | +0.004878 | 5/5 |

增益不是由 ETCM 的单个 fold 偶然驱动，ETCM 为 5/5 同向；TCM-Suite 同样为 5/5 同向但效应很小。SymMap2.0 的均值变化接近 0，不能表述为明显改善。

### 7.3 解释边界与成本

* CHCR 对随机边划分的总体提升较小，不能声称四库均获得大幅提高；
* 主要增益来自关系更丰富的 ETCM2.0 mention10，TCM-Suite/TCMSP 是小幅稳定提升，SymMap2.0 基本持平；
* SymMap2.0 的 F1 下降 `0.000894`，说明 AUPR 非下降不等于固定阈值分类指标全面改善；
* 完整训练耗时倍率约为 TCM-Suite `1.42x`、TCMSP `2.05x`、SymMap2.0 `3.59x`、ETCM2.0 `1.35x`；
* CHCR 只在训练时构造反事实上下文，最终推理图和参数结构仍为 Hctx-P，因此不增加部署阶段推理成本。

最终结果支持将 CHCR 定位为“具有跨库非劣性、在 ETCM 和冷启动场景中更有效的训练期上下文正则”，而不是普遍带来大幅随机折增益的结构模块。
