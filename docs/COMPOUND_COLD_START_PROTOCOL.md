# 成分 C-P 冷启动评估协议

## 1. 目的

随机 pair 五折中，同一成分通常同时出现在训练集和测试集。现有排名审计显示，模型错误主要集中在训练 C-P degree 为 0 或 1 的成分，而随机折整体已接近饱和。因此新增 `compound_cold_start` 协议，用于检验模型仅依靠 H-C 侧信息能否为没有训练 C-P 标签的成分预测靶点。

该协议不会替代原有 `pair_stratified` 主表。二者回答的问题不同：

| 协议 | 测试问题 |
|---|---|
| `pair_stratified` | 已知成分和靶点之间缺失边的补全能力 |
| `compound_cold_start` | 没有训练 C-P 标签的成分能否依靠 H-C 上下文完成预测 |

## 2. 外层五折

配置：

```ini
experiment.protocol=strict
split.strategy=compound_cold_start
split.seed=2026
evaluation.setup=-cv 5
```

划分规则：

1. 以 compound ID 为分组单位，同一成分的全部 C-P 正样本进入同一测试 fold。
2. 为每个成分固定采样与其正样本数相同的未观测 C-P pair，正负样本随成分一起进入同一 fold。
3. 优先使用数据集已有 ZERO 文件中的同成分候选；不足部分从该数据集已知 protein 全集中确定性补采。
4. 按每个成分的正边数进行确定性贪心负载均衡，使五折正样本规模尽量接近。
5. 每个外层 fold 均检查训练 compound 与测试 compound 交集为 0，且每个测试 compound 的正负样本数相等。

划分输出：

```text
dataset/ETCM2.0_core_mention10/splits/
└── strict_compound_cold_start_seed_2026_k5/
    ├── manifest.json
    ├── fold_assignments.tsv
    └── test_fold_0.txt ... test_fold_4.txt
```

manifest 记录 `split_strategy`、划分算法、源文件哈希、负样本来源与补采数量、每折成分数和 assignments 哈希。旧版未包含 `split_strategy` 的 version 3 manifest 只按 `pair_stratified` 解释，不会被冷启动配置误用。

## 3. 内层验证与早停

启用早停时，外层训练部分不能再进行随机 pair 划分。当前实现会自动使用同一个 `split.strategy`：

```text
outer-train compounds
├── inner-train compounds
└── validation compounds
```

inner-train 和 validation 的 compound 交集为 0。验证 compound 的全部 C-P pair 被整体保留，用于选择最佳 validation AUPR；validation 与外层 test 标签都不会进入训练 C-P PageRank、CHCR 或其他标签依赖统计。

## 4. 协议边界

这里的冷启动是“C-P 标签冷启动”，不是完全未知实体的归纳学习：

* 测试 compound 不具有任何训练 C-P 边；
* 测试 compound 仍存在于固定 H-C 侧图中，并通过药材上下文获得表示；
* P-D 仍作为固定、与测试 C-P 标签无关的侧信息；
* H-D 默认关闭；
* 因此结果应表述为 `compound C-P cold-start` 或“基于 H-C 侧信息的成分冷启动”，不能表述为完全无侧信息的新实体预测。

## 5. Pilot 对照

先只运行首折，并保持数据、seed、早停和 Dot decoder 完全一致：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_no_context_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_herb_only_pilot.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_chcr_pilot.conf
```

比较顺序：

1. `HerbOnly - NoContext`：判断静态 Hctx-P 是否真正改善无 C-P 标签成分。
2. `CHCR - HerbOnly`：判断反事实约束是否在冷启动场景提供额外增益。
3. 以 AUPR 为主指标，同时报告 AUC、Recall、Precision 和 F1。

首折只用于 Go/No-Go。只有至少一个创新模型相对 NoContext 明确提升，才运行完整五折；若三者差异很小或创新模型下降，则停止该分支，不做 seed 扩展。

## 6. 代码验证

```bash
/home/zry/.conda/envs/HDCTI_tfnew/bin/python -m unittest tests.test_strict_protocol -v
```

测试覆盖：确定性复用、成分互斥、逐成分正负匹配、ZERO 不足时的笛卡尔补采、manifest 策略冲突拒绝，以及成分互斥的内层验证。

## 7. ETCM2.0 mention10 实际划分审查

2026-07-16 已生成 seed 2026 的五折 manifest，但尚未运行 GPU 训练：

| 项目 | 数值 |
|---|---:|
| compound 总数 | 9,519 |
| H-C 支撑 compound | 9,519（100%） |
| 正样本 | 88,431 |
| 固定未观测样本 | 88,431 |
| ZERO 不足的 compound | 3,817 |
| 确定性补采 pair | 27,212 |
| fold 1 测试 compound | 1,904 |
| fold 1 测试正例 / 未观测例 | 17,687 / 17,687 |
| fold 1 外层 train/test compound 交集 | 0 |
| fold 1 inner-train / validation compound | 6,853 / 762 |
| fold 1 inner-train / validation records | 127,186 / 14,302 |
| assignments SHA-256 | `a4bfe2d50cedb0b9d4fd3e83b0b33c9b56a20c9fd44a395bdca4472e66338c56` |

其余四折测试正例分别为 17,686，测试 compound 为 1,903--1,904。所有 fold 都保持逐成分 1:1 正负匹配。

## 8. 首折 GPU Pilot 结果

2026-07-16 完成 NoContext、静态 HerbOnly 和 HerbOnly + CHCR 的相同首折比较：

| 模型 | Best epoch | Validation AUPR | Outer AUC | Outer AUPR | Recall@0.5 | Precision@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoContext | 2 | 0.323300 | 0.109664 | 0.323027 | 0.055125 | 0.088315 | 0.067880 |
| HerbOnly | 12 | 0.859850 | 0.875951 | 0.856515 | 0.482049 | 0.891934 | 0.625853 |
| HerbOnly + CHCR | 50 | 0.908939 | 0.915252 | 0.900695 | 0.074348 | 0.931303 | 0.137704 |

配对增益：

| 比较 | AUC 增益 | AUPR 增益 |
|---|---:|---:|
| HerbOnly - NoContext | +0.766286 | +0.533488 |
| CHCR - HerbOnly | +0.039301 | +0.044180 |
| CHCR - NoContext | +0.805588 | +0.577668 |

运行时间与 checkpoint：

| 模型 | 运行时间 | Checkpoint |
|---|---:|---|
| NoContext | 118.216 s | `saved_model/2026-07-16 14-15-57/hdcti_model.ckpt` |
| HerbOnly | 213.613 s | `saved_model/2026-07-16 14-21-57/hdcti_model.ckpt` |
| HerbOnly + CHCR | 457.823 s | `saved_model/2026-07-16 14-28-09/hdcti_model.ckpt` |

该结果表明：

1. 原始编码器和 Dot decoder 无法可靠处理没有训练 C-P 标签的成分，NoContext 的 AUC 甚至低于 0.5；该结果不应被表述为一般随机边性能。
2. 显式 Hctx-P 是冷启动性能的主要来源，说明药材上下文不是随机边协议下的微小修补，而是未见 C-P 成分的必要侧信息桥梁。
3. CHCR 在 HerbOnly 上继续带来明显 AUC/AUPR 增益，且 validation 与 outer-test 方向一致，支持反事实上下文约束在冷启动场景中的作用。
4. CHCR 的固定阈值 Recall/F1 很低，但 AUC/AUPR 很高，说明固定决策阈值失配，不能解释为排序能力下降。阈值只能从 inner-validation 选择，再原样应用于 outer-test。
5. CHCR 的最佳 validation AUPR 位于最大 epoch 50，尚未出现明确收敛；在完整五折前允许仅提高最大 epoch，CHCR 权重、margin、draw 和 donor 规则继续冻结。

当前决策为 **Go**：先对三个已保存 checkpoint 做纯推理阈值校准，再为 CHCR 运行一轮 max-100 的首折早停验证。只有 max-100 的 validation AUPR 明确高于 0.908939，才用该训练上限进入完整五折。

max-100 验证命令：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_chcr_max100_pilot.conf
```

该配置设置 `evaluation.outer.test=False`，只返回最佳 validation AUPR，不再次查看 outer-test。

max-100 Pilot 于 2026-07-16 完成：epoch 90 触发早停，恢复 epoch 80 checkpoint，最佳 validation AUPR 为 `0.918578`，相对 max-50 的 `0.908939` 提高 `0.009639`。运行时间为 `823.329 s`，checkpoint 为：

```text
saved_model/2026-07-16 14-42-13/hdcti_model.ckpt
```

该结果只使用 inner-validation 进行模型选择，未执行 outer-test，满足预注册的延长训练条件。完整五折统一采用 max-100 上限、相同早停参数和同一 cold-start manifest：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_no_context.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_herb_only.conf
./run_hdcti.sh configs/HDCTI_etcm_mention10_cold_start_chcr.conf
```

NoContext 和 HerbOnly 也设置 max-100，以保持预算规则一致；早停预计仍会在较早 epoch 恢复各自最佳 checkpoint。完整结果以 AUPR/AUC 为主要判断，固定 `0.5` 阈值的 Recall/Precision/F1 另行标记，之后使用各 fold inner-validation 选择阈值做纯推理校准。

## 9. 完整五折结果

2026-07-16 完成统一 max-100 上限和内层早停的三组五折：

| 模型 | AUC | AUPR | Recall@0.5 | Precision@0.5 | F1@0.5 | 运行时间 |
|---|---:|---:|---:|---:|---:|---:|
| NoContext | 0.112368(±0.003231) | 0.324662(±0.001592) | 0.087176(±0.048737) | 0.115651(±0.032939) | 0.097923(±0.043782) | 645.833 s |
| HerbOnly | 0.899021(±0.014472) | 0.893434(±0.021620) | 0.560517(±0.044423) | 0.918498(±0.015352) | 0.695618(±0.039199) | 3597.446 s |
| HerbOnly + CHCR | 0.923912(±0.004491) | 0.915948(±0.004821) | 0.121021(±0.041026) | 0.967994(±0.006941) | 0.213150(±0.066035) | 3600.526 s |

排名指标配对均值增益：

| 比较 | AUC 增益 | AUPR 增益 |
|---|---:|---:|
| HerbOnly - NoContext | +0.786653 | +0.568772 |
| CHCR - HerbOnly | +0.024891 | +0.022514 |
| CHCR - NoContext | +0.811544 | +0.591286 |

五折结果确认首折结论可泛化到整个固定划分。CHCR 还将 AUPR 的 fold 标准差从 HerbOnly 的 `0.021620` 降至 `0.004821`，说明反事实约束同时改善平均排序和折间稳定性。NoContext 的 AUC 在五折均值上仍为 `0.112368`，因此首折的反向排序不是单折偶然现象。

当前不能根据固定 `0.5` 阈值声称 CHCR 改善 Recall 或 F1。CHCR 的 AUC/AUPR 与 Precision 很高，但 Recall 很低，表明输出分数的决策边界发生系统偏移。下一步冻结全部 checkpoint，只从每折 inner-validation 选择 F1 阈值，再对对应 outer-test 做纯推理评价。

五折 checkpoint 按 fold 顺序为：

```text
NoContext:
15-09-46, 15-11-48, 15-14-06, 15-16-26, 15-18-23

HerbOnly:
15-23-38, 15-27-09, 15-41-12, 15-55-40, 16-09-33

CHCR:
17-03-31, 17-16-33, 17-27-28, 17-40-15, 17-52-02
```

三组 checkpoint 的纯推理校准命令已固定为：

```bash
./tools/calibrate_etcm_cold_start.sh
```

脚本逐折验证 split、inner-validation 和 outer-test 哈希，输出到 `results/checkpoint_calibration/`。每折阈值只由对应 inner-validation 的 F1 选择，AUC/AUPR 不受阈值影响；结果文件同时保留固定 `0.5` 与校准阈值两套指标。

## 10. 已保存 checkpoint 的纯推理阈值评价

2026-07-16 对上述 15 个 checkpoint 完成纯推理评价。每折只在 inner-validation 上选择最大 F1 对应的阈值，再将该阈值原样应用于对应 outer-test；训练和优化器更新次数均为 0。

| 模型 | 每折阈值范围 | AUC | AUPR | 校准 Recall | 校准 Precision | 校准 F1 |
|---|---:|---:|---:|---:|---:|---:|
| NoContext | 0.282898--0.295033 | 0.112368(±0.003231) | 0.324662(±0.001592) | 0.999423(±0.000530) | 0.499856(±0.000132) | 0.666410(±0.000235) |
| HerbOnly | 0.007164--0.151731 | 0.899021(±0.014472) | 0.893434(±0.021620) | 0.865500(±0.015781) | 0.804726(±0.028431) | 0.833607(±0.012639) |
| HerbOnly + CHCR | 0.006618--0.012058 | 0.923912(±0.004491) | 0.915948(±0.004821) | 0.890830(±0.012908) | 0.829265(±0.010573) | 0.858833(±0.004066) |

CHCR 相对校准后的 HerbOnly 提高 Recall `0.025330`、Precision `0.024539` 和 F1 `0.025226`；同时保留 AUC `+0.024891`、AUPR `+0.022514` 的阈值无关增益。CHCR 的五折阈值集中在 `0.0066--0.0121`，校准 F1 的 fold 标准差也由 HerbOnly 的 `0.012639` 降至 `0.004066`。

NoContext 的校准 F1 不能视为性能恢复。测试集逐成分保持 1:1 正负比例，而该模型的 Recall 接近 1、Precision 接近 0.5，等价于几乎将全部 pair 判为正；平衡二分类中“全判正”的 F1 理论值就是 `2/3`。其 AUC `0.112368` 和 AUPR `0.324662` 仍表明排序方向失效。

这里完成的是**决策阈值选择**，不是概率校准：它没有检验 reliability curve、ECE 或 Brier score，也不能声称 sigmoid 输出可解释为校准概率。论文结果应以 AUC/AUPR 作为主要冷启动排序证据；需要报告 Precision/Recall/F1 时，只使用 inner-validation 选择的阈值，并同时说明测试集正负采样比例。

本地逐折报告位于：

```text
results/checkpoint_calibration/etcm_mention10_cold_start_no_context/report.md
results/checkpoint_calibration/etcm_mention10_cold_start_herb_only/report.md
results/checkpoint_calibration/etcm_mention10_cold_start_chcr/report.md
```

`results/` 属于忽略的运行产物目录；可复现脚本、checkpoint 映射和上述汇总进入 Git，原始报告保留在本地。
