# HDCTI 对比模型复现审计

## 1. 审计目的

HDCTI 原论文在 TCM-Suite、TCMSP 和 SymMap2.0 上比较了
HyperAttentionDTI、CoaDTI、HGNNLDA、DrugBAN、MCL-DTI、
PerceiverCPI、BINDTI 和 HGHDA。本文档区分三件容易混淆的事情：

1. 使用相同名称的数据库；
2. 使用相同的模型就绪输入、负样本和 fold；
3. 在当前 Strict 协议下重新训练同一算法。

结论不是“原论文基线不能使用”，而是：

> 原论文表 2 可以作为 Legacy 文献结果引用；只有获得对应的模型输入、
> 适配代码、负样本和 fold 后，才能称为原样复现。缺少这些材料时，
> 自行改造的模型必须标为 reimplementation 或 CTI-adapted baseline。

## 2. 已确认的公开材料

HDCTI 论文说明三库的 H-C、C-P 和 P-D 关系用于构建实验数据，并从未记录
C-P 对中随机抽取与正例等量的负例。论文的基线章节只列出八个模型及类别，
没有给出每个基线的输入恢复、适配方法、超参数或 fold 文件：

- [HDCTI 论文](https://academic.oup.com/bib/article/26/4/bbaf399/8229711)
- [HDCTI 官方仓库](https://github.com/tong87-bio/HDCTI)

官方仓库公开的三库目录与本地原始目录一致，主要包含：

| 数据集 | 公开文件类型 | 未公开的基线材料 |
|---|---|---|
| TCM-Suite | H-C、C-P、P-D、H-D、正负 pair | 实体名称、SMILES、蛋白序列、基线代码与 fold |
| TCMSP | 四类关系、正 pair、两列数字 ID 映射 | 可直接训练的 SMILES、蛋白序列、基线代码与 fold |
| SymMap2.0 | 四类关系、正负 pair、数字 ID 列表 | 可直接训练的 SMILES、蛋白序列、基线代码与 fold |

论文结论还把 compound SMILES 和 target sequence 的集成列为未来工作。这说明
HDCTI 主模型本身没有使用这些属性，但不能据此反推出对比模型究竟采用了哪种
未公开适配。现有公开材料不足以在“外部恢复属性”和“替换原模型输入”之间作出
可靠判断。

## 3. 八个模型的复现状态

| 模型 | 原模型主要输入 | 公开原模型代码 | 当前三库原样重训 | 当前允许用途 |
|---|---|---|---|---|
| HyperAttentionDTI | SMILES、蛋白序列 | 有 | 缺少共同属性输入 | 引用 HDCTI 报告值；获得输入后复现 |
| CoaDTI | SMILES 分子图、蛋白序列 | 需进一步核对版本 | 缺少共同属性输入 | 引用 HDCTI 报告值；获得输入后复现 |
| DrugBAN | 分子图、蛋白序列 | [有](https://github.com/peizhenbai/DrugBAN) | 缺少共同属性输入 | 引用报告值；属性子集补充实验 |
| MCL-DTI | 药物多模态属性、蛋白序列 | [有](https://github.com/wowowoj/MCL-DTI) | 缺少共同属性输入 | 引用报告值；属性子集补充实验 |
| PerceiverCPI | SMILES、ECFP、蛋白序列 | [有](https://github.com/dmis-lab/PerceiverCPI) | 缺少共同属性输入 | 引用报告值；属性子集补充实验 |
| BINDTI | 分子图、蛋白序列 | 需进一步核对版本 | 缺少共同属性输入 | 引用 HDCTI 报告值；获得输入后复现 |
| HGNNLDA | 原任务为其他生物关联预测 | 需核对论文引用与代码版本 | 必须改造任务和输入 | 只能标为 CTI-adapted |
| HGHDA | H-C/P-D 双超图及 herb-disease 监督 | [有](https://github.com/bioxjz/HGHDA) | 输出与监督对象不是 C-P | 只能标为 HGHDA-inspired CTI adapter |

PerceiverCPI 的公开接口明确要求 `smiles,sequences,label` 三列；DrugBAN
公开实现也明确使用二维分子图与蛋白序列。不能仅把 C-P 两列数字 ID 填入这些
接口后仍称为原模型复现。

## 4. 为什么同名数据库仍不能直接合并结果

### 4.1 实体与正例口径

论文表 1 的 TCMSP C-P 数为 56,169；当前本地去重后为 56,102。
SymMap2.0 原始 C-P 为 38,043 行，去重后为 37,991 条。原始行数、唯一边数和
模型实际读取的 pair 数必须分开报告。

### 4.2 负例与 fold

负例来自未记录 pair，随机种子不同会产生不同测试集。当前 Strict 协议还固定了
fold manifest、逐折训练图、内层验证和早停；HDCTI 论文及公开仓库没有提供
八个基线共用的完整输入 manifest。因此原论文表 2 与 Strict 主表不能进行逐折
显著性检验。

### 4.3 标签依赖结构

当前 Strict 协议要求 PageRank、C-P 邻接及任何 C-P 派生统计只使用 fold
训练正边。即使取得原基线代码，也必须重新接入当前 split，才能进入 Strict
主比较表。

## 5. 论文中的两层比较

### 5.1 Legacy 文献比较

单独整理 HDCTI 表 2 的八个已发表结果，表题明确写为：

```text
Results reported by the original HDCTI study under its published protocol
```

这些数字用于说明历史比较范围，不与当前 Strict 结果计算配对差值或显著性。

### 5.2 Strict 重新训练比较

所有重新训练模型必须共享：

```text
相同正负 pair
相同 fold manifest
相同训练/验证/测试边界
相同指标实现
相同模型选择规则
仅训练折可见的 C-P 派生结构
```

模型名称必须区分：

```text
Original / author implementation
Our reimplementation
CTI-adapted
Inspired baseline
```

## 6. 下一步执行顺序

### 步骤 A：请求作者复现材料

优先询问 HDCTI 作者是否可以提供：

1. 八个基线实际使用的 SMILES、蛋白序列或替代特征；
2. 匿名 ID 到模型输入的映射；
3. 基线适配代码和超参数；
4. 共用负样本与五折清单；
5. HGHDA/HGNNLDA 转换为 C-P 输出的实现。

在材料到达前，不等待、不阻塞当前论文。

### 步骤 B：实现同输入 Strict 基线

首个实现为 `Dual-HGNN-CTI`：

```text
H-C compound hypergraph encoder
P-D protein hypergraph encoder
Dot decoder
Strict fold training C-P supervision
```

它用于回答“双超图传播本身能做到什么”，不使用 PageRank、原全节点注意力、
Hctx-P、CHCR 或 SDIS。由于监督对象和 HGHDA 原模型不同，论文中标为
`HGHDA-inspired Dual-HGNN-CTI`，不标为原始 HGHDA。

截至 2026-07-28，该适配基线已通过统一编码器角色
`encoder.profile=dual_hgnn_cti` 实现。该角色会实际裁掉而非仅旁路以下模块：

```text
self-gating
PageRank weighting
dense full-node self-attention
node-dimension attention
Hctx-P / CHCR / SDIS / HILGA / hyperedge attention
```

冻结角色还会拒绝非 Strict 协议、非 Dot 解码器和误开启的增强模块。四库单折
配置与批处理入口为：

```bash
./run_dual_hgnn_cti_pilot_batch.sh --dry-run
./run_dual_hgnn_cti_pilot_batch.sh
```

四库单折 pilot 已于 2026-07-28 完成：

| 数据集 | Validation AUPR | 最佳 epoch | 运行时间 |
|---|---:|---:|---:|
| TCM-Suite | 0.992936 | 20 | 16.6s |
| TCMSP | 0.982229 | 38 | 27.2s |
| SymMap2.0 | 0.949886 | 50 | 28.1s |
| ETCM2.0 mention10 | 0.969900 | 50 | 130.4s |

四个任务均正常完成，未出现 NaN、图构建错误或训练崩溃，因此适配可行性通过。
SymMap2.0 和 ETCM2.0 在 epoch 50 仍刷新最佳值，表明固定预算下可能尚未完全
收敛。正式配对比较先保持与现有 Strict 配置一致的 50-epoch 上限，不能只为
该基线单独增加预算；若后续统一修改训练预算，所有比较方法必须同步重跑。

这些单折结果只用于确认适配模型可稳定训练，不进入最终论文主表。三种基线的
四库正式五折配置和可续跑批处理入口现已冻结，运行方法见
[外部同输入基线正式五折运行说明](EXTERNAL_BASELINE_FULL_RUN.md)。

第二、第三个 Strict 基线候选为：

```text
LightGCN：只使用 fold 训练 C-P 二部图
R-GCN：使用 H-C、fold 训练 C-P、P-D 异构图
```

这样主表同时覆盖 pair-only、异构图和双超图三类结构归纳偏置。

截至 2026-07-28，第二个基线 `LightGCN-CTI` 已实现并完成四库单折 pilot。
该模型只从 fold inner-train 正 C-P 边构造归一化二部图，采用三层无参数邻居
传播和均匀层聚合。为复用固定正负 pair，目标函数使用 BCE，因此论文名称必须
写为 `LightGCN-CTI (same-input BCE adaptation)`。

| 数据集 | Validation AUPR | 最佳 epoch | 运行时间 |
|---|---:|---:|---:|
| TCM-Suite | 0.990551 | 38 | 14.0s |
| TCMSP | 0.977929 | 6 | 9.6s |
| SymMap2.0 | 0.928304 | 2 | 8.0s |
| ETCM2.0 mention10 | 0.964636 | 34 | 24.6s |

四库均稳定完成，适配可行性通过。该结果只决定模型是否进入正式五折，不作为
论文性能结论。实现边界和运行命令见
[外部同输入基线正式五折运行说明](EXTERNAL_BASELINE_FULL_RUN.md)。

截至 2026-07-28，第三个基线 `R-GCN-CTI` 已实现并完成四库单折 pilot。它把
H-C、inner-train positive C-P 和 P-D 组织成四类节点、六类有向关系的异构图，
采用两层关系特异稀疏传播、Dot decoder 和固定 pair BCE。C-P 传播边仍严格排除
validation 与 outer-test 正边。

| 数据集 | Validation AUPR | 最佳 epoch | 运行时间 |
|---|---:|---:|---:|
| TCM-Suite | 0.993539 | 4 | 9.3s |
| TCMSP | 0.981896 | 2 | 9.2s |
| SymMap2.0 | 0.945934 | 2 | 9.0s |
| ETCM2.0 mention10 | 0.974539 | 2 | 35.1s |

四库均稳定完成，适配可行性通过。结果只决定进入正式五折，不作为最终性能
结论，详见[外部同输入基线正式五折运行说明](EXTERNAL_BASELINE_FULL_RUN.md)。

截至 2026-07-28，第四个基线 `HGT-CTI` 已完成实现、测试和四库冻结单折
pilot。它在与 R-GCN 相同的四类节点、六类有向
关系上加入节点类型 Q/K/V 投影和关系特异双头注意力，只在真实边上计算，不使用
全节点稠密注意力。为控制 ETCM2.0 百万级 P-D 边的开销，四库统一采用每种关系
对每个目标节点最多 64 个确定性入邻居，并记录采样哈希。

| 数据集 | Validation AUPR | 最佳 epoch | 运行时间 |
|---|---:|---:|---:|
| TCM-Suite | 0.989968 | 22 | 49s |
| TCMSP | 0.975035 | 8 | 37s |
| SymMap2.0 | 0.943607 | 8 | 63s |
| ETCM2.0 mention10 | 0.942948 | 18 | 208s |

四库均稳定结束，工程可行性判定为 Go。该模型用于补足“现代异构注意力”对比
覆盖，准确命名为 `HGT-CTI (same-input sparse attention adaptation)`。
其 pilot 在四库均低于 R-GCN，尤其 ETCM 差距较大；不得据此进行数据库特定
调参。已机械生成独立正式五折配置，正式结果完成前不进入论文结果表。详见
[外部同输入基线正式五折运行说明](EXTERNAL_BASELINE_FULL_RUN.md)。

### 步骤 B 正式五折结果

四种同输入基线的 16 个冻结外层五折任务已于 2026-07-28 全部完成。前三种
基线来自原 12 任务批次，HGT 来自独立四任务批次。下表列出 AUPR；完整 AUC、
Recall、Precision 和 F1-score 见
[最终统一实验结果表](FINAL_RESULTS_TABLES.md)。

| 数据集 | Dual-HGNN | LightGCN | R-GCN | HGT | Hctx-P + CHCR |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 0.992152±0.000452 | 0.990759±0.000668 | **0.993130±0.000404** | 0.989482±0.000677 | 0.992363±0.000453 |
| TCMSP | 0.982018±0.001289 | 0.978862±0.001224 | **0.985264±0.001306** | 0.976212±0.002601 | 0.985253±0.001996 |
| SymMap2.0 | 0.949971±0.003808 | 0.934204±0.002738 | 0.951821±0.003799 | 0.951169±0.001379 | **0.954387±0.002812** |
| ETCM2.0 mention10 | 0.969338±0.000897 | 0.965810±0.001194 | 0.976484±0.001367 | 0.945994±0.005670 | **0.980326±0.000980** |

R-GCN-CTI 是四个同输入基线中四库 AUPR 均最高的模型。最终随机边模型相对
R-GCN-CTI 的 macro AUPR 增量为 `+0.001408`：在 SymMap2.0 和 ETCM2.0
mention10 上较高，在 TCMSP 基本持平，在 TCM-Suite 低 `0.000767`。因此论文
应表述为“总体具有竞争力并在两个数据库提高”，不能表述为“四库全面最优”。

### 步骤 C：有条件恢复属性模型

若作者提供完整属性输入，优先复现一个序列模型和一个分子图模型，例如
HyperAttentionDTI 与 DrugBAN。若只能自行补全部分实体，则仅形成“共同属性
子集”补充表，不能替换四库 Strict 主表。

## 7. 当前决策

1. 保留 HDCTI 原论文八模型结果，不再写成“不可使用”。
2. 不把原论文报告值与当前 Strict 结果放入同一统计比较。
3. 不继续为跨库多模态门槛进行高成本人工补全。
4. `Dual-HGNN-CTI` 四库正式五折已完成。
5. `LightGCN-CTI` 四库正式五折已完成。
6. `R-GCN-CTI` 四库正式五折已完成。
7. 四种外部基线的 16 个冻结任务全部成功，结果已进入统一论文结果表；不再
   根据结果进行数据库特定调参。
8. `HGT-CTI` 使用统一 64 入邻居上限；ETCM 结果不能表述为无采样完整 HGT
   性能，HGT 低于 R-GCN 也不触发事后参数搜索。
9. 获得作者材料后，再把可原样复现的 HDCTI 对比模型补入 Strict 管线。
