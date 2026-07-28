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

单折结果只用于确认适配模型可稳定训练；通过后再生成五折配置并进入 Strict
重新训练比较表。

第二、第三个 Strict 基线候选为：

```text
LightGCN：只使用 fold 训练 C-P 二部图
R-GCN：使用 H-C、fold 训练 C-P、P-D 异构图
```

这样主表同时覆盖 pair-only、异构图和双超图三类结构归纳偏置。

### 步骤 C：有条件恢复属性模型

若作者提供完整属性输入，优先复现一个序列模型和一个分子图模型，例如
HyperAttentionDTI 与 DrugBAN。若只能自行补全部分实体，则仅形成“共同属性
子集”补充表，不能替换四库 Strict 主表。

## 7. 当前决策

1. 保留 HDCTI 原论文八模型结果，不再写成“不可使用”。
2. 不把原论文报告值与当前 Strict 结果放入同一统计比较。
3. 不继续为跨库多模态门槛进行高成本人工补全。
4. `Dual-HGNN-CTI` 代码与四库单折配置已完成，尚未产生四库 pilot 结果。
5. pilot 通过后的下一项代码工作是 LightGCN，随后是 R-GCN。
6. 获得作者材料后，再把可原样复现的 HDCTI 对比模型补入 Strict 管线。
