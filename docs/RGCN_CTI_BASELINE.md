# R-GCN-CTI 同输入基线

## 1. 模型定位

`R-GCN-CTI` 是当前 Strict 协议下的异构关系图基线，用于回答：

> 在相同实体和关系输入下，把 H-C、C-P、P-D 组织成一个带类型的异构图，
> 使用关系特异消息传播，能达到怎样的 C-P 预测性能？

模型依据 R-GCN 的关系特异传播原则实现：

```text
h_i^(l+1) =
activation(
    W_self^(l) h_i^(l)
    + sum_r sum_(j in N_i^r) 1 / |N_i^r| W_r^(l) h_j^(l)
)
```

参考：

- [R-GCN 论文：Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)

## 2. Strict 异构图

节点类型：

```text
Herb
Compound
Protein
Disease
```

六类有向关系：

```text
Herb -> Compound
Compound -> Herb
Compound -> Protein
Protein -> Compound
Protein -> Disease
Disease -> Protein
```

其中 H-C 和 P-D 是固定侧信息；C-P 只使用当前 outer fold 的 inner-train 正边：

```text
C-P_RGCN(fold) = inner_train_positive_C-P
```

validation 正边、outer-test 正边和完整 `C_P.txt` 均不进入 R-GCN 传播图。每个
有向关系分别按目标节点在该关系下的入度归一化，计算图使用矩形稀疏邻接，不构造
全节点稠密矩阵。

## 3. 冻结实现

统一设置：

```text
R-GCN layers = 2
relation transformations = 6 matrices per layer
self-loop transformation = 1 shared matrix per layer
hidden activation = ReLU
final layer activation = identity
pair decoder = Dot
objective = BCE
embedding dimension = 64
```

当前实现不使用：

```text
PageRank
self-gating
dense full-node attention
Hctx-P / CHCR / SDIS
hyperedge attention / HILGA
```

## 4. 与原始 R-GCN 的差异

原始 R-GCN 链路预测设置使用关系解码器处理多关系三元组。本项目预测单一
C-P 关系，并已冻结正负 pair 与二分类评价协议，因此采用 Dot decoder 和 BCE。
论文中必须写为：

```text
R-GCN-CTI (same-input BCE adaptation)
```

不能写成未经修改的原始 R-GCN 复现。

## 5. 实现与审计

模型文件：

```text
RGCNCTI.py
util/rgcn.py
```

metadata 记录：

- 四类节点数量；
- H-C、inner-train C-P、P-D 唯一边数；
- 六类有向关系的矩阵形状、活跃节点和边哈希；
- 图来源、传播层数、归一化方式和早停 epoch。

测试同时检查：

- validation/test C-P 正边不进入传播图；
- 六类关系均为有向、去重、按目标关系度归一化；
- 计算图含 relation-specific message passing；
- 不含 LightGCN 传播算子；
- frozen 配置拒绝项目增强模块。

## 6. 四库单折 Pilot

```bash
./run_rgcn_cti_pilot_batch.sh --dry-run
./run_rgcn_cti_pilot_batch.sh
```

统一协议：

```text
Strict pair-stratified split
seed = 2026
first fold only
maximum epoch = 50
inner-validation AUPR early stopping
outer-test disabled
```

pilot 只用于判断适配是否稳定，不进入最终论文主表。

四库单折 pilot 已于 2026-07-28 完成：

| 数据集 | Validation AUPR | 最佳 epoch | 停止 epoch | 运行时间 |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.993539 | 4 | 14 | 9.3s |
| TCMSP | 0.981896 | 2 | 12 | 9.2s |
| SymMap2.0 | 0.945934 | 2 | 12 | 9.0s |
| ETCM2.0 mention10 | 0.974539 | 2 | 12 | 35.1s |

四个任务均正常完成，未出现 NaN、图构建错误或训练崩溃，因此适配可行性通过。
R-GCN-CTI 相对 LightGCN-CTI 的四库单折 Validation AUPR 分别提高
`0.002988`、`0.003967`、`0.017630` 和 `0.009903`；相对 Dual-HGNN-CTI
则为 `+0.000603`、`-0.000333`、`-0.003952` 和 `+0.004639`。该方向说明
关系异构图与双超图具有互补归纳偏置，但单折内层验证不能用于正式优劣声明。

下一步统一生成 Dual-HGNN-CTI、LightGCN-CTI 和 R-GCN-CTI 的冻结五折配置，
不根据本次 pilot 为任何单独数据集调整参数。
