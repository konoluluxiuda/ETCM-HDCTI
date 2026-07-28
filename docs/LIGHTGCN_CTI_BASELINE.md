# LightGCN-CTI 同输入基线

## 1. 模型定位

`LightGCN-CTI` 是当前 Strict 协议下的 pair-only 结构基线，用于回答：

> 只使用训练折已知 C-P 二部图传播，不使用 H-C、P-D 或 HDCTI 模块时，
> 能达到怎样的 C-P 预测性能？

它依据 LightGCN 的核心传播原则实现：

- 不使用特征变换矩阵；
- 不使用非线性激活；
- 只执行归一化邻居聚合；
- 均匀聚合第 0 层至第 K 层表示；
- 使用点积产生 pair score。

参考：

- [LightGCN 论文](https://arxiv.org/abs/2002.02126)
- [LightGCN 官方 TensorFlow 仓库](https://github.com/kuandeng/LightGCN)

## 2. Strict 图构建

每个 outer fold 先划分 inner-train 和 validation。LightGCN 图只使用
inner-train 中标签为 1 的 C-P pair：

```text
G_lightgcn(fold) = inner_train_positive_C-P
```

validation 正边、outer-test 正边以及完整 `C_P.txt` 均不进入传播邻接。
通用数据加载器仍使用完整实体全集建立稳定 ID 空间，但 LightGCN 前向图不读取
H-C 或 P-D 边。

对于 C-P 二部邻接矩阵 A，使用：

```text
A_norm = D^(-1/2) A D^(-1/2)
E^(k+1) = A_norm E^k
E_final = mean(E^0, E^1, ..., E^K)
s(c,p) = <e_c, e_p>
```

当前冻结设置为 `K=3`。

L2 正则仅施加在第 0 层可训练 compound/protein embedding；传播层没有可训练
变换矩阵。

## 3. 与原始 LightGCN 的差异

原始 LightGCN 面向隐式反馈推荐，通常采用 BPR 目标。当前项目已经固定了每折
正负 pair，并以 AUC/AUPR 等二分类指标评估，因此本实现使用：

```text
sigmoid cross-entropy on the frozen positive/negative pairs
```

论文中必须标记为：

```text
LightGCN-CTI (same-input BCE adaptation)
```

不能写成未经修改的原始 LightGCN 复现。

## 4. 实现约束

模型文件：`LightGCNCTI.py`

训练 metadata 会记录：

- 传播层数；
- 训练图唯一正边数；
- 活跃 compound/protein 数；
- 训练图 edge SHA-256；
- 图来源；
- early-stopping 最佳 epoch。

测试同时检查：

- 测试 compound 不会因测试边进入训练图；
- TensorFlow 图包含 LightGCN 稀疏传播；
- 不包含 H-C/P-D 传播算子；
- 可训练模型参数只有 compound/protein 初始 embedding。

## 5. 四库单折 Pilot

```bash
./run_lightgcn_cti_pilot_batch.sh --dry-run
./run_lightgcn_cti_pilot_batch.sh
```

统一设置：

```text
Strict pair-stratified split
seed = 2026
first fold only
K = 3
embedding dimension = 64
maximum epoch = 50
inner-validation AUPR early stopping
outer-test disabled
```

pilot 只用于判断适配是否稳定，不进入最终论文主表。

四库单折 pilot 已于 2026-07-28 完成：

| 数据集 | Validation AUPR | 最佳 epoch | 停止 epoch | 运行时间 |
|---|---:|---:|---:|---:|
| TCM-Suite | 0.990551 | 38 | 48 | 14.0s |
| TCMSP | 0.977929 | 6 | 16 | 9.6s |
| SymMap2.0 | 0.928304 | 2 | 12 | 8.0s |
| ETCM2.0 mention10 | 0.964636 | 34 | 44 | 24.6s |

四个任务均正常完成，未出现 NaN、训练崩溃或图构建错误，因此
`LightGCN-CTI` 的适配可行性通过，可以进入冻结五折比较。

同口径单折下，LightGCN-CTI 相对 Dual-HGNN-CTI 的 Validation AUPR 差值依次为
`-0.002385`、`-0.004300`、`-0.021582` 和 `-0.005264`。这一方向提示仅依赖
C-P 二部图的传播可能不足以覆盖 H-C/P-D 侧信息，尤其在 SymMap2.0 上差距较大；
但单折内层验证不能用于正式优劣声明，最终结论仍需冻结五折外层测试。
