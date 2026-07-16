# Top-K 稀疏多跳全局扩散

## 研究问题

原始 HDCTI 已经在 H-C 和 P-D 超图上执行节点—超边—节点传播，因此一层 `C-H-C` 或 `P-D-P` 聚合不是新的全局视图。本路线只考察投影图二阶以上路径能否为冻结 CHCR 模型提供独立排序信息：

```text
Compound: C-H-C-H-C ...
Protein:  P-D-P-D-P ...
```

若冻结特征没有稳定增量，则不实现可训练模块，避免把同源重复传播包装成创新。

## 固定构造

从 incidence matrix `H` 构造行随机游走：

```text
T = D_node^-1 H D_context^-1 H^T
```

审计固定：

```text
Top-K = 20
投影图 hop = 2, 3, 4
restart = 0.15
中间候选上限 = 4K
移除自身邻居
```

一阶 `T` 只用于判断邻居是否新颖，不进入全局表示。第 2-4 阶结果按截断 PPR 权重累积，每行最终只保留 20 个非自身邻居，再聚合冻结节点 embedding。

## 冻结审计协议

使用 ETCM2.0_core_mention10、Strict fold 1 和已冻结 CHCR checkpoint：

```text
saved_model/2026-07-15 20-14-07/hdcti_model.ckpt
```

约束：

* 编码器和 checkpoint 完全冻结，优化步数为 0；
* 只使用 H-C/P-D，不使用 H-D；
* 扩散图不使用 C-P 标签；
* inner-validation 按类别确定性二分为 selection/audit；
* selection 只选择 compound/protein/dual 三个预定义特征及非负残差系数；
* outer-test 不计算，也不参与方法选择。

单侧结构可用条件：覆盖率至少 `95%`，且 Top-K 中至少 `25%` 不是一阶投影邻居。只有结构可用的侧进入残差选择。

进入训练 Pilot 必须同时满足：

```text
至少一个结构可用侧
selection 选择 alpha > 0
独立 audit AUPR 增量 >= 0.001
度数分层置换检验 p <= 0.05
正样本扩散特征均值 > 负样本
```

未通过时停止该路线，不搜索 K、hop、restart 或 alpha 网格。

## 命令

```bash
python tools/audit_sparse_global_diffusion.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/sparse_global_diffusion/etcm_mention10_fold1
```

该审计只用于判断多跳稀疏视图是否值得实现，不作为最终模型结果。

## 冻结审计结果

审计于 2026-07-16 完成，编码器优化步数为 0，outer-test 与 H-D 均未使用。

| 侧 | 节点数 | 一阶投影边 | Top-K 覆盖率 | 非一阶邻居比例 | 判定 |
|---|---:|---:|---:|---:|---|
| Compound / H-C | 9,519 | 940,766 | 99.72% | 10.58% | 未通过 |
| Protein / P-D | 509 | 206,550 | 100.00% | 0.39% | 未通过 |

P-D-P 一阶投影已经覆盖约 `79.88%` 的有向非自身蛋白对，因此第 2-4 阶扩散的 Top-20 邻居几乎全部也是一阶邻居。H-C 侧虽然整体一阶图密度仅约 `1.04%`，但最高权重多跳邻居仍有 `89.42%` 与一阶邻居重合。两侧均未达到预注册的 `25%` 结构新颖度门槛，所以没有进入残差系数选择与置换检验。

冻结特征在独立 audit 子集上的结果为：

| 特征 | AUC | AUPR | 与 baseline Spearman |
|---|---:|---:|---:|
| Compound global cosine | 0.736260 | 0.751154 | 0.368633 |
| Protein global cosine | 0.553640 | 0.551992 | 0.138992 |
| Dual global cosine | 0.508820 | 0.508744 | 0.004278 |

最终判定为 `stop_sparse_global_diffusion_route`。按照预注册协议，不搜索其他 K、hop、restart 或残差系数，也不实现可训练多跳扩散编码器。该结果说明当前 ETCM 图更需要抑制宽泛超边和过度平滑，而不是继续扩大同源传播范围。
