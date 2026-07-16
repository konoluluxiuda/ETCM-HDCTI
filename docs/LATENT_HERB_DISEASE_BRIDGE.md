# 潜在药材—疾病集合桥接

## 目的

本阶段先审计冻结 HDCTI/CHCR 表示中是否已经存在可利用的药材—疾病集合对齐信号，不立即增加可训练参数，也不运行新的完整训练。

候选成分 `c` 保留其关联药材超边集合 `H(c)`，候选蛋白 `p` 保留其关联疾病超边集合 `D(p)`。对每个候选 pair 计算药材与疾病超边嵌入之间的余弦相似度，并分别汇总 Top-1、Top-3 和 Top-5：

```text
S_HD(c,p) = TopKMean({cos(e_h, e_d) | h in H(c), d in D(p)})
```

该审计只读取 H-C 与 P-D，不读取 H-D，也不使用完整 C-P 标签构造桥接关系。

## 审计协议

1. 使用一个已保存 checkpoint，冻结所有模型参数。
2. 只在对应 Strict fold 的 inner-validation 上计算指标；outer-test 不计算指标，也不参与方法选择。Strict 协议仍使用完整实体 ID 全集维持 transductive 矩阵维度。
3. 将 inner-validation 按标签确定性二分为 selection 与 audit 两半。
4. selection 半集只选择 Top-K 聚合方式和非负残差系数。
5. audit 半集只评价一次 AUC/AUPR 增量。
6. 在药材上下文度数和疾病上下文度数联合分层内置换桥接分数，排除明显的度数替代解释。

预注册通过条件：

```text
上下文覆盖率 >= 95%
选择的残差系数 > 0
audit AUPR 增量 >= 0.001
度数分层置换 one-sided p <= 0.05
audit 正样本平均对齐分数 > 负样本
```

全部通过时，进入可训练稀疏桥接单折 Pilot。未通过只说明冻结的两个超边空间不能直接进行无参数余弦对齐，不能据此否定带低秩投影的桥接模型。

## 命令

先检查协议和文件，不加载 TensorFlow：

```bash
python tools/audit_latent_herb_disease_bridge.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --dry-run
```

执行冻结 checkpoint 审计：

```bash
python tools/audit_latent_herb_disease_bridge.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/latent_hd_bridge/etcm_mention10_fold1
```

输出包括：

```text
report.json       完整协议、文件哈希、参数选择与判定
report.md         中文摘要
pair_scores.tsv   每个内层验证 pair 的上下文规模和 Top-K 对齐分数
```

## Fold 1 审计结果

审计使用 `configs/HDCTI_etcm_mention10_chcr_pilot.conf` 和冻结 checkpoint
`saved_model/2026-07-15 20-14-07/hdcti_model.ckpt` 完成。模型参数没有更新，outer-test 未计算指标且未参与方法选择，也未使用 H-D。

| 项目 | 结果 |
|---|---:|
| Inner-validation records | 14,148 |
| 上下文覆盖率 | 100.00% |
| Selection / audit records | 7,074 / 7,074 |
| 选择的特征 | Top-1 mean |
| 选择的残差系数 | 0.000 |
| Audit baseline AUPR | 0.984019 |
| Audit fused AUPR | 0.984019 |
| Audit AUPR 增量 | +0.000000 |
| 正样本减负样本标准化对齐均值 | -0.014488 |
| 度数分层置换 p | 1.000000 |

独立对齐分数在 audit 半集上的结果为：

| 特征 | AUC | AUPR |
|---|---:|---:|
| Top-1 mean | 0.499087 | 0.502433 |
| Top-3 mean | 0.507680 | 0.509076 |
| Top-5 mean | 0.508784 | 0.509889 |

结论为 `raw_alignment_inconclusive_consider_low_rank_probe`。所有候选正残差系数都会降低 selection AUPR，因此停止直接无参数 Top-K 桥接，不进入该版本的完整训练。该结果符合预期风险：药材超边与疾病超边由两个独立编码分支生成，原训练目标从未要求两个超边空间具有可直接比较的余弦几何。

需要保留一个协议边界：该 checkpoint 曾使用完整 inner-validation 进行 early stopping。本次再二分能够隔离桥接特征、Top-K 和残差系数的选择，却不是完全独立于 checkpoint 选择的外部重复。由于本次结果为阴性，这一依赖没有被用来支持正向结论；后续若低秩 probe 得到正结果，仍需使用重新划分的内层训练/选择/审计协议确认。

若继续该方向，下一步只能测试一个冻结编码器上的小型低秩投影 probe，并继续使用 selection/audit 隔离。只有低秩 probe 在 audit 半集达到预注册增益，才值得把投影加入可训练主模型；否则终止 SHDLB 路线并转向上下文掩码归纳训练。

## 低秩投影 Probe 协议

低秩 probe 使用模型已有的候选成分药材聚合上下文与候选蛋白疾病聚合上下文。编码器、Hctx-P、CHCR 和基础 logit 全部冻结，只在 selection 半集训练 rank-8 双线性残差：

```text
r(c,p) = <h_c A, d_p B> / sqrt(8)
logit_probe(c,p) = logit_frozen(c,p) + r(c,p)
```

固定设置为 `rank=8`、`steps=500`、`learning_rate=0.01`、`L2=1e-4`，使用三个廉价 probe 初始化。不得根据 audit 结果继续搜索这些参数。通过条件为：三个初始化 AUPR 均为正增益、ensemble audit AUPR 增量至少 `0.001`、度数分层置换 `p<=0.05`，且正样本 residual 均值高于负样本。

运行命令：

```bash
python tools/probe_latent_herb_disease_bridge.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/latent_hd_bridge_probe/etcm_mention10_fold1
```

## 低秩 Probe 结果

固定 rank-8 probe 已在相同 fold 1 checkpoint 上完成。三个 probe 初始化均能降低 selection 训练损失，但都无法泛化到 audit 半集：

| Probe seed | 初始 loss | 最终 loss | Audit AUC 增量 | Audit AUPR 增量 |
|---:|---:|---:|---:|---:|
| 82026 | 0.195085 | 0.188984 | -0.000706 | -0.001059 |
| 82027 | 0.195022 | 0.188988 | -0.000704 | -0.001057 |
| 82028 | 0.195057 | 0.188984 | -0.000707 | -0.001059 |

Ensemble audit AUC 从 `0.984002` 降至 `0.983296`，AUPR 从 `0.984019` 降至 `0.982960`。独立 residual AUC/AUPR 仅为 `0.465121/0.477435`，正样本减负样本 residual 均值为 `-0.074836`；度数分层置换 `p=0.502488`。判定为：

```text
stop_latent_bridge_route_use_context_masked_inductive_training
```

因此 SHDLB 路线在当前 ETCM 表示和数据条件下终止：不实现集合级可训练桥接，不搜索其他 rank、学习率、正则、Top-K 或投影结构。原始余弦与低秩投影两级审计均未发现可泛化增量，继续修改会变成针对同一 inner-validation 的事后方法搜索。下一项独立创新转向上下文掩码归纳训练。

完整结果位于：

```text
results/latent_hd_bridge_probe/etcm_mention10_fold1/
```

完整机器可读结果位于：

```text
results/latent_hd_bridge/etcm_mention10_fold1/
```
