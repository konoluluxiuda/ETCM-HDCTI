# 特异性约束的超边重加权

## 动机

Top-K 多跳扩散审计发现，ETCM2.0_core_mention10 的 P-D-P 一阶投影已经覆盖约 `79.88%` 的有向非自身蛋白对，多跳扩散几乎不能提供新邻居。问题更接近宽泛疾病超边造成的高密度聚合与过平滑，而不是传播深度不足。

因此，本路线不增加传播层数，而是降低覆盖大量节点的宽泛药材/疾病超边贡献，使较具体的超边获得更高权重。

## 固定权重

对超边 `e` 定义：

```text
s_e = log(1 + |V| / degree(e))
```

其中 `|V|` 是该侧节点数。该公式固定，不搜索指数、温度或混合强度。候选节点上下文为 incident hyperedge embedding 的特异性加权平均，并进行 L2 归一化。

## 冻结审计

审计使用：

```text
ETCM2.0_core_mention10
Strict fold 1
CHCR checkpoint: saved_model/2026-07-15 20-14-07/hdcti_model.ckpt
```

约束：

* checkpoint、节点表示和超边表示完全冻结；
* 优化步数为 0；
* 只使用 H-C/P-D，不使用 H-D；
* outer-test 不计算；
* uniform 聚合必须以平均余弦 `>=0.999` 复现 checkpoint 上下文；
* inner-validation 确定性二分为 selection/audit。

单侧结构可用条件：

```text
上下文覆盖率 >= 95%
超边权重 CV >= 0.10
uniform 与 specificity 上下文平均余弦距离 >= 0.01
最高度数 10% 超边的贡献质量相对降低 >= 10%
```

进入训练 Pilot 还必须满足：

```text
selection 选择非负残差 alpha > 0
独立 audit AUPR 增量 >= 0.001
度数分层置换检验 p <= 0.05
正样本特异性特征均值 > 负样本
```

未通过时不搜索其他 IDF 公式或权重强度。

## 命令

```bash
python tools/audit_hyperedge_specificity.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/hyperedge_specificity/etcm_mention10_fold1
```

## 冻结审计结果

审计于 2026-07-16 完成，uniform 聚合与 checkpoint 上下文的平均余弦均为 `1.000000`，说明复算实现与模型一致。

| 侧 | 权重 CV | 平均上下文余弦距离 | 宽泛超边质量相对降低 | 判定 |
|---|---:|---:|---:|---|
| Herb / H-C | 0.163493 | 0.000890 | 16.18% | 未通过 |
| Disease / P-D | 0.378634 | 0.000008 | 15.65% | 未通过 |

固定特异性权重确实降低了最高度数 10% 超边的总体贡献，但加权前后的上下文几乎相同，均未达到平均余弦距离 `0.01` 的结构变化门槛。冻结特征也没有显示可用的独立判别能力：

| 特征 | Audit AUC | Audit AUPR |
|---|---:|---:|
| Herb specificity replacement delta | 0.474698 | 0.461348 |
| Compound-specific disease cosine | 0.540679 | 0.524786 |
| Specific context cosine | 0.493019 | 0.499084 |

最终判定为 `stop_hyperedge_specificity_route`，未进入残差系数选择与置换检验。按照预注册协议，不搜索其他 IDF 公式、指数或强度，也不实现该固定重加权训练模块。结果说明冻结超边表示已经高度同质化，传播完成后再调节聚合权重无法恢复被平滑掉的信息。
