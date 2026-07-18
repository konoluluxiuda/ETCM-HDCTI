# 反事实上下文蒸馏可行性审计

## 1. 研究问题

CMIT 已表明，仅依赖 H-C 药材上下文的 compound-side 分支具有较强预测能力，但共享参数联合训练会损害完整主任务。一个自然替代方案是冻结已经训练好的 Hctx-P + CHCR Teacher，单独训练不读取 compound ID embedding 的轻量 Student。

不过，“图 Teacher 向冷启动 Student 蒸馏”本身不是新的研究问题：

* [Cold Brew（ICLR 2022）](https://openreview.net/pdf?id=1ugNpm7W6E) 已使用 Teacher-Student 图蒸馏处理缺失或不完整邻域；
* [Privileged Graph Distillation（SIGIR 2021）](https://arxiv.org/abs/2105.14975) 已让拥有交互图的 Teacher 向只使用属性图的 Student 迁移知识；
* [MOTIVE（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fdb3fa770c2e0ecbb4b7dc7083ef5be9-Abstract-Datasets_and_Benchmarks_Track.html) 强调 DTI 归纳链接预测必须分别报告随机、cold-source 和 cold-target 场景。

因此，普通 logit KD 不能作为当前论文的第三项创新。本路线只审计一个更窄的假设：

> CHCR Teacher 学到的“事实上下文应优于 H-C degree 匹配、药材集合不相交的反事实上下文”关系，能否迁移到不使用 compound ID embedding 的 Student，并同时保持候选排序性能？

该候选暂称 **Counterfactual Context Distillation（CCD）**。这里的“反事实”是可审计的合成上下文扰动，不代表已经识别生物学因果效应。

## 2. 冻结审计设计

审计使用 ETCM2.0_core_mention10、Strict fold 1 和统一无稠密注意力 CHCR checkpoint：

```text
saved_model/2026-07-17 17-42-26/hdcti_model.ckpt
```

只使用 Strict inner-validation，并按 compound 分组拆分 Student 数据：

```text
Student fit compounds:       5,721
Student evaluation compounds: 1,430
Compound overlap:                0
```

Teacher、H-C/P-D 编码器和全部 checkpoint 参数保持冻结。Student 输入为：

```text
H-C compound context
protein embedding
element-wise product
absolute difference
```

Student 不直接读取 compound ID embedding。普通 KD 只拟合 Teacher factual logits；CCD 在相同 Ridge Student 上额外拟合 Teacher 的 factual-counterfactual margin。反事实 donor 只能来自 Student fit compound 池，并保持 H-C degree 相同且药材集合不相交。

outer-test 完全不读取、不评分，也不参与方法选择。

## 3. 预注册闸门

CCD 进入正式 compound cold-start Pilot 需要同时满足：

1. evaluation compound 不少于 100；
2. H-C 上下文覆盖率不低于 95%；
3. 正样本反事实 donor 覆盖率不低于 80%；
4. CCD AUPR 不低于冻结 context head AUPR 减 0.005；
5. CCD AUPR 不低于普通 KD AUPR 减 0.001；
6. CCD 与 Teacher logits 的 Spearman 不低于 0.70；
7. CCD 与 Teacher counterfactual margin 的 Spearman 不低于 0.50；
8. counterfactual margin 方向一致率不低于 70%。

上述阈值在查看完整审计结果前写入脚本，结果后不放宽。

## 4. 运行命令

```bash
python tools/audit_counterfactual_context_distillation.py \
  --config configs/HDCTI_etcm_mention10_pair_stratified_chcr_no_dense_full.conf \
  --checkpoint "saved_model/2026-07-17 17-42-26/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/counterfactual_context_distillation/etcm_mention10_fold1
```

## 5. 审计结果

| 模型 | AUC | AUPR | Teacher Spearman | Teacher logit MAE |
|---|---:|---:|---:|---:|
| Frozen Teacher | 0.981106 | 0.980785 | - | - |
| Frozen context head | 0.951380 | 0.950931 | 0.889273 | 3.164312 |
| KD Student | 0.957461 | 0.955709 | 0.912124 | 1.661424 |
| CCD Student | 0.955434 | 0.954358 | 0.907725 | 1.722824 |

反事实关系迁移结果：

| 指标 | KD | CCD |
|---|---:|---:|
| Teacher margin Spearman | 0.966234 | 0.982934 |
| Teacher margin 方向一致率 | 0.932208 | 0.948143 |
| Teacher margin MAE | 1.679626 | 0.868069 |

CCD 明显改善了反事实 margin 拟合，但其 AUPR 比普通 KD 低 `0.001351`，超过预注册的 `0.001` 非劣界限；Teacher logit Spearman 也由 `0.912124` 降至 `0.907725`。八项条件中只有 `ccd_kd_noninferiority` 未通过，因此最终判定为：

```text
mechanism-positive / ranking-no-go
```

对应脚本决策为：

```text
stop_ccd_route_before_joint_training
```

## 6. 结论与边界

本审计支持以下结论：

* H-C context-only Student 能较好迁移冻结 Teacher 的候选排序；
* CHCR 的事实—反事实 margin 可以被轻量 Student 高保真拟合；
* 但额外 margin 蒸馏没有优于普通 KD，不能据此形成新的模型贡献。

因此不实现联合训练、不运行 outer-test/完整五折，也不搜索 Ridge alpha、margin 权重、draw 数量或更复杂 Student。普通 KD 由于已有直接近邻工作，同样不进入论文创新。机器可读结果位于：

```text
results/counterfactual_context_distillation/etcm_mention10_fold1/
```

下一项候选不再围绕 ETCM 单库或 Hctx-P 同族机制继续微调，而转向四库共享且不依赖实体名称映射的跨数据库结构角色表示可行性审计。

