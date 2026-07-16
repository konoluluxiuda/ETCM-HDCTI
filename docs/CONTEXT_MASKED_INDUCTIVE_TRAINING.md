# 上下文掩码归纳训练

## 研究问题

当前 HDCTI 仍以可训练 compound/protein ID embedding 为主要预测载体。即使 Hctx-P 和 CHCR 有效，模型也可能依赖训练图中已经充分观测的实体 ID，难以推广到 C-P 关系稀疏或未见实体。

上下文掩码归纳训练（Context-Masked Inductive Training，CMIT）计划在训练阶段暂时隐藏候选实体 ID，并用独立侧信息上下文替代：

```text
compound ID masked -> H-C herb context
protein ID masked  -> P-D disease context
```

标准推理仍使用完整表示。真正的 cold-start 结论必须在 compound/target 实体级隔离划分上验证，随机边 Strict fold 只能作为开发 Pilot。

## 冻结可行性审计

在实现训练损失前，先使用冻结 CHCR checkpoint 测量三种压力条件：

```text
compound_masked
protein_masked
both_masked
```

单侧进入 Pilot 的预注册门槛：masked AUPR 至少 `0.80`，并保留 full AUPR 的至少 `85%`，对应上下文非零覆盖率至少 `95%`。双侧均通过时才实现 dual-side CMIT；仅一侧通过时只实现该侧。

命令：

```bash
python tools/audit_context_mask_headroom.py \
  --config configs/HDCTI_etcm_mention10_chcr_pilot.conf \
  --checkpoint "saved_model/2026-07-15 20-14-07/hdcti_model.ckpt" \
  --fold 1 \
  --output-dir results/context_mask_headroom/etcm_mention10_fold1
```

## 冻结审计结果

审计使用 fold 1 CHCR checkpoint `saved_model/2026-07-15 20-14-07/hdcti_model.ckpt`，14,148 条 inner-validation 记录的两侧上下文覆盖率均为 100%。

| 模式 | AUC | AUPR | AUPR 保留率 |
|---|---:|---:|---:|
| Full | 0.981710 | 0.979721 | 100.00% |
| Compound masked | 0.953249 | 0.948395 | 96.80% |
| Protein masked | 0.542633 | 0.533695 | 54.47% |
| Both masked | 0.525219 | 0.521558 | 53.24% |

判定为 `supports_compound_side_cmit_pilot`。蛋白侧和双侧均未达到预注册门槛，因此 Pilot 只掩码成分 ID，不实现对称双侧版本。

## Compound-side Pilot

Pilot 保留完整预测 BCE、Hctx-P 和 CHCR，只增加成分掩码辅助 BCE：

```text
L = L_full_BCE + L_reg + L_CHCR + 0.1 * L_compound_masked_BCE
```

没有新增可训练参数。Checkpoint 仍只按 full validation AUPR 选择，masked validation AUPR 仅在恢复最佳 checkpoint 后报告。固定 `weight=0.1`，不根据 Pilot 搜索权重。

进入完整五折的预注册条件：

```text
full validation AUPR >= 0.978722
compound-masked validation AUPR >= 0.953395
```

第一项是 CHCR fold 1 Pilot `0.979722` 的 `-0.001` 非劣性界限；第二项要求相对冻结审计 `0.948395` 至少提高 `0.005`。两项必须同时满足。

运行：

```bash
./run_hdcti.sh configs/HDCTI_etcm_mention10_cmit_compound_pilot.conf
```

该配置设置 `evaluation.fold.limit=1` 和 `evaluation.outer.test=False`，只运行 fold 1 validation-only Pilot。

## Compound-side Pilot 结果

Pilot 于 2026-07-16 完成，在 epoch 24 触发早停并恢复 epoch 14 的最佳 checkpoint：

| 指标 | 结果 | 预注册门槛 | 判定 |
|---|---:|---:|---|
| Full validation AUPR | 0.972252 | >= 0.978722 | 未通过 |
| Compound-masked validation AUPR | 0.955121 | >= 0.953395 | 通过 |

相对冻结审计，compound-masked AUPR 从 `0.948395` 提高到 `0.955121`，增量为 `+0.006726`，说明辅助目标确实增强了仅依赖 H-C 药材上下文的预测能力。但 full validation AUPR 相对 CHCR fold 1 Pilot 的 `0.979722` 下降 `0.007470`，比非劣性门槛低 `0.006470`。两项指标没有同时通过，因此总体判定为：

```text
mechanism-positive / primary-task-no-go
```

按照预注册协议，compound-side CMIT 不进入完整五折，不搜索其他权重、mask 比例或双侧版本，也不作为当前论文模型创新。该结果可作为负面实验保留：H-C 上下文具有较强的独立预测能力，但在当前共享参数训练方式下，强制提高上下文归纳能力会损害完整随机边预测任务。
