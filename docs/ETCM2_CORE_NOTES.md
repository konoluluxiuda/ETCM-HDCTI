# ETCM2.0 Core 数据集集成记录

## 修改内容

本次更新加入了 ETCM2.0 数据集准备流程，并将活动训练配置切换到生成的 core 数据集：

- `tools/build_etcm2_entity_mappings.py`
  - 解析 ETCM2.0 原始 JSON 页面。
  - 构建中药、化合物、蛋白/靶点和疾病的实体映射表。
  - 写出实体数量和来源的审查统计。
- `tools/build_etcm2_relations.py`
  - 构建正向关系表：`H_C.txt`、`C_P.txt`、`P_D.txt`、`H_D.txt`。
  - 根据正向化合物-蛋白边写出 `ONE_indices.txt`。
  - 写出关系覆盖和交集审查统计。
- `tools/create_etcm2_core.py`
  - 从 `dataset/ETCM2.0_processed` 创建 `dataset/ETCM2.0_core`。
  - 仅保留化合物出现在 `H_C` 且蛋白出现在 `P_D` 的化合物-蛋白正样本。
  - 使用随机种子 `2026` 在 `ZERO_indices.txt` 中生成 1:1 负样本。

生成的数据集目录已通过 `.gitignore` 有意排除在 git 跟踪之外：

```text
dataset/
```

脚本和本文档会被跟踪；大型数据文件仅保存在本地。

## 当前活动数据集

`HDCTI.conf` 当前指向：

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
```

当前本地 core 数据集统计：

```text
H_C.txt          36,216
C_P.txt         109,747
P_D.txt       1,991,225
H_D.txt          41,076
ONE_indices     109,747
ZERO_indices    109,747
```

core 连通性检查：

```text
有 H_C 支撑的 C_P 化合物：100%
有 P_D 支撑的 C_P 蛋白：100%
有 C_P 支撑的 P_D 蛋白：100%
有 H_C 支撑的 H_D 中药：100%
有 H_D 支撑的 P_D 疾病：51.07%
```

`P_D 中有 H_D 支撑的疾病` 比例较低，是因为 ETCM2.0 中存在大量疾病-靶点记录，但这些疾病没有对应的相关中药记录。core 数据集是针对当前 HDCTI 的化合物-蛋白训练目标优化的，并不是为了形成完整的疾病-中药闭环。

## 运行时路径修改

模型不再把辅助文件路径硬编码为 `dataset/TCMsuite`：

- `rating.py`
  - 从包含 `datapath` 的目录读取 `H_C.txt`、`C_P.txt`、`P_D.txt` 和 `H_D.txt`。
- `util/io.py`
  - 从 `datapath` 所在目录读取 `ZERO_indices.txt`。
  - 按照已加载正样本 1:1 抽取负样本，而不再使用固定的 TCMsuite 数量。
- `HDR.py`
  - 将交叉验证 `test_fold_*.txt` 文件写入当前活动数据集目录。

这样只需修改 `datapath`，就能更方便地在 TCMsuite、Symmap、TCMSP 风格和 ETCM2.0 风格目录之间切换。

## 实验配置历史

### ETCM2.0 之前的项目基线配置

项目最初使用 TCMsuite，并硬编码辅助路径：

```text
datapath=./dataset/TCMsuite/ONE_indices.txt
batch_size=2000
learnRate=-init 0.005 -max 1
num.max.epoch=50
evaluation.setup=-cv 5
```

辅助文件在 `rating.py` 和 `util/io.py` 中直接从 `./dataset/TCMsuite/` 读取。

### 第一次 ETCM2.0_core 配置

创建 `ETCM2.0_core` 后，第一次 ETCM2.0 训练尝试使用：

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
batch_size=2000
learnRate=-init 0.005 -max 1
attention.max.nodes=2000
gpu.multiprocessing=False
```

该配置保留蛋白 full self-attention，但跳过化合物 full self-attention，因为 ETCM2.0_core 有 19,242 个化合物。在本地 RTX 5060 Ti 环境中仍然遇到 CUDA 不稳定。

### 最终稳定的 ETCM2.0_core 配置

能够完成训练的配置为：

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
batch_size=1000
learnRate=-init 0.001 -max 1
attention.max.nodes=0
gpu.multiprocessing=False
num.max.epoch=50
evaluation.setup=-cv 5
```

这是当前针对本地 RTX 5060 Ti 运行 ETCM2.0_core 的推荐配置。

## GPU/OOM 与 CUDA 稳定性修改

ETCM2.0_core 有 19,242 个化合物。原始 full self-attention 代码会为化合物构建 `N x N` 注意力矩阵：

```text
19242 x 19242 ~= 每个 attention logits 矩阵 1.38 GiB
```

在 RTX 5060 Ti 上，这会在 `Softmax_6` 处导致 GPU OOM。

`HDCTI.py` 现在支持：

```text
attention.max.nodes=<integer>
```

如果某类节点数量超过该阈值，就跳过该节点类型上开销巨大的 full self-attention 模块。

在出现 CUDA illegal-address 崩溃后，还加入了以下稳定性修改：

- 替换手写 BCE：

  ```python
  log(sigmoid(x)) + log(1 - sigmoid(x))
  ```

  替换为数值更稳定的：

  ```python
  tf.nn.sigmoid_cross_entropy_with_logits(...)
  ```

- 替换已废弃的 dense/sparse matmul 路径：

  ```python
  tf.sparse_tensor_to_dense(...)
  tf.matmul(..., a_is_sparse=True)
  ```

  替换为：

  ```python
  tf.sparse_tensor_dense_matmul(...)
  ```

- 将 PageRank 权重数组转换为 `float32`。
- 在 `util/gpu.py` 中默认设置 `NVIDIA_TF32_OVERRIDE=0`。
- 将 `batch_size` 从 `2000` 降到 `1000`。
- 将学习率从 `0.005` 降到 `0.001`。
- 设置 `attention.max.nodes=0`，同时禁用化合物和蛋白节点的 full self-attention。

## 与原始 HDCTI 运行的模型差异

相对于原始 HDCTI 代码路径，稳定版 ETCM2.0_core 运行在模型/训练行为上有以下变化：

- 化合物和蛋白的完整 `N x N` self-attention 都被禁用。
- 基于 `H_C` 和 `P_D` 的超图式稀疏传播仍然启用。
- self-gating 仍然启用。
- 传播后的特征维度注意力仍然启用。
- 损失函数在数学上仍是同一个二元交叉熵目标，但现在使用 TensorFlow 基于 logits 的稳定实现计算。
- 稀疏图传播使用 TensorFlow 标准的 sparse-dense matmul 操作，而不是先把稀疏矩阵转换为 dense tensor。

这意味着稳定版 ETCM2.0_core 结果应解释为兼容 ETCM2.0_core 的 HDCTI 变体，而不是小数据集 full-attention 架构的逐位一致复现。

## 预期影响

正面影响：

- ETCM2.0_core 可以被现有训练流程加载。
- 化合物-蛋白正样本与辅助图具有较强连通性。
- GPU 训练可以避开大规模化合物 self-attention OOM，以及本地 RTX 5060 Ti 环境中观察到的 CUDA illegal-address 崩溃。
- 数据集切换现在由 `datapath` 控制，而不是依赖硬编码路径。

权衡：

- 跳过 full self-attention 会改变大数据集上的模型结构。图传播、gating 以及后续的特征维度 attention 仍然启用。
- 因此 ETCM2.0_core 上的结果不能与小数据集上使用完整化合物/蛋白 self-attention 的运行直接等同。
- 降低 `attention.max.nodes` 会进一步减少内存占用；提高该值可能增强表达能力，但也可能重新引入 OOM。

ETCM2.0_core 的推荐默认配置：

```text
attention.max.nodes=0
batch_size=1000
learnRate=-init 0.001 -max 1
```

这会同时禁用化合物和蛋白节点的 full self-attention，并使用更保守的优化器设置。选择该配置，是因为较不保守的 GPU 配置在 RTX 5060 Ti 上触发了 `CUDA_ERROR_ILLEGAL_ADDRESS`。

如果后续使用对 RTX 50 系列支持更稳定的新 TensorFlow/CUDA 栈，可以放宽该设置用于对比：

```text
attention.max.nodes=2000
```

该设置会保留蛋白 full self-attention，同时跳过化合物 full self-attention。但它在本地 RTX 5060 Ti + TensorFlow 2.6/CUDA 11.2 环境下不够稳定，因此不是当前默认配置。

## 训练运行记录

### 2026-07-02 模型/配置恢复记录

在记录稳定运行结果后，模型实现和活动配置已恢复到稳定性降级之前的 ETCM2.0_core 状态。当前恢复后的运行配置为：

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
batch_size=2000
learnRate=-init 0.005 -max 1
attention.max.nodes=2000
gpu.multiprocessing=False
```

下面的稳定运行记录作为历史实验记录保留。它们描述的是在本地 RTX 5060 Ti 上完成训练的保守配置，但在本次恢复后已不再是当前活动模型/配置状态。

### 2026-07-01 ETCM2.0_core 稳定 GPU 运行

配置：

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
evaluation.setup=-cv 5
num.max.epoch=50
batch_size=1000
attention.max.nodes=0
learnRate=-init 0.001 -max 1
gpu.multiprocessing=False
```

观察到的结果：

```text
training: 50 batch 175 loss: 232.37389
model checkpoint: ./saved_model/2026-07-01 20-42-32/hdcti_model.ckpt
fold [5] auc: 0.9692204571558858
running time: 1579.260556 s
```

说明：

- 该运行完成到 epoch 50，并保存了模型权重。
- 该结果使用上方最终稳定配置，包括禁用 full self-attention、稳定 BCE、标准 sparse-dense 图传播、较低学习率、较小 batch size，以及禁用 TF32。
- 控制台输出在 `The result of 5-fold cross validation:` 后没有给出完整的五折汇总；从提供日志中可见并记录的指标是上面的 fold `[5]` AUC。
- 该运行确认了“保守稀疏传播 + 禁用 full self-attention”的配置可以避免本地 RTX 5060 Ti 上早前出现的 GPU OOM 和 illegal-address 崩溃。

### 2026-07-04 ETCM2.0_core CPU 串行 Full-Attention 运行，Fold 1

运行上下文配置快照：

```text
datapath=./dataset/ETCM2.0_core/ONE_indices.txt
num.max.epoch=50
batch_size=2000
learnRate=-init 0.005 -max 1
gpu.multiprocessing=False
attention.max.nodes 未设置/已注释，使用 full self-attention
运行脚本：./run_hdcti_cpu.sh
设备：通过 HDCTI_FORCE_CPU=1 和 CUDA_VISIBLE_DEVICES=-1 强制仅使用 CPU
```

观察到的 fold 1 结果：

```text
epoch 50/50 finished in 9m44s
model checkpoint: ./saved_model/2026-07-04 02-18-56/hdcti_model.ckpt
fold [1]
AUC: 0.9472443569994866
AUPR: 0.9445870892292849
Recall: 0.9061486314101396
Precision: 0.8564096381391985
F1-score: 0.8805773232390313
```

说明：

- 可见日志仅为 fold `[1]`，不是最终交叉验证汇总。
- 打印出的划分大小为 `training record count: 175595` 和 `test record count: 43899`，对应 ETCM2.0_core 正负样本集合的五折 80/20 划分。
- 写入该记录时，检查到的 `HDCTI.conf` 包含 `evaluation.setup=-cv 2`；这与提供日志中的划分大小不一致，因此该指标应视为已经运行/记录的五折 fold `[1]` 进程结果。
- 该运行保持论文风格的 full self-attention 行为，因为在有效配置解析中 `attention.max.nodes` 已被注释。

### 2026-07-04 GPU Full-Attention 显存探测

已添加一个最小 attention 显存探测脚本：

```text
tools/test_attention_memory.py
```

轻量版“一头、一层、仅前向”探测在 RTX 5060 Ti 上可以完成：

```text
./tools/test_attention_memory.py
mode=attention device=gpu dim=32 heads=1 layers=1 backward=False
nodes=19242
lower_bound_matrix=1.38 GiB
result: OK
```

更接近 HDCTI 训练态的探测失败：

```text
./tools/test_attention_memory.py \
  --nodes 19242 \
  --heads 2 \
  --layers 2 \
  --backward

mode=attention device=gpu dim=32 heads=2 layers=2 backward=True
nodes=19242
lower_bound_matrix=1.38 GiB
error: RESOURCE_EXHAUSTED
失败算子： Softmax_2
失败张量形状： [19242,19242]
```

失败时的分配器摘要：

```text
创建 GPU 设备，可用显存 13358 MB
7 个大小为 1481018368 的块，总计 9.65 GiB
已使用块总计：9.73 GiB
显存池总大小：13.04 GiB
为 Softmax_2 分配 1.38 GiB 时 OOM
```

解释：

- 单个 `[19242,19242]` 前向 attention 矩阵可以放入该 GPU。
- 类 HDCTI 训练图无法放入显存，因为两头、两层、反向传播激活/梯度、Adam 状态以及 TensorFlow 分配器碎片会要求同时保留多个大型 `[nodes,nodes]` 张量。
- 因此 ETCM full-attention 的 GPU 失败是真实的显存压力问题，来自 ETCM 化合物节点规模上的 full self-attention 训练，而不是简单的数据加载或配置错误。

## 剪枝版 ETCM2.0 Core 数据集

为了减少 full self-attention 实验中的化合物节点数，已添加可复现的剪枝脚本：

```text
tools/create_etcm2_pruned_core.py
```

该脚本读取已有的 `ETCM2.0_core` 风格目录，筛选化合物，然后重建 `H_C.txt`、`C_P.txt`、`P_D.txt`、`H_D.txt`、`ONE_indices.txt`、`ZERO_indices.txt`、筛选后的映射表以及每个数据集的统计文件。原始实体 ID 会被保留；运行时加载器会将其映射为内部连续 ID。

已生成的数据集：

| 数据集 | 筛选规则 | H_C | C_P / ONE | P_D | H_D | ZERO | 中药 | 化合物 | 蛋白 | 疾病 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ETCM2.0_core_cpdeg3` | 化合物 C_P degree >= 3 | 23,625 | 99,450 | 1,815,405 | 41,075 | 99,450 | 1,708 | 11,659 | 506 | 7,693 |
| `ETCM2.0_core_cpdeg5` | 化合物 C_P degree >= 5 | 16,426 | 86,555 | 1,712,022 | 40,956 | 86,555 | 1,518 | 7,907 | 478 | 7,693 |
| `ETCM2.0_core_mention10` | 化合物 mention_count >= 10 | 25,606 | 88,431 | 1,824,967 | 41,063 | 88,431 | 1,757 | 9,519 | 509 | 7,693 |

三个生成数据集的一致性检查：

```text
ONE_indices == C_P
ZERO_indices count == ONE_indices count
ONE/ZERO overlap == 0
stats/pruned_core_stats.{json,md} present
```

生成命令：

```bash
./tools/create_etcm2_pruned_core.py \
  --input dataset/ETCM2.0_core \
  --output dataset/ETCM2.0_core_cpdeg3 \
  --min-cp-degree 3 \
  --seed 2026 \
  --negative-ratio 1.0 \
  --overwrite

./tools/create_etcm2_pruned_core.py \
  --input dataset/ETCM2.0_core \
  --output dataset/ETCM2.0_core_cpdeg5 \
  --min-cp-degree 5 \
  --seed 2026 \
  --negative-ratio 1.0 \
  --overwrite

./tools/create_etcm2_pruned_core.py \
  --input dataset/ETCM2.0_core \
  --output dataset/ETCM2.0_core_mention10 \
  --min-mention-count 10 \
  --seed 2026 \
  --negative-ratio 1.0 \
  --overwrite
```

推荐优先进行的 GPU full-attention 测试：

```text
dataset/ETCM2.0_core_cpdeg5
```

它将化合物数量从 `19,242` 降到 `7,907`，同时保留 `86,555` 条 C_P 正样本，是 16GB GPU full-attention 测试最实际的首选候选数据集。

### 2026-07-04 ETCM2.0_core_cpdeg5 GPU Full-Attention 运行，Fold 1

配置快照：

```text
datapath=./dataset/ETCM2.0_core_cpdeg5/ONE_indices.txt
evaluation.setup=-cv 5
num.max.epoch=50
batch_size=2000
learnRate=-init 0.005 -max 1
attention.max.nodes 未设置/已注释，使用 full self-attention
gpu.multiprocessing=False
环境： HDCTI_tfnew, TensorFlow 2.21.0
```

观察到的 fold 1 结果：

```text
model checkpoint: ./saved_model/2026-07-04 16-39-00/hdcti_model.ckpt
fold [1]
AUC: 0.9819161395483813
AUPR: 0.9784277950536414
Recall: 0.9733695338224251
Precision: 0.890450774190139
F1-score: 0.9300656841640448
```

说明：

- 这是可见的 fold `[1]` 结果，不是最终五折汇总。
- 该运行使用 `cpdeg5` 剪枝 core，即保留 C_P degree 至少为 5 的化合物。
- 因为 `attention.max.nodes` 被注释，所以 full self-attention 已启用。
- 与未剪枝 ETCM2.0_core 的 full-attention GPU 运行相比，该剪枝数据集把化合物节点从 `19,242` 降到 `7,907`，避免了原始化合物节点规模下观察到的 full-attention GPU OOM。

### 2026-07-04 ETCM2.0_core_mention10 GPU Full-Attention 运行

配置快照：

```text
datapath=./dataset/ETCM2.0_core_mention10/ONE_indices.txt
evaluation.setup=-cv 5
num.max.epoch=50
batch_size=2000
learnRate=-init 0.005 -max 1
attention.max.nodes 未设置/已注释，使用 full self-attention
gpu.multiprocessing=False
环境： HDCTI_tfnew, TensorFlow 2.21.0
```

Fold 5 终端输出：

```text
epoch 50/50 finished in 14s
model checkpoint: ./saved_model/2026-07-04 17-44-33/hdcti_model.ckpt
fold [5]
AUC: 0.980149874456271
AUPR: 0.9785156192005987
Recall: 0.9655094424968902
Precision: 0.8947340843594446
F1-score: 0.9287753936526066
```

五折汇总：

```text
AUC: 0.979852(±0.000780)
AUPR: 0.977567(±0.000752)
Recall: 0.966132(±0.001282)
Precision: 0.894716(±0.003983)
F1-score: 0.929050(±0.002315)
运行时间： 3780.682873 s
```

说明：

- 该运行使用 `mention10` 剪枝 core，即保留 `mention_count >= 10` 的化合物。
- 因为 `attention.max.nodes` 被注释，所以 full self-attention 已启用。
- 该运行完成全部五折，并生成了汇总指标。
- 运行时间约为 `63.01 min`。

## 重建命令

从仓库根目录执行：

```bash
python tools/build_etcm2_entity_mappings.py \
  --input dataset/ETCM2.0 \
  --output dataset/ETCM2.0_processed \
  --progress-every 500

python tools/build_etcm2_relations.py \
  --input dataset/ETCM2.0 \
  --output dataset/ETCM2.0_processed \
  --progress-every 500

python tools/create_etcm2_core.py \
  --input dataset/ETCM2.0_processed \
  --output dataset/ETCM2.0_core \
  --seed 2026 \
  --negative-ratio 1.0
```

运行训练：

```bash
conda activate HDCTI
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python main.py
```
