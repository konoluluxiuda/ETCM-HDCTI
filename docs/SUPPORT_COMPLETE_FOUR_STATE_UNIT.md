# 支持状态完备的共享训练单元

## 1. 目的

此前 Target-cold 与 Double-cold 使用不同训练单元，只能筛选分支可行性，不能
证明一个 checkpoint 能同时处理多种监督可用性。现新增：

```text
build_four_state_support_unit(...)
```

它在同一个训练 C-P 图上构造四个互斥评价集合：

| 状态 | Compound 训练 C-P 支持 | Target 训练 C-P 支持 |
|---|---|---|
| warm-warm | 有 | 有 |
| cold-warm | 无 | 有 |
| warm-cold | 有 | 无 |
| cold-cold | 无 | 无 |

该协议是后续统一路由模型的实验基础，不单独构成模型创新。

## 2. 构造规则

固定：

```text
compound group: 0
protein group: 0
warm-warm pair holdout ratio: 0.1
seed: 2026
negative ratio: 1:1
```

1. 从既有 support-complete manifest 读取 H-C 支撑 compound 分组和 P-D 支撑
   protein 分组；
2. 将 `C0` 的 compound 与 `P0` 的 protein 从共享训练图全部移除；
3. 在剩余 warm-warm 正边中确定性留出 10%，并保证每个被留出端点仍在训练
   正边中出现；
4. 用最终训练正边重新构造 compound-matched 训练负例；
5. 在四个状态各自的合法矩形候选空间中确定性采样 1:1 未观测对；
6. 所有负例均排除完整数据库中的已知 C-P 正边；
7. 训练 pair 与四个测试集合、四个测试集合彼此均不得重叠；
8. 输出训练记录、各状态记录和完整 assignment SHA-256。

## 3. 四库真实规模

| 数据集 | 训练正例 | warm-warm | cold-warm | warm-cold | cold-cold |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 27,711 | 3,079 | 6,953 | 4,074 | 999 |
| TCMSP | 34,507 | 3,834 | 9,113 | 6,374 | 1,645 |
| SymMap2.0 | 22,235 | 2,470 | 5,757 | 5,681 | 1,470 |
| ETCM2.0-mention10 | 50,916 | 5,657 | 14,160 | 13,932 | 3,516 |

表中四状态数字为正例数；每个状态另有等量负例。四库均不存在空状态，最小状态
为 TCM-Suite cold-cold 的 999 条正例。

## 4. 冻结哈希

| 数据集 | Assignment SHA-256 |
|---|---|
| TCM-Suite | `22cb8ff6b091c9c754f84555b435189ef1d819661d15a93995b32d2c7b02ae38` |
| TCMSP | `815ed866f7b89c968068f17c7defaafa3cb134c501e51686a8cbcd7a28583c92` |
| SymMap2.0 | `3618f0e32c8e6e7aabed7a9b65eb6c03a99e863e87da1aaaba0f97272bc87f6c` |
| ETCM2.0-mention10 | `927b35baed2913dee55808bbb3f69443f22bb2c8f076ca829eccd20735c9b41f` |

当前哈希由代码和冻结源 manifest 确定，还未写成新的磁盘 manifest。模型接入前
需要增加显式 artifact writer，并在重复运行时校验这些哈希。

## 5. 验证

新增：

```text
tests/test_support_complete_four_state.py
```

已验证：

- 四状态均使用同一训练图；
- 状态由训练 C-P 支持度唯一确定；
- 训练和测试 pair 无重叠；
- 四个测试集合彼此无重叠；
- 正负数量相同；
- 未将已知正例采为负例；
- 相同 seed 输出完全一致；
- 非法 holdout ratio 会被拒绝。

相关 support-complete 与 Strict 回归测试共 37 项通过。

## 6. 下一步

1. 将共享单元写成不可变 manifest/artifact，并增加加载时哈希校验；
2. 为共享单元构建同样四状态完备的 inner-validation；
3. 实现单 checkpoint 的支持状态解耦评分：

```text
warm-warm: base + Hctx-P
cold-warm: Hctx-P
warm-cold: base
cold-cold: isolated Hctx-Dctx
```

4. 所有路由只读取当前训练单元的 C-P degree 与 H-C/P-D 可用性；
5. 首个模型 Gate 仍只用 TCM-Suite `C0-P0` inner-validation，不先读取四库
   outer test。
