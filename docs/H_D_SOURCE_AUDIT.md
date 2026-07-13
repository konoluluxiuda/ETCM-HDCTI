# H-D 关系来源与泄漏风险审查

本文档审查本仓库各数据集 H-D（Herb-Disease）关系的文件来源、语义来源、与 H-C/C-P/P-D 的可重构程度，以及将其用于 C-P 预测时的潜在标签泄漏风险。核验日期：2026-07-13。

## 1. 核心结论

| 数据集 | H-D 来源判定 | 是否可作为固定、独立侧信息 |
|---|---|---|
| TCM-Suite | H-D 与完整 `H-C-P-D` 闭包逐边完全相等 | **不能** |
| TCMSP | 99.68% H-D 可由完整 `H-C-P-D` 重构 | **不能** |
| SymMap2.0 | 仅 29.18% H-D 可由该闭包重构，来源与前三者不同，但仓库缺少原始出处说明 | **暂不判定为安全** |
| ETCM2.0 | H-D 直接读取自疾病页 `Herbs`，但 100% 有完整 `H-C-P-D` 支撑；官方说明数据库同时包含算法驱动映射 | **不能作为独立生物学先验** |
| ETCM2.0 core/剪枝版 | 从 processed H-D 过滤继承，没有产生新来源 | **与 processed 相同** |

当前 `HDCTI.py` 前向传播没有使用 H-D，因此已有 HDCTI 结果不受本审查结论影响。风险只在后续把 H-D 或 `C-H-D-P` 路径加入模型时出现。

## 2. 审查方法

新增可复查脚本：[audit_hd_sources.py](../tools/audit_hd_sources.py)。

脚本执行三类检查：

1. 读取各数据集 H-C、C-P、P-D 和 H-D 唯一边。
2. 计算布尔关系闭包：

$$
R_{HD}^{path}=R_{HC}\circ R_{CP}\circ R_{PD}
$$

3. 对已有 `test_fold_*.txt`，检查测试正例和负例是否具有静态 `C-H-D-P` 路径支持。

复查命令：

```bash
python tools/audit_hd_sources.py \
  dataset/TCMsuite \
  dataset/TCMSP \
  dataset/Symmap \
  dataset/ETCM2.0_processed \
  dataset/ETCM2.0_core \
  dataset/ETCM2.0_core_mention10 \
  dataset/ETCM2.0_core_cpdeg3 \
  dataset/ETCM2.0_core_cpdeg5 \
  --output /tmp/hd_source_audit.json
```

ETCM 原始页面字段复查：

```bash
python tools/audit_hd_sources.py \
  dataset/ETCM2.0_processed \
  --etcm-raw dataset/ETCM2.0 \
  --output /tmp/etcm_hd_raw_audit.json
```

## 3. 闭包重构结果

| 数据集 | H-D 边 | H-C-P-D 闭包边 | H-D 被闭包支撑 | 闭包被 H-D 保留 | 是否完全相等 |
|---|---:|---:|---:|---:|---|
| TCM-Suite | 2,354,225 | 2,354,225 | 100.0000% | 100.0000% | 是 |
| TCMSP | 39,934 | 45,092 | 99.6820% | 88.2795% | 否，近似 |
| SymMap2.0 | 382,930 | 1,985,957 | 29.1771% | 5.6259% | 否 |
| ETCM2.0_processed | 41,076 | 13,414,548 | 100.0000% | 0.3062% | 否，H-D 是稀疏子集 |
| ETCM2.0_core | 41,076 | 13,414,548 | 100.0000% | 0.3062% | 否，继承 processed |
| ETCM2.0_core_mention10 | 41,063 | 12,912,335 | 99.9951% | 0.3180% | 否，剪枝过滤后继承 |
| ETCM2.0_core_cpdeg3 | 41,075 | 12,716,224 | 100.0000% | 0.3230% | 否，剪枝过滤后继承 |
| ETCM2.0_core_cpdeg5 | 40,956 | 11,445,819 | 99.9927% | 0.3578% | 否，剪枝过滤后继承 |

### 3.1 TCM-Suite

TCM-Suite 的 H-D 与完整关系闭包完全一致：边数相同、交集等于全集、没有额外边或缺失边。

因此可以确定：

```text
H_D = boolean(H_C × C_P × P_D)
```

如果把该 H-D 固定用于 C-P 交叉验证，就等价于以另一种关系形式重新暴露完整 C-P 标签。

### 3.2 TCMSP

TCMSP 的 39,934 条 H-D 中有 39,807 条可由完整链路重构，比例为 99.6820%。闭包中的 88.2795% 路径关系被写入 H-D。

虽然不是逐边完全相等，但这种重合度足以排除“与 C-P 独立”的假设。少量差异可能来自去重、映射、过滤或原始版本差异，不能据此视为独立关系。

### 3.3 SymMap2.0

SymMap 的 H-D 只有 29.1771% 可由当前 H-C/C-P/P-D 重构，说明其 H-D 不是简单闭包复制，可能来自独立的草药—症状/疾病映射或不同版本关系。

但当前仓库没有保存：

```text
原始下载页面
生成脚本
关系证据类型
数据库版本/日期
是否由其他隐藏靶点关系推导
```

因此它属于“尚未证明标签依赖”，而不是“已经证明独立安全”。在找到 SymMap 原始数据说明前，不建议将其与其他数据集统一作为固定 H-D 输入。

## 4. ETCM2.0 页面级来源

### 4.1 本地生成脚本

[build_etcm2_relations.py](../tools/build_etcm2_relations.py) 对两类字段做了区分：

- 疾病页面 `Related Tables → Herbs` 写入正式 `H_D.txt`，来源标签为 `disease_related_herb`。
- 药材页面 `Related Tables → Enriched Diseases` 只写入审查集合，默认不加入 H-D。
- 只有显式传入 `--include-herb-enriched-diseases` 才会把后者并入 H-D。

当前统计文件确认：

```text
include_herb_enriched_diseases = false
H_D disease_related_herb       = 41,076
H_D enriched audit             = 14,176
parse errors                   = 0
```

因此，当前 H-D 不是本仓库主动把药材页 `Enriched Diseases` 直接复制得到的。

### 4.2 两类页面字段不是简单正反索引

从 2,076 个药材 JSON 重建 `Enriched Diseases` 后：

| 指标 | 数值 |
|---|---:|
| 疾病页 `Herbs` 形成的 H-D | 41,076 |
| 药材页 `Enriched Diseases` | 14,176 |
| 两者交集 | 1,467 |
| 药材页富集边落入正式 H-D | 10.3485% |
| 正式 H-D 被药材页富集边覆盖 | 3.5714% |

这否定了“疾病页 Herbs 只是药材页 Enriched Diseases 的完全反向索引”这一假设。两类字段可能采用不同阈值、数据源或计算流程。

### 4.3 官方语义证据

ETCM v2.0 正式论文说明：

- 成分靶点同时包含确认关系和基于 BindingDB 配体二维相似性的潜在靶点。
- 药材的靶点是其所有成分靶点的集合。
- 与这些靶基因显著相关的疾病通过富集分析得到，并被展示为药材可能治疗的疾病。
- 数据库同时提供 primary data 和 data-/algorithm-driven mappings。

来源：[ETCM v2.0 正式论文](https://doi.org/10.1016/j.apsb.2023.03.012)、[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10326295/)。

结合本地量化结果，ETCM H-D 的恰当表述是：

> 疾病页面提供的、与药材及靶点网络高度一致的数据库关系；当前证据不足以把它解释为独立临床治疗关系，其中至少包含算法驱动或靶点介导的关联。

不能写成：

> ETCM2.0 提供了独立实验验证的药材—疾病疗效标签。

## 5. 测试折中的 C-H-D-P 信号

下表使用当前静态 H-D，检查每个测试 C-P pair 是否至少存在一条：

```text
Compound → Herb → Disease → Protein
```

| 数据集 | folds | 测试正例有路径 | 测试负例有路径 | 差值（百分点） |
|---|---:|---:|---:|---:|
| TCM-Suite | 5 | 58.3526% | 27.8871% | +30.4655 |
| TCMSP | 5 | 69.4539% | 8.1013% | +61.3525 |
| SymMap2.0 | 5 | 92.3587% | 78.4007% | +13.9579 |
| ETCM2.0_core | 5 | 15.3557% | 11.4286% | +3.9271 |
| ETCM2.0_core_mention10 | 5 | 16.1018% | 14.1930% | +1.9089 |
| ETCM2.0_core_cpdeg3 | 5 | 15.3163% | 11.4741% | +3.8421 |
| ETCM2.0_core_cpdeg5 | 2 | 15.7558% | 11.9404% | +3.8155 |

`cpdeg5` 当前只有两个 fold 文件，因此该行不能与完整五折结果等量比较。

路径支持率差异本身不能证明泄漏，因为真正的生物机制关系也应与正例相关。但当 H-D 已被证明由完整 C-P 闭包生成或高度重构时，这个差异表示它能向测试标签提供直接信号：TCM-Suite 和 TCMSP 的风险尤其高。

## 6. 对模型设计的判定

### 6.1 Strict 主实验

以下关系不能作为所有 fold 共享的固定 H-D：

```text
TCM-Suite/H_D.txt
TCMSP/drug-disease.txt
ETCM2.0*/H_D.txt
```

建议 Strict 主实验默认：

```text
use_hd = false
```

### 6.2 折内重建 H-D

可以只用当前 fold 的训练正边重建：

$$
R_{HD}^{train}=R_{HC}\circ R_{CP}^{train}\circ R_{PD}
$$

这样能避免测试边直接进入关系，但必须将其命名为：

```text
training-label-derived H-D view
```

不能称为独立生物学先验。它本质上是训练 C-P 标签的高阶传播视图，与 RSGCL-DTI 的标签派生关系图属于同类设计，需要单独消融。

### 6.3 机制解释

现有 ETCM H-D 可以用于：

- 训练完成后的候选路径展示；
- 提出待文献或实验验证的机制假设；
- 作为算法生成证据单独标注。

不能用于：

- 证明模型预测得到了独立数据库验证；
- 同时作为模型输入和外部验证证据；
- 把算法富集关系描述为临床疗效证据。

### 6.4 真正独立的 H-D

若要恢复 C-H-D-P 作为主创新，需要重新构建带来源的 H-D：

| 必需字段 | 示例 |
|---|---|
| herb_id / disease_id | 统一实体映射 |
| source_database | HERB 2.0、文献数据库、临床处方数据等 |
| evidence_type | curated literature、clinical、experimental、algorithmic |
| source_id | PMID、数据库记录号 |
| confidence | 数据库等级或自定义证据等级 |
| retrieved_at | 获取日期 |
| depends_on_cp | 是否由 C-P 推导 |

只有 `depends_on_cp=false` 且来源可追溯的关系，才能在所有 fold 中作为固定侧信息。

## 7. 最终建议

1. 当前主模型不使用固定 H-D。
2. 优先研究只依赖 H-C 和 P-D 的候选级上下文交互。
3. 若保留 H-D 实验，只做“训练标签派生视图”并逐折重建。
4. ETCM H-D 只用于 post-hoc 假设生成，并标注为数据库/算法驱动关联。
5. SymMap H-D 在找到原始数据说明前单独处理，不与其他数据集做统一语义声明。
6. 后续若获取独立文献或临床 H-D，再启用 C-H-D-P 跨超图桥接作为正式模型模块。

