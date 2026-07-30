# 支持状态冷启动显式加载 Smoke Test

- 不导入 TensorFlow，不训练模型。
- Target-cold 固定加载 fold `0`。
- Double-cold 固定加载 cell `C0/P0`。
- 训练正例、训练负例和测试记录均与 manifest 哈希核对。

| 数据集 | Target train (+/-) | Target test (+/-) | Target protein 交集 | Double train (+/-) | Double test (+/-) | Double C/P 交集 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| TCM-Suite | 38572/38572 | 5065/5065 | 0 | 30790/30790 | 999/999 | 0/0 | PASS |
| TCMSP | 47707/47707 | 7892/7892 | 0 | 38341/38341 | 1645/1645 | 0/0 | PASS |
| SymMap2.0 | 30745/30745 | 7074/7074 | 0 | 24705/24705 | 1470/1470 | 0/0 | PASS |
| ETCM2.0-mention10 | 70744/70744 | 17391/17391 | 0 | 56573/56573 | 3516/3516 | 0/0 | PASS |

## 判定

**PASS_explicit_support_complete_loader（4/4 数据集通过）**

该结果只证明显式单元可被无泄漏、可复现地重建，不代表 SCCI、C-Dctx 或 Hctx-Dctx 已经有效。下一步才是把 loader 接入单单元实验入口，先运行现有模型协议 smoke test。
