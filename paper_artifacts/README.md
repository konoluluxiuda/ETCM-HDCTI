# 论文可移植结果包

`paper_artifacts/results/` 保存生成 `docs/FINAL_RESULTS_TABLES.md` 所需的最小
TSV/JSON 结果集。原始训练日志、checkpoint 和数据集仍保留在本地且不进入 Git。

重新导出并校验：

```bash
python -m tools.export_paper_result_bundle
python tools/build_paper_results_tables.py
```

`results/MANIFEST.json` 记录每个文件的原始本地路径、SHA-256 和大小。导出的
文件内容与原始冻结结果逐字节一致，因此迁移路径不会改变任何论文数字。
