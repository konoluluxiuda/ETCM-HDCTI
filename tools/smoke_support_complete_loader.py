#!/usr/bin/env python3
"""Smoke-test frozen support-complete units without importing TensorFlow."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from util.support_complete_split import load_support_complete_unit


DEFAULT_MANIFESTS = {
    "TCM-Suite": (
        REPOSITORY_ROOT
        / "dataset/TCMsuite/splits/support_complete_seed_2026_k5/manifest.json"
    ),
    "TCMSP": (
        REPOSITORY_ROOT
        / "dataset/TCMSP/splits/support_complete_seed_2026_k5/manifest.json"
    ),
    "SymMap2.0": (
        REPOSITORY_ROOT
        / "dataset/Symmap/splits/support_complete_seed_2026_k5/manifest.json"
    ),
    "ETCM2.0-mention10": (
        REPOSITORY_ROOT
        / "dataset/ETCM2.0_core_mention10/splits/"
        "support_complete_seed_2026_k5/manifest.json"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one frozen target and double-cold unit per dataset."
    )
    parser.add_argument(
        "--manifest", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--target-fold", type=int, default=0)
    parser.add_argument("--compound-group", type=int, default=0)
    parser.add_argument("--protein-group", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="results/support_complete_loader_smoke"
    )
    parser.add_argument(
        "--documentation",
        default="docs/SUPPORT_COMPLETE_LOADER_SMOKE.md",
    )
    return parser.parse_args()


def parse_manifest_overrides(values):
    if not values:
        return dict(DEFAULT_MANIFESTS)
    manifests = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--manifest must use NAME=PATH: %s" % value)
        name, path = value.split("=", 1)
        manifests[name.strip()] = Path(path).expanduser().resolve()
    return manifests


def smoke_manifest(
        name, manifest_path, target_fold,
        compound_group, protein_group):
    started = time.time()
    _, _, target = load_support_complete_unit(
        manifest_path,
        "target_cold",
        fold=target_fold,
    )
    target_seconds = time.time() - started
    started = time.time()
    _, _, double = load_support_complete_unit(
        manifest_path,
        "double_cold",
        compound_group=compound_group,
        protein_group=protein_group,
    )
    double_seconds = time.time() - started
    return {
        "dataset": name,
        "manifest_path": str(Path(manifest_path).resolve()),
        "target_cold": dict(target, load_seconds=target_seconds),
        "double_cold": dict(double, load_seconds=double_seconds),
        "passed": (
            target["train_test_pair_overlap"] == 0
            and target["train_test_protein_overlap"] == 0
            and double["train_test_pair_overlap"] == 0
            and double["train_test_compound_overlap"] == 0
            and double["train_test_protein_overlap"] == 0
        ),
    }


def build_markdown(report):
    lines = [
        "# 支持状态冷启动显式加载 Smoke Test",
        "",
        "- 不导入 TensorFlow，不训练模型。",
        "- Target-cold 固定加载 fold `%d`。" % report["target_fold"],
        "- Double-cold 固定加载 cell `C%d/P%d`。" % (
            report["compound_group"], report["protein_group"]
        ),
        "- 训练正例、训练负例和测试记录均与 manifest 哈希核对。",
        "",
        "| 数据集 | Target train (+/-) | Target test (+/-) | Target protein 交集 | Double train (+/-) | Double test (+/-) | Double C/P 交集 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["datasets"]:
        target = row["target_cold"]
        double = row["double_cold"]
        lines.append(
            "| %s | %d/%d | %d/%d | %d | %d/%d | %d/%d | %d/%d | %s |"
            % (
                row["dataset"],
                target["training_positive_count"],
                target["training_negative_count"],
                target["test_positive_count"],
                target["test_negative_count"],
                target["train_test_protein_overlap"],
                double["training_positive_count"],
                double["training_negative_count"],
                double["test_positive_count"],
                double["test_negative_count"],
                double["train_test_compound_overlap"],
                double["train_test_protein_overlap"],
                "PASS" if row["passed"] else "FAIL",
            )
        )
    lines.extend([
        "",
        "## 判定",
        "",
        "**%s（%d/%d 数据集通过）**" % (
            report["decision"],
            report["passed_datasets"],
            len(report["datasets"]),
        ),
        "",
        "该结果只证明显式单元可被无泄漏、可复现地重建，不代表 SCCI、C-Dctx"
        " 或 Hctx-Dctx 已经有效。下一步才是把 loader 接入单单元实验入口，"
        "先运行现有模型协议 smoke test。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    rows = [
        smoke_manifest(
            name,
            path,
            args.target_fold,
            args.compound_group,
            args.protein_group,
        )
        for name, path in parse_manifest_overrides(args.manifest).items()
    ]
    passed = sum(row["passed"] for row in rows)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "training_steps": 0,
        "tensorflow_imported": False,
        "target_fold": args.target_fold,
        "compound_group": args.compound_group,
        "protein_group": args.protein_group,
        "passed_datasets": passed,
        "decision": (
            "PASS_explicit_support_complete_loader"
            if passed == len(rows)
            else "FAIL_explicit_support_complete_loader"
        ),
        "datasets": rows,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    documentation_path = Path(args.documentation).expanduser()
    if not documentation_path.is_absolute():
        documentation_path = REPOSITORY_ROOT / documentation_path
    documentation_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Results written to: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
