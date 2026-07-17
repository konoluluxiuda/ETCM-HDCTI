#!/usr/bin/env python3
"""Audit compound cold-start feasibility across all HDCTI datasets."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "TCM-Suite": REPOSITORY_ROOT / "dataset" / "TCMsuite",
    "TCMSP": REPOSITORY_ROOT / "dataset" / "TCMSP",
    "SymMap2.0": REPOSITORY_ROOT / "dataset" / "Symmap",
    "ETCM2.0-mention10": REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_mention10",
}
RELATION_CANDIDATES = {
    "H_C": ("H_C.txt", "herb-compound.txt", "HI.txt"),
    "C_P": ("C_P.txt", "compound-protein.txt", "IT.txt"),
    "P_D": ("P_D.txt", "target-disease.txt", "TD.txt"),
}
NEGATIVE_CANDIDATES = ("ZERO_indices.txt", "zero1.txt", "zero.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit H-C support, fold size and negative capacity for a common "
            "compound C-P cold-start protocol."
        )
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-hc-support", type=float, default=0.95)
    parser.add_argument("--minimum-chcr-coverage", type=float, default=0.90)
    parser.add_argument("--minimum-chcr-edge-coverage", type=float, default=0.90)
    parser.add_argument("--minimum-compounds", type=int, default=500)
    parser.add_argument("--minimum-fold-positives", type=int, default=1000)
    parser.add_argument(
        "--output-dir", default="results/multidataset_cold_start_feasibility"
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(part, total):
    return float(part) / float(total) if total else 0.0


def resolve_file(dataset_dir, candidates, required=True):
    for filename in candidates:
        path = dataset_dir / filename
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(
            "None of %s found in %s" % (", ".join(candidates), dataset_dir)
        )
    return None


def read_edges(path):
    edges = set()
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            edges.add((parts[0], parts[1]))
    return edges, malformed


def degree_summary(degrees):
    values = sorted(degrees.values())
    if not values:
        return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
    return {
        "min": values[0],
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "max": values[-1],
    }


def counterfactual_donor_audit(compounds, cp_degree, hc_edges):
    """Audit exact-H-C-degree, herb-disjoint CHCR donor availability."""
    compound_to_herbs = defaultdict(set)
    for herb, compound in hc_edges:
        if compound in compounds:
            compound_to_herbs[compound].add(herb)

    degree_buckets = defaultdict(set)
    herb_to_compounds = defaultdict(set)
    for compound, herbs in compound_to_herbs.items():
        if not herbs:
            continue
        degree_buckets[len(herbs)].add(compound)
        for herb in herbs:
            herb_to_compounds[herb].add(compound)

    eligible = set()
    pool_sizes = {}
    for compound, herbs in compound_to_herbs.items():
        blocked = {compound}
        for herb in herbs:
            blocked.update(herb_to_compounds[herb])
        pool_size = len(degree_buckets[len(herbs)] - blocked)
        pool_sizes[compound] = pool_size
        if pool_size:
            eligible.add(compound)

    eligible_positive_edges = sum(cp_degree[compound] for compound in eligible)
    total_positive_edges = sum(cp_degree.values())
    return {
        "match_rule": "exact_hc_degree_disjoint",
        "eligible_compounds": len(eligible),
        "compound_coverage": ratio(len(eligible), len(compounds)),
        "eligible_positive_edges": eligible_positive_edges,
        "positive_edge_coverage": ratio(
            eligible_positive_edges, total_positive_edges
        ),
        "donor_pool_size": degree_summary(pool_sizes),
    }


def balanced_compound_folds(cp_degree, folds):
    if len(cp_degree) < folds:
        raise ValueError("Fewer compounds than requested folds.")
    fold_compounds = [[] for _ in range(folds)]
    fold_positives = [0 for _ in range(folds)]
    for compound, degree in sorted(
            cp_degree.items(), key=lambda item: (-item[1], item[0])):
        target = min(
            range(folds),
            key=lambda index: (
                fold_positives[index], len(fold_compounds[index]), index
            ),
        )
        fold_compounds[target].append(compound)
        fold_positives[target] += degree
    return {
        "compound_counts": [len(values) for values in fold_compounds],
        "positive_counts": fold_positives,
        "minimum_positives": min(fold_positives),
        "maximum_positives": max(fold_positives),
    }


def fixed_negative_audit(path, positives, cp_degree):
    if path is None:
        return {
            "path": None,
            "rows": 0,
            "compounds_with_enough_fixed_negatives": 0,
            "coverage": 0.0,
            "supplement_required": len(cp_degree),
        }
    edges, malformed = read_edges(path)
    negatives = edges - positives
    counts = Counter(compound for compound, _ in negatives)
    enough = sum(
        counts.get(compound, 0) >= degree
        for compound, degree in cp_degree.items()
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(edges),
        "malformed_rows": malformed,
        "compounds_with_enough_fixed_negatives": enough,
        "coverage": ratio(enough, len(cp_degree)),
        "supplement_required": len(cp_degree) - enough,
    }


def audit_dataset(name, dataset_dir, folds, thresholds):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    paths = {
        relation: resolve_file(dataset_dir, candidates)
        for relation, candidates in RELATION_CANDIDATES.items()
    }
    negative_path = resolve_file(
        dataset_dir, NEGATIVE_CANDIDATES, required=False
    )
    hc, hc_malformed = read_edges(paths["H_C"])
    cp, cp_malformed = read_edges(paths["C_P"])
    pd, pd_malformed = read_edges(paths["P_D"])
    compounds = {compound for compound, _ in cp}
    proteins = {protein for _, protein in cp}
    hc_compounds = {compound for _, compound in hc}
    pd_proteins = {protein for protein, _ in pd}
    cp_degree = Counter(compound for compound, _ in cp)
    protein_degree = Counter(protein for _, protein in cp)
    supported_compounds = compounds & hc_compounds
    supported_proteins = proteins & pd_proteins
    supported_edges = sum(
        compound in hc_compounds and protein in pd_proteins
        for compound, protein in cp
    )
    insufficient_capacity = sum(
        degree > len(proteins) - degree for degree in cp_degree.values()
    )
    fold_balance = balanced_compound_folds(cp_degree, folds)
    fixed_negatives = fixed_negative_audit(negative_path, cp, cp_degree)
    counterfactual_donors = counterfactual_donor_audit(
        compounds, cp_degree, hc
    )
    cold_start_criteria = {
        "hc_support": ratio(len(supported_compounds), len(compounds))
        >= thresholds["minimum_hc_support"],
        "compound_count": len(compounds) >= thresholds["minimum_compounds"],
        "fold_positives": fold_balance["minimum_positives"]
        >= thresholds["minimum_fold_positives"],
        "negative_capacity": insufficient_capacity == 0,
    }
    chcr_criteria = {
        "compound_coverage": counterfactual_donors["compound_coverage"]
        >= thresholds["minimum_chcr_coverage"],
        "positive_edge_coverage": counterfactual_donors[
            "positive_edge_coverage"
        ] >= thresholds["minimum_chcr_edge_coverage"],
    }
    cold_start_decision = (
        "supports_compound_cold_start_pilot"
        if all(cold_start_criteria.values())
        else "insufficient_compound_cold_start_support"
    )
    chcr_decision = (
        "supports_uniform_chcr_pilot"
        if all(chcr_criteria.values()) else "selective_chcr_required"
    )
    return {
        "name": name,
        "path": str(dataset_dir),
        "files": {
            relation: {"path": str(path), "sha256": sha256_file(path)}
            for relation, path in paths.items()
        },
        "malformed_rows": {
            "H_C": hc_malformed,
            "C_P": cp_malformed,
            "P_D": pd_malformed,
        },
        "compounds": len(compounds),
        "proteins": len(proteins),
        "positive_edges": len(cp),
        "hc_supported_compounds": len(supported_compounds),
        "hc_support_coverage": ratio(len(supported_compounds), len(compounds)),
        "pd_supported_proteins": len(supported_proteins),
        "pd_support_coverage": ratio(len(supported_proteins), len(proteins)),
        "both_supported_positive_edges": supported_edges,
        "both_supported_edge_coverage": ratio(supported_edges, len(cp)),
        "compound_cp_degree": degree_summary(cp_degree),
        "protein_cp_degree": degree_summary(protein_degree),
        "compounds_without_1to1_negative_capacity": insufficient_capacity,
        "fold_balance": fold_balance,
        "fixed_negatives": fixed_negatives,
        "counterfactual_donors": counterfactual_donors,
        "cold_start_criteria": cold_start_criteria,
        "chcr_criteria": chcr_criteria,
        "criteria": dict(cold_start_criteria, **{
            "chcr_compound_coverage": chcr_criteria["compound_coverage"],
            "chcr_positive_edge_coverage": chcr_criteria[
                "positive_edge_coverage"
            ],
        }),
        "cold_start_decision": cold_start_decision,
        "chcr_decision": chcr_decision,
        "decision": cold_start_decision,
    }


def parse_dataset_overrides(values):
    if not values:
        return dict(DEFAULT_DATASETS)
    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset must use NAME=PATH: %s" % value)
        name, path = value.split("=", 1)
        datasets[name.strip()] = Path(path).expanduser().resolve()
    return datasets


def audit_datasets(datasets, folds=5, thresholds=None):
    default_thresholds = {
        "minimum_hc_support": 0.95,
        "minimum_chcr_coverage": 0.90,
        "minimum_chcr_edge_coverage": 0.90,
        "minimum_compounds": 500,
        "minimum_fold_positives": 1000,
    }
    thresholds = dict(default_thresholds, **(thresholds or {}))
    rows = [
        audit_dataset(name, path, folds, thresholds)
        for name, path in datasets.items()
    ]
    cold_start_passed = sum(
        row["cold_start_decision"] == "supports_compound_cold_start_pilot"
        for row in rows
    )
    chcr_passed = sum(
        row["chcr_decision"] == "supports_uniform_chcr_pilot"
        for row in rows
    )
    if cold_start_passed < len(rows):
        decision = "partial_compound_cold_start_support"
    elif chcr_passed < len(rows):
        decision = "supports_multidataset_compound_cold_start_with_selective_CHCR"
    else:
        decision = "supports_multidataset_cold_start_and_uniform_CHCR"
    return {
        "audit_type": "multidataset_compound_cold_start_feasibility",
        "created_at": datetime.now().astimezone().isoformat(),
        "network_accessed": False,
        "training_steps": 0,
        "folds": folds,
        "thresholds": thresholds,
        "passed_datasets": cold_start_passed,
        "cold_start_passed_datasets": cold_start_passed,
        "uniform_chcr_passed_datasets": chcr_passed,
        "decision": decision,
        "datasets": rows,
    }


def build_markdown(report):
    lines = [
        "# 多数据集 Compound C-P Cold-Start 可行性审计",
        "",
        "- 只读取 H-C、C-P、P-D 和现有未观测样本文件。",
        "- 不访问网络、不生成 split、不训练模型。",
        "- 固定未观测样本不足时允许 Strict 协议从同一 protein 全集确定性补采。",
        "",
        "## 总体判定",
        "",
        "**%s**" % report["decision"],
        "",
        "| 数据集 | Compounds | Positives | H-C 支撑 | CHCR compound 覆盖 | CHCR 正边覆盖 | P-D protein 支撑 | 双侧支撑边 | Fold 最少正例 | Cold-start | CHCR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["datasets"]:
        fixed = row["fixed_negatives"]
        lines.append(
            "| %s | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %d | %s | %s |" % (
                row["name"], row["compounds"], row["positive_edges"],
                100.0 * row["hc_support_coverage"],
                100.0 * row["counterfactual_donors"]["compound_coverage"],
                100.0 * row["counterfactual_donors"][
                    "positive_edge_coverage"
                ],
                100.0 * row["pd_support_coverage"],
                100.0 * row["both_supported_edge_coverage"],
                row["fold_balance"]["minimum_positives"],
                row["cold_start_decision"], row["chcr_decision"],
            )
        )
    lines.extend([
        "",
        "## 预注册门槛",
        "",
        "- C-P compound 的 H-C 支撑率 >= %.0f%%。" % (
            100.0 * report["thresholds"]["minimum_hc_support"]),
        "- 至少 %.0f%% 的 C-P compound 存在同 H-C 度数且药材集合不相交的 CHCR 供体。" % (
            100.0 * report["thresholds"]["minimum_chcr_coverage"]),
        "- CHCR 供体可用 compound 覆盖的 C-P 正边比例 >= %.0f%%。" % (
            100.0 * report["thresholds"]["minimum_chcr_edge_coverage"]),
        "- 至少 %d 个 C-P compound。" % (
            report["thresholds"]["minimum_compounds"]),
        "- 贪心平衡后的每个测试 fold 至少 %d 条正例。" % (
            report["thresholds"]["minimum_fold_positives"]),
        "- 每个 compound 在全 protein 候选空间中都可构造 1:1 未观测 pair。",
        "",
        "固定 ZERO 文件覆盖不是硬门槛；补采必须排除全部已知 C-P 正边，并由 split seed 确定性生成和写入 manifest。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    thresholds = {
        "minimum_hc_support": args.minimum_hc_support,
        "minimum_chcr_coverage": args.minimum_chcr_coverage,
        "minimum_chcr_edge_coverage": args.minimum_chcr_edge_coverage,
        "minimum_compounds": args.minimum_compounds,
        "minimum_fold_positives": args.minimum_fold_positives,
    }
    report = audit_datasets(
        parse_dataset_overrides(args.dataset),
        folds=args.folds,
        thresholds=thresholds,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = build_markdown(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Results written to: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
