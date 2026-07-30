#!/usr/bin/env python3
"""Audit target-cold and double-cold feasibility without training a model."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "TCM-Suite": REPOSITORY_ROOT / "dataset" / "TCMsuite",
    "TCMSP": REPOSITORY_ROOT / "dataset" / "TCMSP",
    "SymMap2.0": REPOSITORY_ROOT / "dataset" / "Symmap",
    "ETCM2.0-mention10": (
        REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_mention10"
    ),
}
RELATION_CANDIDATES = {
    "H_C": ("H_C.txt", "herb-compound.txt", "HI.txt"),
    "C_P": ("C_P.txt", "compound-protein.txt", "IT.txt"),
    "P_D": ("P_D.txt", "target-disease.txt", "TD.txt"),
}
DEFAULT_THRESHOLDS = {
    "minimum_supported_edge_coverage": 0.50,
    "minimum_supported_targets": 100,
    "minimum_target_fold_positives": 500,
    "minimum_target_state_purity": 0.90,
    "minimum_double_cell_positives": 100,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether four HDCTI datasets can support target-cold and "
            "double-cold evaluation using H-C/P-D side contexts."
        )
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--minimum-supported-edge-coverage",
        type=float,
        default=DEFAULT_THRESHOLDS["minimum_supported_edge_coverage"],
    )
    parser.add_argument(
        "--minimum-supported-targets",
        type=int,
        default=DEFAULT_THRESHOLDS["minimum_supported_targets"],
    )
    parser.add_argument(
        "--minimum-target-fold-positives",
        type=int,
        default=DEFAULT_THRESHOLDS["minimum_target_fold_positives"],
    )
    parser.add_argument(
        "--minimum-target-state-purity",
        type=float,
        default=DEFAULT_THRESHOLDS["minimum_target_state_purity"],
    )
    parser.add_argument(
        "--minimum-double-cell-positives",
        type=int,
        default=DEFAULT_THRESHOLDS["minimum_double_cell_positives"],
    )
    parser.add_argument(
        "--output-dir",
        default="results/support_complete_cold_start_feasibility",
    )
    parser.add_argument(
        "--documentation",
        default="docs/SUPPORT_COMPLETE_COLD_START_FEASIBILITY.md",
    )
    return parser.parse_args()


def ratio(part, total):
    return float(part) / float(total) if total else 0.0


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed, entity_type, entity_id):
    value = "%d|%s|%s" % (int(seed), entity_type, str(entity_id))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_file(dataset_dir, candidates):
    for filename in candidates:
        path = dataset_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of %s found in %s" % (", ".join(candidates), dataset_dir)
    )


def read_edges(path):
    edges = set()
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            edges.add((str(parts[0]), str(parts[1])))
    return edges, malformed


def distribution_summary(values):
    values = sorted(values)
    if not values:
        return {
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0,
        }
    return {
        "min": values[0],
        "median": float(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "max": values[-1],
    }


def balanced_entity_folds(entity_degrees, folds, seed, entity_type):
    """Greedily balance positive degrees with deterministic hashed tie breaks."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(entity_degrees) < folds:
        raise ValueError(
            "%s has fewer entities (%d) than folds (%d)"
            % (entity_type, len(entity_degrees), folds)
        )
    groups = [set() for _ in range(folds)]
    loads = [0 for _ in range(folds)]
    entities = sorted(
        entity_degrees,
        key=lambda entity: (
            -entity_degrees[entity],
            stable_hash(seed, entity_type, entity),
            str(entity),
        ),
    )
    for entity in entities:
        target = min(
            range(folds),
            key=lambda index: (loads[index], len(groups[index]), index),
        )
        groups[target].add(entity)
        loads[target] += entity_degrees[entity]
    return groups, loads


def target_cold_audit(cp_edges, supported_edges, supported_compounds,
                      supported_proteins, folds, seed):
    target_degrees = Counter(protein for _, protein in supported_edges)
    protein_groups, assigned_loads = balanced_entity_folds(
        target_degrees, folds, seed, "protein"
    )
    fold_rows = []
    all_compounds = {compound for compound, _ in cp_edges}
    for fold_index, heldout_proteins in enumerate(protein_groups):
        training_edges = {
            edge for edge in cp_edges if edge[1] not in heldout_proteins
        }
        training_compounds = {
            compound for compound, _ in training_edges
        }
        training_proteins = {
            protein for _, protein in training_edges
        }
        raw_test_edges = {
            edge for edge in supported_edges
            if edge[1] in heldout_proteins
        }
        state_valid_edges = {
            edge for edge in raw_test_edges
            if edge[0] in training_compounds
        }
        candidate_compounds = supported_compounds & training_compounds
        known_by_target = Counter(
            protein
            for compound, protein in cp_edges
            if compound in candidate_compounds
            and protein in heldout_proteins
        )
        positives_by_target = Counter(
            protein for _, protein in state_valid_edges
        )
        negative_capacity = sum(
            len(candidate_compounds) - known_by_target[protein]
            for protein in heldout_proteins
        )
        insufficient_targets = sum(
            len(candidate_compounds) - known_by_target[protein]
            < positives_by_target[protein]
            for protein in heldout_proteins
        )
        fold_rows.append({
            "fold": fold_index + 1,
            "heldout_proteins": len(heldout_proteins),
            "assigned_supported_positives": assigned_loads[fold_index],
            "training_positives": len(training_edges),
            "raw_supported_test_positives": len(raw_test_edges),
            "state_valid_test_positives": len(state_valid_edges),
            "state_purity": ratio(
                len(state_valid_edges), len(raw_test_edges)
            ),
            "test_compounds": len({
                compound for compound, _ in state_valid_edges
            }),
            "candidate_compounds": len(candidate_compounds),
            "negative_candidate_capacity": negative_capacity,
            "targets_without_1to1_negative_capacity": insufficient_targets,
            "train_test_protein_overlap": len(
                heldout_proteins & training_proteins
            ),
        })
    return {
        "protocol": "warm_compound_cold_target",
        "eligible_proteins": len(supported_proteins),
        "folds": fold_rows,
        "fold_positive_summary": distribution_summary([
            row["state_valid_test_positives"] for row in fold_rows
        ]),
        "state_purity_summary": distribution_summary([
            row["state_purity"] for row in fold_rows
        ]),
        "all_folds_entity_disjoint": all(
            row["train_test_protein_overlap"] == 0 for row in fold_rows
        ),
        "all_folds_have_1to1_negative_capacity": all(
            row["negative_candidate_capacity"]
            >= row["state_valid_test_positives"]
            for row in fold_rows
        ),
        "all_folds_have_per_target_1to1_negative_capacity": all(
            row["targets_without_1to1_negative_capacity"] == 0
            for row in fold_rows
        ),
    }


def double_cold_audit(cp_edges, supported_edges, supported_compounds,
                      supported_proteins, folds, seed):
    compound_degrees = Counter(compound for compound, _ in supported_edges)
    protein_degrees = Counter(protein for _, protein in supported_edges)
    compound_groups, compound_loads = balanced_entity_folds(
        compound_degrees, folds, seed, "compound"
    )
    protein_groups, protein_loads = balanced_entity_folds(
        protein_degrees, folds, seed, "protein"
    )
    cells = []
    matrix = []
    for compound_index, heldout_compounds in enumerate(compound_groups):
        matrix_row = []
        for protein_index, heldout_proteins in enumerate(protein_groups):
            training_edges = {
                edge for edge in cp_edges
                if edge[0] not in heldout_compounds
                and edge[1] not in heldout_proteins
            }
            test_edges = {
                edge for edge in supported_edges
                if edge[0] in heldout_compounds
                and edge[1] in heldout_proteins
            }
            known_block_edges = sum(
                compound in heldout_compounds
                and protein in heldout_proteins
                for compound, protein in cp_edges
            )
            candidate_pairs = (
                len(heldout_compounds) * len(heldout_proteins)
            )
            negative_capacity = candidate_pairs - known_block_edges
            positives_by_target = Counter(
                protein for _, protein in test_edges
            )
            known_by_target = Counter(
                protein
                for compound, protein in cp_edges
                if compound in heldout_compounds
                and protein in heldout_proteins
            )
            insufficient_targets = sum(
                len(heldout_compounds) - known_by_target[protein]
                < positives_by_target[protein]
                for protein in heldout_proteins
            )
            training_compounds = {
                compound for compound, _ in training_edges
            }
            training_proteins = {
                protein for _, protein in training_edges
            }
            row = {
                "compound_fold": compound_index + 1,
                "protein_fold": protein_index + 1,
                "heldout_compounds": len(heldout_compounds),
                "heldout_proteins": len(heldout_proteins),
                "training_positives": len(training_edges),
                "test_positives": len(test_edges),
                "candidate_pairs": candidate_pairs,
                "negative_candidate_capacity": negative_capacity,
                "targets_without_1to1_negative_capacity": (
                    insufficient_targets
                ),
                "train_test_compound_overlap": len(
                    heldout_compounds & training_compounds
                ),
                "train_test_protein_overlap": len(
                    heldout_proteins & training_proteins
                ),
            }
            cells.append(row)
            matrix_row.append(len(test_edges))
        matrix.append(matrix_row)
    return {
        "protocol": "cold_compound_cold_target_cartesian_grid",
        "grid_shape": [folds, folds],
        "evaluation_cells": folds * folds,
        "eligible_compounds": len(supported_compounds),
        "eligible_proteins": len(supported_proteins),
        "compound_group_assigned_loads": compound_loads,
        "protein_group_assigned_loads": protein_loads,
        "positive_matrix": matrix,
        "cells": cells,
        "cell_positive_summary": distribution_summary([
            row["test_positives"] for row in cells
        ]),
        "covered_supported_positives": sum(
            row["test_positives"] for row in cells
        ),
        "expected_supported_positives": len(supported_edges),
        "positive_coverage": ratio(
            sum(row["test_positives"] for row in cells),
            len(supported_edges),
        ),
        "empty_cells": sum(row["test_positives"] == 0 for row in cells),
        "all_cells_entity_disjoint": all(
            row["train_test_compound_overlap"] == 0
            and row["train_test_protein_overlap"] == 0
            for row in cells
        ),
        "all_cells_have_1to1_negative_capacity": all(
            row["negative_candidate_capacity"] >= row["test_positives"]
            for row in cells
        ),
        "all_cells_have_per_target_1to1_negative_capacity": all(
            row["targets_without_1to1_negative_capacity"] == 0
            for row in cells
        ),
    }


def audit_dataset(name, dataset_dir, folds=5, seed=2026,
                  thresholds=None):
    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    paths = {
        relation: resolve_file(dataset_dir, candidates)
        for relation, candidates in RELATION_CANDIDATES.items()
    }
    hc_edges, hc_malformed = read_edges(paths["H_C"])
    cp_edges, cp_malformed = read_edges(paths["C_P"])
    pd_edges, pd_malformed = read_edges(paths["P_D"])
    cp_compounds = {compound for compound, _ in cp_edges}
    cp_proteins = {protein for _, protein in cp_edges}
    hc_compounds = {compound for _, compound in hc_edges}
    pd_proteins = {protein for protein, _ in pd_edges}
    supported_compounds = cp_compounds & hc_compounds
    supported_proteins = cp_proteins & pd_proteins
    supported_edges = {
        (compound, protein)
        for compound, protein in cp_edges
        if compound in supported_compounds
        and protein in supported_proteins
    }
    target_cold = target_cold_audit(
        cp_edges, supported_edges, supported_compounds,
        supported_proteins, folds, seed
    )
    double_cold = double_cold_audit(
        cp_edges, supported_edges, supported_compounds,
        supported_proteins, folds, seed
    )
    criteria = {
        "supported_edge_coverage": ratio(
            len(supported_edges), len(cp_edges)
        ) >= thresholds["minimum_supported_edge_coverage"],
        "supported_target_count": (
            len(supported_proteins)
            >= thresholds["minimum_supported_targets"]
        ),
        "target_fold_positives": (
            target_cold["fold_positive_summary"]["min"]
            >= thresholds["minimum_target_fold_positives"]
        ),
        "target_state_purity": (
            target_cold["state_purity_summary"]["min"]
            >= thresholds["minimum_target_state_purity"]
        ),
        "target_negative_capacity": (
            target_cold["all_folds_have_1to1_negative_capacity"]
        ),
        "target_entity_disjointness": (
            target_cold["all_folds_entity_disjoint"]
        ),
        "double_cell_positives": (
            double_cold["cell_positive_summary"]["min"]
            >= thresholds["minimum_double_cell_positives"]
        ),
        "double_negative_capacity": (
            double_cold["all_cells_have_1to1_negative_capacity"]
        ),
        "double_entity_disjointness": (
            double_cold["all_cells_entity_disjoint"]
        ),
        "double_positive_coverage": (
            double_cold["positive_coverage"] == 1.0
        ),
    }
    return {
        "name": name,
        "path": str(dataset_dir),
        "files": {
            relation: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for relation, path in paths.items()
        },
        "malformed_rows": {
            "H_C": hc_malformed,
            "C_P": cp_malformed,
            "P_D": pd_malformed,
        },
        "compounds": len(cp_compounds),
        "proteins": len(cp_proteins),
        "positive_edges": len(cp_edges),
        "hc_supported_compounds": len(supported_compounds),
        "hc_compound_coverage": ratio(
            len(supported_compounds), len(cp_compounds)
        ),
        "pd_supported_proteins": len(supported_proteins),
        "pd_protein_coverage": ratio(
            len(supported_proteins), len(cp_proteins)
        ),
        "both_supported_positive_edges": len(supported_edges),
        "both_supported_edge_coverage": ratio(
            len(supported_edges), len(cp_edges)
        ),
        "target_cold": target_cold,
        "double_cold": double_cold,
        "criteria": criteria,
        "decision": (
            "supports_state_complete_cold_start_pilot"
            if all(criteria.values())
            else "insufficient_state_complete_cold_start_support"
        ),
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


def audit_datasets(datasets, folds=5, seed=2026, thresholds=None):
    thresholds = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    rows = [
        audit_dataset(
            name, path, folds=folds, seed=seed, thresholds=thresholds
        )
        for name, path in datasets.items()
    ]
    passed = sum(
        row["decision"] == "supports_state_complete_cold_start_pilot"
        for row in rows
    )
    return {
        "audit_type": "support_complete_cold_start_feasibility",
        "created_at": datetime.now().astimezone().isoformat(),
        "network_accessed": False,
        "training_steps": 0,
        "seed": int(seed),
        "folds": int(folds),
        "thresholds": thresholds,
        "passed_datasets": passed,
        "total_datasets": len(rows),
        "decision": (
            "GO_state_complete_four_dataset_pilot"
            if passed == len(rows)
            else "NO_GO_state_complete_four_dataset_pilot"
        ),
        "datasets": rows,
    }


def format_percent(value):
    return "%.2f%%" % (100.0 * value)


def build_markdown(report):
    lines = [
        "# 支持状态完备的冷启动可行性审计",
        "",
        "## 审计目标",
        "",
        "本审计只读取四库的 H-C、C-P 和 P-D 关系，不访问网络、不生成训练权重、"
        "不运行模型。目标是判断统一模型能否在以下两个尚未实现的状态中获得足够"
        "且无实体泄漏的评价样本：",
        "",
        "1. `warm compound / cold target`：成分在训练 C-P 中可见，靶点不可见，"
        "分别由成分表示和 P-D 疾病上下文提供信息；",
        "2. `cold compound / cold target`：成分和靶点均不出现在训练 C-P 中，"
        "只能由 H-C 药材上下文与 P-D 疾病上下文提供信息。",
        "",
        "工作名 `SCCI` 仅表示待验证的支持状态完备交互框架，不是已确认的论文方法名。",
        "",
        "## 总体判定",
        "",
        "**%s（%d/%d 个数据集通过预注册门槛）**" % (
            report["decision"],
            report["passed_datasets"],
            report["total_datasets"],
        ),
        "",
        "| 数据集 | C-P 正边 | H-C 成分覆盖 | P-D 靶点覆盖 | 双侧支撑正边 | Target-cold 每折最少正例 | 最低状态纯度 | Double-cold 每格最少正例 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["datasets"]:
        lines.append(
            "| %s | %d | %s | %s | %s (%d) | %d | %s | %d | %s |"
            % (
                row["name"],
                row["positive_edges"],
                format_percent(row["hc_compound_coverage"]),
                format_percent(row["pd_protein_coverage"]),
                format_percent(row["both_supported_edge_coverage"]),
                row["both_supported_positive_edges"],
                row["target_cold"]["fold_positive_summary"]["min"],
                format_percent(
                    row["target_cold"]["state_purity_summary"]["min"]
                ),
                row["double_cold"]["cell_positive_summary"]["min"],
                row["decision"],
            )
        )
    lines.extend([
        "",
        "## 协议定义",
        "",
        "### Target-cold",
        "",
        "- 仅将具有 P-D 侧信息的 C-P protein 分为 %d 组。" % (
            report["folds"]
        ),
        "- 每折移除测试 protein 的全部训练 C-P 边，保证 train/test protein 交集为 0。",
        "- 仅保留在该折训练 C-P 中仍出现的 compound，确保评价状态确实是"
        " `warm compound / cold target`，而不是混入双冷启动。",
        "- 未观测候选从该折仍可见且有 H-C 支撑的 compound 与测试 protein"
        " 笛卡尔积中产生，并排除全部已知 C-P 正边。",
        "",
        "### Double-cold",
        "",
        "- 将有 H-C 支撑的 compound 和有 P-D 支撑的 protein 分别确定性划为"
        " %d 组。" % report["folds"],
        "- 审计全部 `%d×%d=%d` 个 compound-group × protein-group 测试格；"
        "每格训练集同时排除对应 compound 组和 protein 组。" % (
            report["folds"], report["folds"],
            report["folds"] * report["folds"],
        ),
        "- 使用完整笛卡尔网格而非只取五个对角格，因此四库双侧支撑正边均恰好"
        "被评价一次；代价是正式实验每个数据集需要 25 个训练单元。",
        "",
        "## 各库明细",
        "",
    ])
    for row in report["datasets"]:
        target = row["target_cold"]
        double = row["double_cold"]
        lines.extend([
            "### %s" % row["name"],
            "",
            "- 有 P-D 支撑靶点：`%d/%d`（%s）；双侧支撑正边：`%d/%d`（%s）。"
            % (
                row["pd_supported_proteins"], row["proteins"],
                format_percent(row["pd_protein_coverage"]),
                row["both_supported_positive_edges"], row["positive_edges"],
                format_percent(row["both_supported_edge_coverage"]),
            ),
            "- Target-cold 每折状态有效正例：`%d / %.1f / %d`"
            "（min / mean / max），最低状态纯度 `%s`。"
            % (
                target["fold_positive_summary"]["min"],
                target["fold_positive_summary"]["mean"],
                target["fold_positive_summary"]["max"],
                format_percent(target["state_purity_summary"]["min"]),
            ),
            "- Double-cold 25 格正例：`%d / %.1f / %d`"
            "（min / mean / max），覆盖 `%d/%d` 条双侧支撑正边。"
            % (
                double["cell_positive_summary"]["min"],
                double["cell_positive_summary"]["mean"],
                double["cell_positive_summary"]["max"],
                double["covered_supported_positives"],
                double["expected_supported_positives"],
            ),
            "- 总体 1:1 未观测候选容量：Target-cold `%s`，Double-cold `%s`；"
            "逐 target 1:1 诊断：Target-cold `%s`，Double-cold `%s`；"
            "实体隔离：Target-cold `%s`，Double-cold `%s`。"
            % (
                "通过" if target[
                    "all_folds_have_1to1_negative_capacity"
                ] else "不通过",
                "通过" if double[
                    "all_cells_have_1to1_negative_capacity"
                ] else "不通过",
                "通过" if target[
                    "all_folds_have_per_target_1to1_negative_capacity"
                ] else "存在高连接 target",
                "通过" if double[
                    "all_cells_have_per_target_1to1_negative_capacity"
                ] else "存在高连接 target",
                "通过" if target[
                    "all_folds_entity_disjoint"
                ] else "不通过",
                "通过" if double[
                    "all_cells_entity_disjoint"
                ] else "不通过",
            ),
            "",
            "Double-cold 正例矩阵（行=compound 组，列=protein 组）：",
            "",
            "| C组\\P组 | %s |" % " | ".join(
                "P%d" % (index + 1) for index in range(report["folds"])
            ),
            "|---|%s|" % "|".join(
                "---:" for _ in range(report["folds"])
            ),
        ])
        for index, matrix_row in enumerate(double["positive_matrix"]):
            lines.append(
                "| C%d | %s |" % (
                    index + 1,
                    " | ".join(str(value) for value in matrix_row),
                )
            )
        lines.append("")
    thresholds = report["thresholds"]
    lines.extend([
        "## 预注册门槛",
        "",
        "- 双侧支撑正边覆盖率 >= %s。" % format_percent(
            thresholds["minimum_supported_edge_coverage"]
        ),
        "- 至少 %d 个具有 P-D 支撑的 C-P protein。" % (
            thresholds["minimum_supported_targets"]
        ),
        "- Target-cold 每折至少 %d 条状态有效正例，最低状态纯度 >= %s。"
        % (
            thresholds["minimum_target_fold_positives"],
            format_percent(thresholds["minimum_target_state_purity"]),
        ),
        "- Double-cold 25 个测试格中每格至少 %d 条正例。"
        % thresholds["minimum_double_cell_positives"],
        "- 每折/每格 train-test 实体严格不相交，且具备总体 1:1"
        " 未观测候选采样容量。逐 target 1:1 仅作为度数偏倚诊断，"
        "不作为全局 AUC/AUPR 或全候选排名的硬门槛。",
        "",
        "## 解释边界",
        "",
        "1. 本报告证明的是数据与协议可行性，不证明 SCCI 会提高预测性能。",
        "2. TCM-Suite 与 TCMSP 的 P-D 实体覆盖较低；正式论文必须同时报告"
        "全 C-P 正边覆盖率，不能只呈现过滤后的高分结果。",
        "3. TCMSP 存在一个高连接 target，无法在每个 compound 小格内单独"
        "配出等量未观测边，但各格总体负候选容量均远高于正例。正式评价应采用"
        "全候选排名或全格统一采样，不应删除该 target。",
        "4. Double-cold 的 25 格完整评价成本较高。可以先固定一个预注册格做"
        "一折 Gate 1，但最终结论不能只依赖挑选后的格。",
        "5. 后续实现必须使用同一 checkpoint 和由训练 C-P 支持度决定的路由，"
        "不能按数据库或测试结果选择分支。",
        "",
        "## 下一步",
        "",
        "若总体判定为 `GO_state_complete_four_dataset_pilot`，下一阶段只实现"
        " target-cold 与 double-cold 的固定 split manifest，不立即修改模型；"
        "先用现有 Strict/Hctx-P 检查协议和候选采样，再实现 C-Dctx 与"
        " Hctx-Dctx 两个缺失分支。若任一数据集未通过，则停止统一四状态主张，"
        "改为将该协议限制为侧信息覆盖充分的数据子集并明确降级为补充实验。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    thresholds = {
        "minimum_supported_edge_coverage": (
            args.minimum_supported_edge_coverage
        ),
        "minimum_supported_targets": args.minimum_supported_targets,
        "minimum_target_fold_positives": (
            args.minimum_target_fold_positives
        ),
        "minimum_target_state_purity": args.minimum_target_state_purity,
        "minimum_double_cell_positives": (
            args.minimum_double_cell_positives
        ),
    }
    report = audit_datasets(
        parse_dataset_overrides(args.dataset),
        folds=args.folds,
        seed=args.seed,
        thresholds=thresholds,
    )
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
    documentation_path.parent.mkdir(parents=True, exist_ok=True)
    documentation_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Results written to: %s" % output_dir)
    print("Documentation written to: %s" % documentation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
