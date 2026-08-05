#!/usr/bin/env python3
"""Audit self-excluded dual-hypergraph prototype transfer on four datasets.

The audit is deliberately model-free.  For every frozen support-complete unit it
creates a new four-state inner validation split, builds all transfer statistics
from the inner-training positives, and never reads the frozen outer test files.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from util.support_complete_split import (  # noqa: E402
    build_four_state_inner_validation,
    load_four_state_support_artifact,
    read_pairs,
    records_sha256,
    sha256_file,
)


STATE_NAMES = ("warm_warm", "cold_warm", "warm_cold", "cold_cold")
CHANNEL_NAMES = (
    "herb_to_target",
    "disease_to_compound",
    "dual_transfer",
    "fixed_fusion",
)
EXPECTED_CHANNEL = {
    "warm_warm": "fixed_fusion",
    "cold_warm": "herb_to_target",
    "warm_cold": "disease_to_compound",
    "cold_cold": "dual_transfer",
}
DEFAULT_MANIFESTS = {
    "TCM-Suite": REPOSITORY_ROOT / "dataset" / "TCMsuite" / "splits"
    / "support_complete_seed_2026_k5" / "four_state_seed_2026_c0_p0"
    / "manifest.json",
    "TCMSP": REPOSITORY_ROOT / "dataset" / "TCMSP" / "splits"
    / "support_complete_seed_2026_k5" / "four_state_seed_2026_c0_p0"
    / "manifest.json",
    "SymMap2.0": REPOSITORY_ROOT / "dataset" / "Symmap" / "splits"
    / "support_complete_seed_2026_k5" / "four_state_seed_2026_c0_p0"
    / "manifest.json",
    "ETCM2.0-mention10": REPOSITORY_ROOT / "dataset"
    / "ETCM2.0_core_mention10" / "splits"
    / "support_complete_seed_2026_k5" / "four_state_seed_2026_c0_p0"
    / "manifest.json",
}
DEFAULT_THRESHOLDS = {
    "minimum_cross_dataset_expected_macro_AUPR": 0.60,
    "minimum_dataset_expected_macro_AUPR": 0.55,
    "minimum_passing_datasets": 3,
    "minimum_cold_state_AUPR": 0.50,
    "minimum_passing_cold_state_cells": 10,
    "minimum_expected_channel_coverage": 0.30,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether self-excluded H-C/P-D prototype transfer contains "
            "fold-safe candidate-level signal before changing HDCTI."
        )
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=MANIFEST"
    )
    parser.add_argument("--inner-ratio", type=float, default=0.20)
    parser.add_argument("--inner-seed", type=int, default=42026)
    parser.add_argument(
        "--output-dir",
        default="results/loco_dual_hypergraph_transfer_audit",
    )
    parser.add_argument(
        "--documentation",
        default="docs/LOCO_DUAL_HYPERGRAPH_TRANSFER_AUDIT.md",
    )
    return parser.parse_args()


def parse_datasets(values):
    if not values:
        return dict(DEFAULT_MANIFESTS)
    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset must use NAME=MANIFEST: %s" % value)
        name, path = value.split("=", 1)
        datasets[name.strip()] = Path(path).expanduser().resolve()
    return datasets


def _binary_incidence(edges, row_map, column_map, reverse=False):
    rows = []
    columns = []
    for left_id, right_id in edges:
        row_id, column_id = (
            (right_id, left_id) if reverse else (left_id, right_id)
        )
        if row_id not in row_map or column_id not in column_map:
            continue
        rows.append(row_map[row_id])
        columns.append(column_map[column_id])
    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(row_map), len(column_map)),
        dtype=np.float32,
    ).tocsr()
    if matrix.nnz:
        matrix.data[:] = 1.0
    return matrix


def _row_normalize(matrix, norm="sum"):
    matrix = matrix.tocsr().astype(np.float32)
    if norm == "l2":
        values = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    elif norm == "sum":
        values = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        raise ValueError("Unknown normalization: %s" % norm)
    inverse = np.zeros_like(values, dtype=np.float32)
    nonzero = values > 0
    inverse[nonzero] = 1.0 / values[nonzero]
    return sparse.diags(inverse, format="csr").dot(matrix).tocsr()


def _column_normalize(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    values = np.asarray(matrix.sum(axis=0)).ravel()
    inverse = np.zeros_like(values, dtype=np.float32)
    nonzero = values > 0
    inverse[nonzero] = 1.0 / values[nonzero]
    return matrix.dot(sparse.diags(inverse, format="csr")).tocsr()


def _cross_transition(evaluation_incidence, training_incidence,
                      evaluation_ids, training_map):
    evaluation_normalized = _row_normalize(
        evaluation_incidence, norm="l2"
    )
    training_normalized = _row_normalize(training_incidence, norm="l2")
    similarity = evaluation_normalized.dot(
        training_normalized.transpose()
    ).tolil()
    for evaluation_index, entity_id in enumerate(evaluation_ids):
        training_index = training_map.get(entity_id)
        if training_index is not None:
            similarity[evaluation_index, training_index] = 0.0
    similarity = similarity.tocsr()
    similarity.eliminate_zeros()
    return _row_normalize(similarity, norm="sum")


def build_transfer_matrices(hc_edges, pd_edges, training_records,
                            evaluation_records):
    """Score fixed LOCO channels without using labels from evaluation rows."""
    positive_edges = {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id, label in training_records
        if float(label) > 0
    }
    training_compounds = sorted({edge[0] for edge in positive_edges})
    training_proteins = sorted({edge[1] for edge in positive_edges})
    evaluation_compounds = sorted({str(row[0]) for row in evaluation_records})
    evaluation_proteins = sorted({str(row[1]) for row in evaluation_records})
    herbs = {str(herb_id) for herb_id, _ in hc_edges}
    diseases = {str(disease_id) for _, disease_id in pd_edges}
    training_compound_map = {
        value: index for index, value in enumerate(training_compounds)
    }
    training_protein_map = {
        value: index for index, value in enumerate(training_proteins)
    }
    evaluation_compound_map = {
        value: index for index, value in enumerate(evaluation_compounds)
    }
    evaluation_protein_map = {
        value: index for index, value in enumerate(evaluation_proteins)
    }
    herb_map = {value: index for index, value in enumerate(sorted(herbs))}
    disease_map = {value: index for index, value in enumerate(sorted(diseases))}

    training_compound_herb = _binary_incidence(
        hc_edges, training_compound_map, herb_map, reverse=True
    )
    evaluation_compound_herb = _binary_incidence(
        hc_edges, evaluation_compound_map, herb_map, reverse=True
    )
    training_protein_disease = _binary_incidence(
        pd_edges, training_protein_map, disease_map, reverse=False
    )
    evaluation_protein_disease = _binary_incidence(
        pd_edges, evaluation_protein_map, disease_map, reverse=False
    )
    compound_transition = _cross_transition(
        evaluation_compound_herb,
        training_compound_herb,
        evaluation_compounds,
        training_compound_map,
    )
    protein_transition = _cross_transition(
        evaluation_protein_disease,
        training_protein_disease,
        evaluation_proteins,
        training_protein_map,
    )
    cp = _binary_incidence(
        positive_edges,
        training_compound_map,
        training_protein_map,
        reverse=False,
    )
    cp_by_compound = _row_normalize(cp, norm="sum")
    cp_by_protein = _column_normalize(cp)

    herb_prototypes = compound_transition.dot(cp_by_compound).tocsr()
    herb_prototypes_reverse = compound_transition.dot(cp_by_protein).tocsr()
    record_compound_indices = np.fromiter(
        (evaluation_compound_map[str(row[0])] for row in evaluation_records),
        dtype=np.int64,
        count=len(evaluation_records),
    )
    record_protein_indices = np.fromiter(
        (evaluation_protein_map[str(row[1])] for row in evaluation_records),
        dtype=np.int64,
        count=len(evaluation_records),
    )

    herb_to_target = np.zeros(len(evaluation_records), dtype=np.float64)
    exact_protein_indices = np.asarray([
        training_protein_map.get(str(row[1]), -1)
        for row in evaluation_records
    ], dtype=np.int64)
    valid_protein = exact_protein_indices >= 0
    if np.any(valid_protein):
        herb_to_target[valid_protein] = np.asarray(
            herb_prototypes[
                record_compound_indices[valid_protein],
                exact_protein_indices[valid_protein],
            ]
        ).reshape(-1)

    disease_to_compound = np.zeros(len(evaluation_records), dtype=np.float64)
    exact_compound_indices = np.asarray([
        training_compound_map.get(str(row[0]), -1)
        for row in evaluation_records
    ], dtype=np.int64)
    valid_compound = exact_compound_indices >= 0
    if np.any(valid_compound):
        left = cp_by_protein[exact_compound_indices[valid_compound]]
        right = protein_transition[record_protein_indices[valid_compound]]
        disease_to_compound[valid_compound] = np.asarray(
            left.multiply(right).sum(axis=1)
        ).reshape(-1)

    protein_rows = protein_transition[record_protein_indices]
    forward_rows = herb_prototypes[record_compound_indices]
    reverse_rows = herb_prototypes_reverse[record_compound_indices]
    dual_forward = np.asarray(
        forward_rows.multiply(protein_rows).sum(axis=1)
    ).reshape(-1)
    dual_reverse = np.asarray(
        reverse_rows.multiply(protein_rows).sum(axis=1)
    ).reshape(-1)
    dual_transfer = 0.5 * (dual_forward + dual_reverse)
    fixed_fusion = (
        herb_to_target + disease_to_compound + dual_transfer
    ) / 3.0

    return {
        "scores": {
            "herb_to_target": herb_to_target,
            "disease_to_compound": disease_to_compound,
            "dual_transfer": dual_transfer,
            "fixed_fusion": fixed_fusion,
        },
        "statistics": {
            "training_compounds": len(training_compound_map),
            "training_proteins": len(training_protein_map),
            "evaluation_compounds": len(evaluation_compound_map),
            "evaluation_proteins": len(evaluation_protein_map),
            "herbs": len(herb_map),
            "diseases": len(disease_map),
            "training_positive_edges": len(positive_edges),
            "compound_context_coverage": float(
                np.mean(
                    np.asarray(evaluation_compound_herb.sum(axis=1)).ravel()
                    > 0
                )
            ),
            "protein_context_coverage": float(
                np.mean(
                    np.asarray(evaluation_protein_disease.sum(axis=1)).ravel()
                    > 0
                )
            ),
            "compound_transfer_rows": int(
                np.sum(np.asarray(compound_transition.sum(axis=1)).ravel() > 0)
            ),
            "protein_transfer_rows": int(
                np.sum(np.asarray(protein_transition.sum(axis=1)).ravel() > 0)
            ),
        },
    }


def binary_metrics(records, scores):
    labels = np.asarray(
        [int(float(row[2]) > 0) for row in records], dtype=np.int32
    )
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return {
        "AUPR": float(average_precision_score(labels, scores)),
        "AUC": float(roc_auc_score(labels, scores)),
        "coverage": float(np.mean(scores != 0.0)),
        "positive_mean": float(np.mean(positive)),
        "negative_mean": float(np.mean(negative)),
        "positive_minus_negative": float(np.mean(positive) - np.mean(negative)),
        "records": int(len(records)),
    }


def audit_dataset(name, manifest_path, inner_ratio, inner_seed):
    manifest_path = Path(manifest_path).expanduser().resolve()
    outer_training, _, outer_metadata = load_four_state_support_artifact(
        manifest_path
    )
    outer_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = Path(
        outer_manifest["source_manifest"]["path"]
    ).resolve()
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_paths = {
        relation: Path(source_manifest["sources"][relation]["path"]).resolve()
        for relation in ("H_C", "C_P", "P_D")
    }
    all_positive_pairs = read_pairs(source_paths["C_P"])
    inner_training, states, inner_metadata = build_four_state_inner_validation(
        outer_training,
        all_positive_pairs,
        ratio=inner_ratio,
        seed=inner_seed,
        unit_key="%s|loco_dhpt" % outer_metadata["unit_key"],
    )
    evaluation_records = [row for state in STATE_NAMES for row in states[state]]
    transfer = build_transfer_matrices(
        read_pairs(source_paths["H_C"]),
        read_pairs(source_paths["P_D"]),
        inner_training,
        evaluation_records,
    )
    state_results = {}
    offset = 0
    for state in STATE_NAMES:
        records = states[state]
        next_offset = offset + len(records)
        channel_results = {}
        for channel in CHANNEL_NAMES:
            scores = transfer["scores"][channel][offset:next_offset]
            channel_results[channel] = binary_metrics(records, scores)
        expected = EXPECTED_CHANNEL[state]
        state_results[state] = {
            "expected_channel": expected,
            "channels": channel_results,
            "expected_metrics": channel_results[expected],
            "records_sha256": records_sha256(records),
        }
        offset = next_offset
    expected_macro = float(np.mean([
        state_results[state]["expected_metrics"]["AUPR"]
        for state in STATE_NAMES
    ]))
    return {
        "dataset": name,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "outer_assignments_sha256": outer_metadata["assignments_sha256"],
        },
        "sources": {
            relation: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for relation, path in source_paths.items()
        },
        "inner_protocol": inner_metadata,
        "transfer_statistics": transfer["statistics"],
        "states": state_results,
        "expected_macro_AUPR": expected_macro,
    }


def gate_results(datasets, thresholds):
    dataset_macros = [row["expected_macro_AUPR"] for row in datasets]
    passing_datasets = sum(
        value >= thresholds["minimum_dataset_expected_macro_AUPR"]
        for value in dataset_macros
    )
    cold_cells = [
        row["states"][state]["expected_metrics"]
        for row in datasets
        for state in ("cold_warm", "warm_cold", "cold_cold")
    ]
    passing_cold_cells = sum(
        cell["AUPR"] >= thresholds["minimum_cold_state_AUPR"]
        for cell in cold_cells
    )
    coverage_pass = all(
        row["states"][state]["expected_metrics"]["coverage"]
        >= thresholds["minimum_expected_channel_coverage"]
        for row in datasets for state in STATE_NAMES
    )
    checks = {
        "cross_dataset_expected_macro": (
            float(np.mean(dataset_macros))
            >= thresholds["minimum_cross_dataset_expected_macro_AUPR"]
        ),
        "passing_dataset_count": (
            passing_datasets >= thresholds["minimum_passing_datasets"]
        ),
        "passing_cold_state_cell_count": (
            passing_cold_cells
            >= thresholds["minimum_passing_cold_state_cells"]
        ),
        "expected_channel_coverage": coverage_pass,
    }
    return {
        "decision": "PASS" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "cross_dataset_expected_macro_AUPR": float(np.mean(dataset_macros)),
        "passing_datasets": int(passing_datasets),
        "cold_state_cells": len(cold_cells),
        "passing_cold_state_cells": int(passing_cold_cells),
    }


def build_markdown(report):
    gate = report["gate"]
    lines = [
        "# LOCO 双超图原型迁移可行性审计",
        "",
        "- 审计类型：四库冻结 support-complete 单元内的 inner-validation。",
        "- 模型训练、checkpoint 恢复和外层测试读取次数均为 `0`。",
        "- 统计量仅由 inner-training 正例、H-C 和 P-D 构造。",
        "- compound/protein 上下文相似矩阵均删除对角线，禁止自身标签回流。",
        "- 固定通道：`H→P`、`D→C`、双侧迁移及三者等权融合；未搜索权重。",
        "",
        "## 预注册判定",
        "",
        "**%s**" % gate["decision"],
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for key, value in gate["checks"].items():
        lines.append("| `%s` | `%s` |" % (key, value))
    lines.extend([
        "",
        "跨库 expected-channel Macro-AUPR：`%.6f`；通过数据集：`%d/%d`；"
        "通过 cold-state 单元：`%d/%d`。" % (
            gate["cross_dataset_expected_macro_AUPR"],
            gate["passing_datasets"], len(report["datasets"]),
            gate["passing_cold_state_cells"], gate["cold_state_cells"],
        ),
        "",
        "## Expected-channel 结果",
        "",
        "| 数据集 | WW Fusion | CW H→P | WC D→C | CC Dual | Macro |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["datasets"]:
        states = row["states"]
        lines.append(
            "| %s | %.6f | %.6f | %.6f | %.6f | %.6f |" % (
                row["dataset"],
                states["warm_warm"]["expected_metrics"]["AUPR"],
                states["cold_warm"]["expected_metrics"]["AUPR"],
                states["warm_cold"]["expected_metrics"]["AUPR"],
                states["cold_cold"]["expected_metrics"]["AUPR"],
                row["expected_macro_AUPR"],
            )
        )
    lines.extend([
        "",
        "## 覆盖率与方向",
        "",
        "| 数据集 | 状态 | 通道 | 非零覆盖率 | 正负均值差 |",
        "|---|---|---|---:|---:|",
    ])
    for row in report["datasets"]:
        for state in STATE_NAMES:
            state_row = row["states"][state]
            metric = state_row["expected_metrics"]
            lines.append("| %s | %s | %s | %.6f | %+.6f |" % (
                row["dataset"], state, state_row["expected_channel"],
                metric["coverage"], metric["positive_minus_negative"],
            ))
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- `PASS` 只表示存在值得进入 validation-only Pilot 的统计信号，"
        "不表示新模块已经成为论文创新。",
        "- `NO-GO` 时不得通过查看外层测试、搜索融合权重、Top-K、温度或"
        "数据库专用规则挽救该候选。",
        "- 即使通过，下一阶段也必须比较 `保留 PageRank`、`删除 compound "
        "PageRank` 和 `删除 compound PageRank + LOCO-DHPT`，才能证明替换成立。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    if not 0.0 < args.inner_ratio < 1.0:
        raise ValueError("--inner-ratio must be between 0 and 1")
    datasets = parse_datasets(args.dataset)
    print("LOCO dual-hypergraph transfer audit")
    print("  inner ratio: %.3f; seed: %d" % (
        args.inner_ratio, args.inner_seed
    ))
    print("  outer test scored: no; optimizer steps: 0")
    results = []
    for name, manifest_path in datasets.items():
        print("  auditing %s ..." % name)
        results.append(audit_dataset(
            name, manifest_path, args.inner_ratio, args.inner_seed
        ))
    gate = gate_results(results, DEFAULT_THRESHOLDS)
    report = {
        "version": 1,
        "protocol": "loco_dual_hypergraph_transfer_inner_audit",
        "created_at": datetime.now().astimezone().isoformat(),
        "optimizer_steps": 0,
        "outer_test_scored": False,
        "outer_test_used_for_selection": False,
        "formula_search": False,
        "parameters": {
            "inner_ratio": args.inner_ratio,
            "inner_seed": args.inner_seed,
            "compound_similarity": "row-normalized self-excluded H-C cosine",
            "protein_similarity": "row-normalized self-excluded P-D cosine",
            "C-P_prototypes": "row/column normalized inner-training positives",
            "fusion": "unweighted mean",
        },
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "datasets": results,
        "gate": gate,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(report)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(markdown + "\n", encoding="utf-8")
    documentation = Path(args.documentation).expanduser().resolve()
    documentation.parent.mkdir(parents=True, exist_ok=True)
    documentation.write_text(markdown + "\n", encoding="utf-8")
    print("Decision: %s" % gate["decision"])
    print("Summary: %s" % summary_path)
    print("Documentation: %s" % documentation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
