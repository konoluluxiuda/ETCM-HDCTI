#!/usr/bin/env python3
"""Build a deterministic manual-review queue for unresolved SymMap compounds."""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLIST = (
    REPOSITORY_ROOT / "results" / "symmap_attribute_enrichment"
    / "compound_worklist.csv"
)
DEFAULT_ATTRIBUTES = (
    REPOSITORY_ROOT / "results" / "multidataset_attributes" / "SymMap2.0"
    / "compound_attributes.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "symmap_attribute_enrichment" / "review"
)
VERIFIED_FORMULA_STATUSES = {
    "exact", "composition_equivalent", "composition_match_charge_diff",
    "yes", "matched",
}
OUTPUT_FIELDS = [
    "entity_id", "canonical_name", "source_formula", "source_pubchem_id",
    "source_cas_id", "source_tcmsp_id", "current_status",
    "formula_status", "existing_pubchem_cid", "candidate_cids",
    "review_track", "review_priority", "threshold_review_batch",
    "review_instruction",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", default=str(DEFAULT_WORKLIST))
    parser.add_argument("--attributes", default=str(DEFAULT_ATTRIBUTES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--minimum-coverage", type=float, default=0.70)
    parser.add_argument("--buffer", type=int, default=25)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_key(value):
    value = str(value or "")
    return (0, int(value)) if value.isdigit() else (1, value)


def review_rule(attribute):
    status = attribute.get("resolution_status", "")
    formula_status = attribute.get("formula_match", "")
    has_smiles = bool(attribute.get("canonical_smiles", "").strip())
    if has_smiles and formula_status not in VERIFIED_FORMULA_STATUSES | {"not_available"}:
        return (
            "quality_conflict", 1,
            "核对来源分子式、PubChem CID、盐型/水合物；未确认前保留冲突标记。",
        )
    if has_smiles:
        return None
    rules = {
        "pending_tcmsp_cross_reference": (
            "coverage_candidate", 1,
            "用 TCMSP 页面中的 PubChem CID 或 InChIKey 独立确认后接受。",
        ),
        "manual_review_unique_name": (
            "coverage_candidate", 1,
            "唯一名称候选仍需用同义词、CAS、分子式或另一数据库独立确认。",
        ),
        "ambiguous_pubchem_candidates": (
            "coverage_candidate", 2,
            "逐个比较候选名称、CAS、分子式与 InChIKey，不按返回顺序选择。",
        ),
        "conflict_formula": (
            "coverage_candidate", 2,
            "PubChem 候选与来源分子式冲突；确认来源错误或盐型差异后再接受。",
        ),
        "pending_manual_name_review": (
            "coverage_candidate", 3,
            "先检索 PubChem 候选；名称单独命中不能自动作为确定映射。",
        ),
        "manual_review_ambiguous_name": (
            "coverage_candidate", 3,
            "名称返回多个候选，必须补充独立标识后人工选择。",
        ),
        "not_found_pubchem": (
            "coverage_candidate", 4,
            "检查英文名、拼写和同义词，并尝试 TCMSP/CAS 交叉引用。",
        ),
        "request_error_pubchem_lookup": (
            "coverage_candidate", 5,
            "网络请求失败；重试后按候选证据强度继续审查。",
        ),
    }
    return rules.get(status, (
        "coverage_candidate", 5,
        "当前记录未解析；补充独立来源标识后再决定是否接受。",
    ))


def build_queue(worklist_rows, attribute_rows, minimum_coverage=0.70, buffer=25):
    worklist = {row["local_entity_id"]: row for row in worklist_rows}
    queue = []
    smiles_count = sum(bool(row.get("canonical_smiles", "").strip())
                       for row in attribute_rows)
    required_count = math.ceil(len(attribute_rows) * minimum_coverage)
    coverage_gap = max(0, required_count - smiles_count)
    for attribute in attribute_rows:
        rule = review_rule(attribute)
        if rule is None:
            continue
        entity_id = attribute["entity_id"]
        source = worklist.get(entity_id, {})
        track, priority, instruction = rule
        queue.append({
            "entity_id": entity_id,
            "canonical_name": source.get("canonical_name", ""),
            "source_formula": source.get("molecular_formula", ""),
            "source_pubchem_id": source.get("pubchem_id", ""),
            "source_cas_id": source.get("cas_id", ""),
            "source_tcmsp_id": source.get("tcmsp_id", ""),
            "current_status": attribute.get("resolution_status", ""),
            "formula_status": attribute.get("formula_match", ""),
            "existing_pubchem_cid": attribute.get("pubchem_cid", ""),
            "candidate_cids": attribute.get("candidate_cids", ""),
            "review_track": track,
            "review_priority": priority,
            "threshold_review_batch": "no",
            "review_instruction": instruction,
        })
    queue.sort(key=lambda row: (
        0 if row["review_track"] == "coverage_candidate" else 1,
        int(row["review_priority"]), numeric_key(row["entity_id"]),
    ))
    target_batch_size = min(
        sum(row["review_track"] == "coverage_candidate" for row in queue),
        coverage_gap + max(0, buffer),
    )
    assigned = 0
    for row in queue:
        if row["review_track"] == "coverage_candidate" and assigned < target_batch_size:
            row["threshold_review_batch"] = "yes"
            assigned += 1
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "total_compounds": len(attribute_rows),
        "smiles_entities": smiles_count,
        "smiles_coverage": smiles_count / len(attribute_rows) if attribute_rows else 0.0,
        "minimum_coverage": minimum_coverage,
        "required_smiles_entities": required_count,
        "coverage_gap": coverage_gap,
        "review_buffer": max(0, buffer),
        "threshold_review_batch_size": target_batch_size,
        "coverage_candidates": sum(
            row["review_track"] == "coverage_candidate" for row in queue
        ),
        "quality_conflicts": sum(
            row["review_track"] == "quality_conflict" for row in queue
        ),
        "status_counts": dict(sorted(Counter(
            row["current_status"] for row in queue
        ).items())),
    }
    return queue, report


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    queue, report = build_queue(
        read_csv(args.worklist), read_csv(args.attributes),
        minimum_coverage=args.minimum_coverage, buffer=args.buffer,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_csv(output_dir / "compound_review_queue.csv", queue)
    (output_dir / "review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Review queue written to: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
