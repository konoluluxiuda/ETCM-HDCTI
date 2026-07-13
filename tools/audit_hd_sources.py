#!/usr/bin/env python3
"""Audit whether H-D edges can be reconstructed from H-C, C-P, and P-D."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

from build_etcm2_entity_mappings import basic_info_map, clean_text, first_value, load_relevant_sections, related_blocks


Edge = Tuple[str, str]


DATASET_FILES: Mapping[str, Mapping[str, str]] = {
    "TCMsuite": {"H_C": "H_C.txt", "C_P": "C_P.txt", "P_D": "P_D.txt", "H_D": "H_D.txt"},
    "TCMSP": {
        "H_C": "herb-compound.txt",
        "C_P": "compound-protein.txt",
        "P_D": "target-disease.txt",
        "H_D": "drug-disease.txt",
    },
    "Symmap": {"H_C": "HI.txt", "C_P": "IT.txt", "P_D": "TD.txt", "H_D": "HD.txt"},
}


def read_edges(path: Path, positive_only: bool = False) -> Tuple[Set[Edge], int]:
    edges: Set[Edge] = set()
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            if positive_only and len(parts) >= 3:
                try:
                    if float(parts[2]) <= 0:
                        continue
                except ValueError:
                    malformed += 1
                    continue
            edges.add((parts[0], parts[1]))
    return edges, malformed


def group(edges: Iterable[Edge]) -> Dict[str, Set[str]]:
    grouped: MutableMapping[str, Set[str]] = defaultdict(set)
    for left, right in edges:
        grouped[left].add(right)
    return dict(grouped)


def percent(part: int, total: int) -> float:
    return round(100.0 * part / total, 4) if total else 0.0


def relation_files(dataset_dir: Path) -> Mapping[str, str]:
    if dataset_dir.name in DATASET_FILES:
        return DATASET_FILES[dataset_dir.name]
    return {name: f"{name}.txt" for name in ("H_C", "C_P", "P_D", "H_D")}


def closure_stats(hc: Set[Edge], cp: Set[Edge], pd: Set[Edge], hd: Set[Edge]) -> Dict[str, object]:
    compounds_by_herb = group(hc)
    proteins_by_compound = group(cp)
    diseases_by_protein = group(pd)
    hd_by_herb = group(hd)

    closure_count = 0
    overlap_count = 0
    per_herb_support: Dict[str, Set[str]] = {}
    for herb, compounds in compounds_by_herb.items():
        reachable_diseases: Set[str] = set()
        for compound in compounds:
            for protein in proteins_by_compound.get(compound, ()):
                reachable_diseases.update(diseases_by_protein.get(protein, ()))
        closure_count += len(reachable_diseases)
        overlap_count += len(reachable_diseases & hd_by_herb.get(herb, set()))
        if herb in hd_by_herb:
            per_herb_support[herb] = reachable_diseases

    unsupported_examples: List[Edge] = []
    for herb, disease in sorted(hd):
        if disease not in per_herb_support.get(herb, set()):
            unsupported_examples.append((herb, disease))
            if len(unsupported_examples) == 20:
                break

    return {
        "hc_pd_cp_closure_edges": closure_count,
        "hd_edges_supported_by_hc_cp_pd": overlap_count,
        "hd_support_percent": percent(overlap_count, len(hd)),
        "closure_retained_as_hd_percent": percent(overlap_count, closure_count),
        "hd_equals_hc_cp_pd_closure": overlap_count == len(hd) == closure_count,
        "unsupported_hd_examples": unsupported_examples,
    }


def pair_has_chdp_support(
    compound: str,
    protein: str,
    herbs_by_compound: Mapping[str, Set[str]],
    diseases_by_protein: Mapping[str, Set[str]],
    hd_by_herb: Mapping[str, Set[str]],
) -> bool:
    protein_diseases = diseases_by_protein.get(protein, set())
    if not protein_diseases:
        return False
    return any(
        bool(hd_by_herb.get(herb, set()) & protein_diseases)
        for herb in herbs_by_compound.get(compound, ())
    )


def fold_signal_stats(dataset_dir: Path, hc: Set[Edge], pd: Set[Edge], hd: Set[Edge]) -> List[Dict[str, object]]:
    herbs_by_compound = group((compound, herb) for herb, compound in hc)
    diseases_by_protein = group(pd)
    hd_by_herb = group(hd)
    rows: List[Dict[str, object]] = []

    for fold_path in sorted(dataset_dir.glob("test_fold_*.txt")):
        positives: Set[Edge] = set()
        negatives: Set[Edge] = set()
        malformed = 0
        with fold_path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    malformed += 1
                    continue
                try:
                    target = positives if float(parts[2]) > 0 else negatives
                except ValueError:
                    malformed += 1
                    continue
                target.add((parts[0], parts[1]))

        positive_supported = sum(
            pair_has_chdp_support(c, p, herbs_by_compound, diseases_by_protein, hd_by_herb)
            for c, p in positives
        )
        negative_supported = sum(
            pair_has_chdp_support(c, p, herbs_by_compound, diseases_by_protein, hd_by_herb)
            for c, p in negatives
        )
        rows.append(
            {
                "fold": fold_path.name,
                "positive_pairs": len(positives),
                "positive_pairs_with_chdp_support": positive_supported,
                "positive_support_percent": percent(positive_supported, len(positives)),
                "negative_pairs": len(negatives),
                "negative_pairs_with_chdp_support": negative_supported,
                "negative_support_percent": percent(negative_supported, len(negatives)),
                "support_rate_gap_percentage_points": round(
                    percent(positive_supported, len(positives))
                    - percent(negative_supported, len(negatives)),
                    4,
                ),
                "malformed_rows": malformed,
            }
        )
    return rows


def audit_dataset(dataset_dir: Path) -> Dict[str, object]:
    files = relation_files(dataset_dir)
    relations: Dict[str, Set[Edge]] = {}
    malformed: Dict[str, int] = {}
    for relation, filename in files.items():
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {relation} file: {path}")
        relations[relation], malformed[relation] = read_edges(path)

    hc = relations["H_C"]
    cp = relations["C_P"]
    pd = relations["P_D"]
    hd = relations["H_D"]
    return {
        "dataset": dataset_dir.name,
        "path": str(dataset_dir),
        "files": dict(files),
        "edge_counts": {name: len(edges) for name, edges in relations.items()},
        "malformed_rows": malformed,
        "closure_audit": closure_stats(hc, cp, pd, hd),
        "fold_chdp_signal": fold_signal_stats(dataset_dir, hc, pd, hd),
    }


def load_key_ids(path: Path, key_column: str, id_column: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[clean_text(row[key_column])] = row[id_column]
    return mapping


def audit_etcm_raw_sources(raw_root: Path, processed_root: Path) -> Dict[str, object]:
    mapping_dir = processed_root / "mappings"
    herb_ids = load_key_ids(mapping_dir / "herb_id_map.csv", "herb_key", "herb_id")
    disease_ids = load_key_ids(mapping_dir / "disease_id_map.csv", "disease_key", "disease_id")
    hd, _ = read_edges(processed_root / "H_D.txt")

    enriched_edges: Set[Edge] = set()
    raw_rows = 0
    skipped_rows = 0
    parsed_files = 0
    errors: List[Dict[str, str]] = []
    for path in sorted((raw_root / "etcm_herbs").glob("*.json")):
        try:
            sections = load_relevant_sections(path)
            info = basic_info_map(sections.get("base_information"))
            herb_key = first_value(info, "Herb Name in Pinyin") or clean_text(path.stem)
            herb_id = herb_ids.get(herb_key)
            parsed_files += 1
            for block_type, rows in related_blocks(sections.get("related_table")):
                if block_type != "disease":
                    continue
                for row in rows:
                    raw_rows += 1
                    disease_key = first_value(row, "Disease Name")
                    disease_id = disease_ids.get(disease_key)
                    if herb_id is None or disease_id is None:
                        skipped_rows += 1
                        continue
                    enriched_edges.add((herb_id, disease_id))
        except Exception as exc:
            if len(errors) < 20:
                errors.append({"file": str(path), "error": repr(exc)})

    overlap = len(enriched_edges & hd)
    return {
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "disease_page_hd_edges": len(hd),
        "herb_page_enriched_disease_edges": len(enriched_edges),
        "overlap_edges": overlap,
        "herb_enriched_edges_found_in_disease_page_hd_percent": percent(overlap, len(enriched_edges)),
        "disease_page_hd_covered_by_herb_enriched_edges_percent": percent(overlap, len(hd)),
        "herb_files_parsed": parsed_files,
        "raw_enriched_rows": raw_rows,
        "skipped_rows": skipped_rows,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path, help="Dataset directories to audit")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--etcm-raw", type=Path, help="Optional ETCM2.0 raw JSON root")
    parser.add_argument(
        "--etcm-processed",
        type=Path,
        default=Path("dataset/ETCM2.0_processed"),
        help="ETCM2.0 processed root used with --etcm-raw",
    )
    args = parser.parse_args()

    payload = {
        "definition": "H-D support exists when at least one H-C-P-D path connects the same herb and disease.",
        "datasets": [audit_dataset(path) for path in args.datasets],
    }
    if args.etcm_raw:
        payload["etcm_raw_source_audit"] = audit_etcm_raw_sources(args.etcm_raw, args.etcm_processed)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
