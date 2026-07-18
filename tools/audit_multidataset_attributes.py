#!/usr/bin/env python3
"""Audit entity-attribute readiness across all HDCTI datasets."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "TCM-Suite": REPOSITORY_ROOT / "dataset" / "TCMsuite",
    "TCMSP": REPOSITORY_ROOT / "dataset" / "TCMSP",
    "SymMap2.0": REPOSITORY_ROOT / "dataset" / "Symmap",
    "ETCM2.0-mention10": REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_mention10",
}
CP_CANDIDATES = ("C_P.txt", "compound-protein.txt", "IT.txt")
UNIPROT_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
TRUE_VALUES = {"1", "true", "yes", "y", "match", "matched"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether TCM-Suite, TCMSP, SymMap and ETCM have biological "
            "entity mappings and actual molecular/sequence attributes."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override defaults or add a dataset; may be repeated.",
    )
    parser.add_argument(
        "--alignment-root",
        help=(
            "Optional root containing <dataset>/compound_attributes.csv and "
            "<dataset>/protein_attributes.csv standardized enrichment files."
        ),
    )
    parser.add_argument(
        "--entity-mapping", action="append", default=[], metavar="NAME=DIR",
        help=(
            "Verified directory containing compound_alignment.csv and "
            "protein_alignment.csv for one dataset; may be repeated."
        ),
    )
    parser.add_argument("--minimum-smiles-coverage", type=float, default=0.70)
    parser.add_argument("--minimum-sequence-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-formula-match", type=float, default=0.95)
    parser.add_argument("--minimum-ready-datasets", type=int, default=3)
    parser.add_argument(
        "--output-dir", default="results/multidataset_attribute_coverage"
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


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.casefold().split())


def split_values(value):
    return [
        token.strip()
        for token in re.split(r"[|;]", str(value or ""))
        if token.strip()
    ]


def nonempty(value):
    return bool(str(value or "").strip())


def resolve_cp_path(dataset_dir):
    for filename in CP_CANDIDATES:
        path = dataset_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError("No C-P relation file found in %s" % dataset_dir)


def relation_entities(path):
    compounds = set()
    proteins = set()
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            compounds.add(parts[0])
            proteins.add(parts[1])
    return compounds, proteins, malformed


def rich_mapping_paths(dataset_dir):
    mapping_dir = dataset_dir / "mappings"
    compound = mapping_dir / "compound_id_map.csv"
    protein = mapping_dir / "protein_id_map.csv"
    return (compound, protein) if compound.exists() and protein.exists() else (None, None)


def anonymous_mapping_paths(dataset_dir):
    compound_candidates = (
        dataset_dir / "compound_id_all.csv",
        dataset_dir / "compound_id_all.txt",
    )
    protein_candidates = (
        dataset_dir / "protein_id_all.csv",
        dataset_dir / "protein_id_all.txt",
    )
    compound = next((path for path in compound_candidates if path.exists()), None)
    protein = next((path for path in protein_candidates if path.exists()), None)
    return compound, protein


def read_rich_rows(path, id_column, selected_ids):
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entity_id = str(row[id_column]).strip()
            if entity_id in selected_ids:
                rows[entity_id] = row
    return rows


def read_anonymous_crosswalk(path):
    ids = set()
    labels = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip():
                entity_id = row[0].strip()
                ids.add(entity_id)
                if not entity_id.isdigit():
                    labels.add(entity_id)
    return ids, labels


def valid_uniprot_entities(rows):
    valid = set()
    for entity_id, row in rows.items():
        for accession in split_values(row.get("uniprot_accessions")):
            if UNIPROT_PATTERN.fullmatch(accession.upper()):
                valid.add(entity_id)
                break
    return valid


def compound_lookup_entities(rows):
    lookup = set()
    for entity_id, row in rows.items():
        has_source_id = nonempty(row.get("tcmip_ids"))
        has_name_formula = (
            nonempty(row.get("ingredient_names"))
            and nonempty(row.get("molecular_formula"))
        )
        if has_source_id or has_name_formula:
            lookup.add(entity_id)
    return lookup


def read_verified_alignment(path, selected_ids):
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "local_entity_id" not in (reader.fieldnames or []):
            raise ValueError("%s requires local_entity_id." % path)
        return {
            str(row["local_entity_id"]).strip(): row
            for row in reader
            if str(row.get("local_entity_id", "")).strip() in selected_ids
        }


def verified_mapping_audit(mapping_dir, compound_ids, protein_ids):
    if mapping_dir is None:
        return None
    mapping_dir = Path(mapping_dir).expanduser().resolve()
    compound_path = mapping_dir / "compound_alignment.csv"
    protein_path = mapping_dir / "protein_alignment.csv"
    if not compound_path.exists() or not protein_path.exists():
        raise FileNotFoundError(
            "Verified alignment directory requires compound_alignment.csv and "
            "protein_alignment.csv: %s" % mapping_dir
        )
    compounds = read_verified_alignment(compound_path, compound_ids)
    proteins = read_verified_alignment(protein_path, protein_ids)
    compound_lookup = {
        entity_id for entity_id, row in compounds.items()
        if nonempty(row.get("source_entity_id")) and (
            nonempty(row.get("canonical_name"))
            or nonempty(row.get("pubchem_id"))
            or nonempty(row.get("cas_id"))
            or nonempty(row.get("tcmsp_id"))
        )
    }
    protein_lookup = {
        entity_id for entity_id, row in proteins.items()
        if nonempty(row.get("source_entity_id")) and any(
            nonempty(row.get(column)) for column in (
                "uniprot_id", "ensembl_id", "ncbi_id", "gene_symbol"
            )
        )
    }
    return {
        "mapping_type": "verified_official_biological_metadata",
        "compound_mapping_path": str(compound_path),
        "protein_mapping_path": str(protein_path),
        "compound_mapping_sha256": sha256_file(compound_path),
        "protein_mapping_sha256": sha256_file(protein_path),
        "compound_rows": len(compounds),
        "protein_rows": len(proteins),
        "compound_mapping_coverage": ratio(len(compounds), len(compound_ids)),
        "protein_mapping_coverage": ratio(len(proteins), len(protein_ids)),
        "compound_biological_lookup_entities": len(compound_lookup),
        "compound_biological_lookup_coverage": ratio(
            len(compound_lookup), len(compound_ids)
        ),
        "protein_biological_lookup_entities": len(protein_lookup),
        "protein_biological_lookup_coverage": ratio(
            len(protein_lookup), len(protein_ids)
        ),
        "available_fields": {
            "compound": sorted(next(iter(compounds.values())).keys())
            if compounds else [],
            "protein": sorted(next(iter(proteins.values())).keys())
            if proteins else [],
        },
    }


def local_mapping_audit(
        dataset_dir, compound_ids, protein_ids, verified_mapping_dir=None):
    verified = verified_mapping_audit(
        verified_mapping_dir, compound_ids, protein_ids
    )
    if verified is not None:
        return verified
    rich_compound, rich_protein = rich_mapping_paths(dataset_dir)
    if rich_compound is not None:
        compounds = read_rich_rows(rich_compound, "compound_id", compound_ids)
        proteins = read_rich_rows(rich_protein, "protein_id", protein_ids)
        compound_lookup = compound_lookup_entities(compounds)
        protein_lookup = valid_uniprot_entities(proteins)
        return {
            "mapping_type": "biological_metadata",
            "compound_mapping_path": str(rich_compound),
            "protein_mapping_path": str(rich_protein),
            "compound_mapping_sha256": sha256_file(rich_compound),
            "protein_mapping_sha256": sha256_file(rich_protein),
            "compound_rows": len(compounds),
            "protein_rows": len(proteins),
            "compound_mapping_coverage": ratio(len(compounds), len(compound_ids)),
            "protein_mapping_coverage": ratio(len(proteins), len(protein_ids)),
            "compound_biological_lookup_entities": len(compound_lookup),
            "compound_biological_lookup_coverage": ratio(
                len(compound_lookup), len(compound_ids)
            ),
            "protein_biological_lookup_entities": len(protein_lookup),
            "protein_biological_lookup_coverage": ratio(
                len(protein_lookup), len(protein_ids)
            ),
            "available_fields": {
                "compound": sorted(next(iter(compounds.values())).keys())
                if compounds else [],
                "protein": sorted(next(iter(proteins.values())).keys())
                if proteins else [],
            },
        }

    anonymous_compound, anonymous_protein = anonymous_mapping_paths(dataset_dir)
    if anonymous_compound is not None and anonymous_protein is not None:
        compound_rows, compound_labels = read_anonymous_crosswalk(
            anonymous_compound
        )
        protein_rows, protein_labels = read_anonymous_crosswalk(anonymous_protein)
        mapped_compounds = compound_rows & compound_ids
        mapped_proteins = protein_rows & protein_ids
        named_source_queries = (
            "molecule_ID" in compound_labels and "target_ID" in protein_labels
        )
        if named_source_queries:
            return {
                "mapping_type": "source_database_query_ids",
                "compound_mapping_path": str(anonymous_compound),
                "protein_mapping_path": str(anonymous_protein),
                "compound_mapping_sha256": sha256_file(anonymous_compound),
                "protein_mapping_sha256": sha256_file(anonymous_protein),
                "compound_rows": len(mapped_compounds),
                "protein_rows": len(mapped_proteins),
                "compound_mapping_coverage": ratio(
                    len(mapped_compounds), len(compound_ids)
                ),
                "protein_mapping_coverage": ratio(
                    len(mapped_proteins), len(protein_ids)
                ),
                "compound_biological_lookup_entities": len(mapped_compounds),
                "compound_biological_lookup_coverage": ratio(
                    len(mapped_compounds), len(compound_ids)
                ),
                "protein_biological_lookup_entities": len(mapped_proteins),
                "protein_biological_lookup_coverage": ratio(
                    len(mapped_proteins), len(protein_ids)
                ),
                "available_fields": {
                    "compound": ["TCMSP:molecule_query", "matrix_entity_id"],
                    "protein": ["TCMSP:target_query", "matrix_entity_id"],
                },
            }
        return {
            "mapping_type": "anonymous_numeric_ids",
            "compound_mapping_path": str(anonymous_compound),
            "protein_mapping_path": str(anonymous_protein),
            "compound_mapping_sha256": sha256_file(anonymous_compound),
            "protein_mapping_sha256": sha256_file(anonymous_protein),
            "compound_rows": len(mapped_compounds),
            "protein_rows": len(mapped_proteins),
            "compound_mapping_coverage": ratio(
                len(mapped_compounds), len(compound_ids)
            ),
            "protein_mapping_coverage": ratio(len(mapped_proteins), len(protein_ids)),
            "compound_biological_lookup_entities": 0,
            "compound_biological_lookup_coverage": 0.0,
            "protein_biological_lookup_entities": 0,
            "protein_biological_lookup_coverage": 0.0,
            "available_fields": {
                "compound": ["anonymous_numeric_id"],
                "protein": ["anonymous_numeric_id"],
            },
        }

    return {
        "mapping_type": "missing",
        "compound_mapping_path": None,
        "protein_mapping_path": None,
        "compound_rows": 0,
        "protein_rows": 0,
        "compound_mapping_coverage": 0.0,
        "protein_mapping_coverage": 0.0,
        "compound_biological_lookup_entities": 0,
        "compound_biological_lookup_coverage": 0.0,
        "protein_biological_lookup_entities": 0,
        "protein_biological_lookup_coverage": 0.0,
        "available_fields": {"compound": [], "protein": []},
    }


def alignment_paths(alignment_root, dataset_name):
    if alignment_root is None:
        return None, None
    directory = Path(alignment_root).expanduser().resolve() / dataset_name
    return directory / "compound_attributes.csv", directory / "protein_attributes.csv"


def read_attribute_rows(path, selected_ids):
    if path is None or not path.exists():
        return {}, None
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "entity_id" not in (reader.fieldnames or []):
            raise ValueError("%s requires an entity_id column." % path)
        for row in reader:
            entity_id = str(row["entity_id"]).strip()
            if entity_id in selected_ids:
                rows[entity_id] = row
    return rows, sha256_file(path)


def actual_attribute_audit(
        compound_path, protein_path, compound_ids, protein_ids):
    compounds, compound_sha = read_attribute_rows(compound_path, compound_ids)
    proteins, protein_sha = read_attribute_rows(protein_path, protein_ids)
    smiles_ids = set()
    formula_match_ids = set()
    for entity_id, row in compounds.items():
        if any(nonempty(row.get(column)) for column in (
                "canonical_smiles", "isomeric_smiles", "smiles")):
            smiles_ids.add(entity_id)
        if normalize_text(row.get("formula_match")) in TRUE_VALUES:
            formula_match_ids.add(entity_id)
    sequence_ids = {
        entity_id for entity_id, row in proteins.items()
        if nonempty(row.get("sequence"))
    }
    return {
        "compound_path": str(compound_path) if compound_sha else None,
        "protein_path": str(protein_path) if protein_sha else None,
        "compound_sha256": compound_sha,
        "protein_sha256": protein_sha,
        "compound_rows": len(compounds),
        "protein_rows": len(proteins),
        "smiles_entities": len(smiles_ids),
        "smiles_coverage": ratio(len(smiles_ids), len(compound_ids)),
        "formula_match_entities": len(smiles_ids & formula_match_ids),
        "formula_match_rate_among_smiles": ratio(
            len(smiles_ids & formula_match_ids), len(smiles_ids)
        ),
        "sequence_entities": len(sequence_ids),
        "sequence_coverage": ratio(len(sequence_ids), len(protein_ids)),
    }


def dataset_decision(local, actual, thresholds):
    actual_ready = (
        actual["smiles_coverage"] >= thresholds["minimum_smiles_coverage"]
        and actual["formula_match_rate_among_smiles"]
        >= thresholds["minimum_formula_match"]
        and actual["sequence_coverage"] >= thresholds["minimum_sequence_coverage"]
    )
    if actual_ready:
        return "supports_multimodal_pilot"
    biological_lookup_ready = (
        local["compound_biological_lookup_coverage"]
        >= thresholds["minimum_smiles_coverage"]
        and local["protein_biological_lookup_coverage"]
        >= thresholds["minimum_sequence_coverage"]
    )
    if biological_lookup_ready:
        return "pending_external_enrichment"
    return "blocked_missing_biological_mapping"


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


def parse_mapping_overrides(values):
    mappings = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--entity-mapping must use NAME=DIR: %s" % value)
        name, path = value.split("=", 1)
        mappings[name.strip()] = Path(path).expanduser().resolve()
    return mappings


def audit_datasets(
        datasets, alignment_root=None, thresholds=None, mapping_overrides=None):
    thresholds = thresholds or {
        "minimum_smiles_coverage": 0.70,
        "minimum_sequence_coverage": 0.95,
        "minimum_formula_match": 0.95,
        "minimum_ready_datasets": 3,
    }
    mapping_overrides = mapping_overrides or {}
    rows = []
    for name, directory in datasets.items():
        directory = Path(directory).expanduser().resolve()
        cp_path = resolve_cp_path(directory)
        compound_ids, protein_ids, malformed = relation_entities(cp_path)
        local = local_mapping_audit(
            directory, compound_ids, protein_ids, mapping_overrides.get(name)
        )
        compound_path, protein_path = alignment_paths(alignment_root, name)
        actual = actual_attribute_audit(
            compound_path, protein_path, compound_ids, protein_ids
        )
        rows.append({
            "name": name,
            "path": str(directory),
            "cp_path": str(cp_path),
            "cp_sha256": sha256_file(cp_path),
            "cp_malformed_rows": malformed,
            "compound_entities": len(compound_ids),
            "protein_entities": len(protein_ids),
            "local_mapping": local,
            "actual_attributes": actual,
            "decision": dataset_decision(local, actual, thresholds),
        })
    ready = sum(row["decision"] == "supports_multimodal_pilot" for row in rows)
    pending = sum(
        row["decision"] == "pending_external_enrichment" for row in rows
    )
    if ready >= thresholds["minimum_ready_datasets"]:
        decision = "supports_cross_dataset_multimodal_pilot"
    elif ready + pending >= thresholds["minimum_ready_datasets"]:
        decision = "pending_cross_dataset_enrichment"
    else:
        decision = "blocked_cross_dataset_entity_alignment"
    return {
        "audit_type": "multidataset_entity_attribute_coverage",
        "created_at": datetime.now().astimezone().isoformat(),
        "network_accessed": False,
        "training_steps": 0,
        "thresholds": thresholds,
        "decision": decision,
        "ready_datasets": ready,
        "datasets": rows,
    }


def build_markdown(report):
    lines = [
        "# 多数据集实体属性覆盖审计",
        "",
        "- 本步骤不访问网络、不训练模型。",
        "- 匿名数字映射只能证明矩阵 ID 可追踪，不能用于检索 SMILES 或蛋白序列。",
        "- 只有标准化 enrichment 文件中的真实 SMILES/sequence 才计入模态覆盖。",
        "",
        "## 总体判定",
        "",
        "**%s**" % report["decision"],
        "",
        "| 数据集 | C | P | 映射类型 | 化合物生物标识 | 蛋白生物标识 | SMILES | Sequence | 判定 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report["datasets"]:
        local = row["local_mapping"]
        actual = row["actual_attributes"]
        lines.append(
            "| %s | %d | %d | %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %s |" % (
                row["name"], row["compound_entities"], row["protein_entities"],
                local["mapping_type"],
                100.0 * local["compound_biological_lookup_coverage"],
                100.0 * local["protein_biological_lookup_coverage"],
                100.0 * actual["smiles_coverage"],
                100.0 * actual["sequence_coverage"], row["decision"]
            )
        )
    lines.extend([
        "",
        "## 主模型 Go/No-Go 门槛",
        "",
        "- 单数据集 SMILES 覆盖率 >= %.0f%%。" % (
            100.0 * report["thresholds"]["minimum_smiles_coverage"]),
        "- 已映射 SMILES 的分子式确认率 >= %.0f%%。" % (
            100.0 * report["thresholds"]["minimum_formula_match"]),
        "- 单数据集蛋白序列覆盖率 >= %.0f%%。" % (
            100.0 * report["thresholds"]["minimum_sequence_coverage"]),
        "- 至少 %d 个数据集同时达到门槛，才将多模态作为共享主创新。" % (
            report["thresholds"]["minimum_ready_datasets"]),
        "",
        "## 标准化 enrichment 接口",
        "",
        "```text",
        "<alignment-root>/<dataset>/compound_attributes.csv",
        "  entity_id,canonical_smiles,pubchem_cid,formula_match",
        "<alignment-root>/<dataset>/protein_attributes.csv",
        "  entity_id,uniprot_accession,sequence",
        "```",
        "",
        "当前总体判定未通过时，不应先实现多模态模型；应先补齐原始数据库实体映射，或改选不依赖实体属性且能覆盖全部数据集的结构创新。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    thresholds = {
        "minimum_smiles_coverage": args.minimum_smiles_coverage,
        "minimum_sequence_coverage": args.minimum_sequence_coverage,
        "minimum_formula_match": args.minimum_formula_match,
        "minimum_ready_datasets": args.minimum_ready_datasets,
    }
    report = audit_datasets(
        parse_dataset_overrides(args.dataset),
        alignment_root=args.alignment_root,
        thresholds=thresholds,
        mapping_overrides=parse_mapping_overrides(args.entity_mapping),
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
