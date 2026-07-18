#!/usr/bin/env python3
"""Prepare traceable entity-alignment worklists for all HDCTI datasets."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
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
FIELDNAMES = (
    "dataset",
    "entity_type",
    "local_entity_id",
    "cp_positive_degree",
    "priority_rank",
    "matrix_entity_id",
    "source_entity_id",
    "source_identifier_namespace",
    "canonical_identifier",
    "canonical_name",
    "molecular_formula",
    "organism",
    "mapping_method",
    "mapping_source_file",
    "mapping_confidence",
    "review_status",
    "review_note",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic compound/protein alignment worklists without "
            "guessing biological identities from anonymous numeric IDs."
        )
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--output-dir", default="results/entity_alignment_recovery"
    )
    parser.add_argument(
        "--alignment", action="append", default=[], metavar="NAME=DIR",
        help=(
            "Use verified external compound_alignment.csv and "
            "protein_alignment.csv for one dataset; may be repeated."
        ),
    )
    return parser.parse_args()


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


def parse_alignment_overrides(values):
    alignments = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--alignment must use NAME=DIR: %s" % value)
        name, path = value.split("=", 1)
        alignments[name.strip()] = Path(path).expanduser().resolve()
    return alignments


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cp_path(dataset_dir):
    for filename in CP_CANDIDATES:
        path = dataset_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError("No C-P relation file found in %s" % dataset_dir)


def read_cp_degrees(path):
    edges = set()
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            edges.add((parts[0], parts[1]))
    compounds = Counter(compound for compound, _ in edges)
    proteins = Counter(protein for _, protein in edges)
    return edges, compounds, proteins, malformed


def split_values(value):
    return [
        token.strip()
        for token in re.split(r"[|;]", str(value or ""))
        if token.strip()
    ]


def first_value(value):
    values = split_values(value)
    return values[0] if values else ""


def valid_uniprot(value):
    for accession in split_values(value):
        accession = accession.upper()
        if UNIPROT_PATTERN.fullmatch(accession):
            return accession
    return ""


def id_sort_key(value):
    text = str(value)
    if text.isdigit():
        return (0, int(text), text)
    return (1, 0, text)


def rank_entities(degrees):
    ordered = sorted(
        degrees, key=lambda entity_id: (-degrees[entity_id], id_sort_key(entity_id))
    )
    return {entity_id: rank for rank, entity_id in enumerate(ordered, 1)}


def repository_relative(path):
    if path is None:
        return ""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def read_rich_mapping(path, id_column):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row[id_column]).strip(): row
            for row in csv.DictReader(handle)
            if str(row.get(id_column, "")).strip()
        }


def read_anonymous_crosswalk(path):
    if path is None or not path.exists():
        return {}, set()
    rows = {}
    labels = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            entity_id = row[0].strip()
            rows[entity_id] = row[1].strip() if len(row) > 1 else ""
            if not entity_id.isdigit():
                labels.add(entity_id)
    return rows, labels


def read_external_alignment(path):
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"local_entity_id", "source_entity_id", "match_method"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "%s missing alignment columns: %s"
                % (path, ", ".join(sorted(missing)))
            )
        return {
            str(row["local_entity_id"]).strip(): row
            for row in reader if str(row.get("local_entity_id", "")).strip()
        }


def mapping_paths(dataset_dir, entity_type):
    stem = "compound" if entity_type == "compound" else "protein"
    rich = dataset_dir / "mappings" / (stem + "_id_map.csv")
    anonymous = None
    for suffix in (".csv", ".txt"):
        candidate = dataset_dir / (stem + "_id_all" + suffix)
        if candidate.exists():
            anonymous = candidate
            break
    return rich if rich.exists() else None, anonymous


def blank_row(dataset_name, entity_type, entity_id, degree, rank):
    row = {field: "" for field in FIELDNAMES}
    row.update({
        "dataset": dataset_name,
        "entity_type": entity_type,
        "local_entity_id": entity_id,
        "cp_positive_degree": degree,
        "priority_rank": rank,
    })
    return row


def rich_compound_row(row, mapping):
    source_id = first_value(mapping.get("tcmip_ids"))
    name = first_value(mapping.get("ingredient_names"))
    formula = str(mapping.get("molecular_formula") or "").strip()
    evidence = sum(bool(value) for value in (source_id, name, formula))
    row.update({
        "source_entity_id": source_id,
        "source_identifier_namespace": "TCMIP" if source_id else "",
        "canonical_name": name,
        "molecular_formula": formula,
        "mapping_method": "local_biological_metadata",
        "mapping_confidence": "high" if evidence == 3 else "medium",
        "review_status": (
            "ready_for_external_enrichment" if source_id or (name and formula)
            else "blocked_insufficient_biological_metadata"
        ),
        "review_note": (
            "Resolve to PubChem; confirm name and molecular formula."
        ),
    })


def rich_protein_row(row, mapping):
    accession = valid_uniprot(mapping.get("uniprot_accessions"))
    gene = first_value(mapping.get("gene_symbols"))
    source_id = accession or gene or str(mapping.get("protein_key") or "").strip()
    row.update({
        "source_entity_id": source_id,
        "source_identifier_namespace": "UniProt" if accession else "gene_symbol",
        "canonical_identifier": accession,
        "canonical_name": first_value(mapping.get("target_names")) or gene,
        "organism": first_value(mapping.get("organisms")),
        "mapping_method": "local_biological_metadata",
        "mapping_confidence": "high" if accession else "medium",
        "review_status": (
            "canonical_identifier_available" if accession
            else "needs_uniprot_resolution"
        ),
        "review_note": (
            "Fetch sequence and verify organism."
            if accession else "Resolve gene symbol to species-specific UniProt."
        ),
    })


def named_source_crosswalk_row(row, matrix_id, namespace):
    entity_type = row["entity_type"]
    row.update({
        "matrix_entity_id": matrix_id,
        "source_entity_id": row["local_entity_id"],
        "source_identifier_namespace": namespace,
        "mapping_method": "source_query_id_to_matrix_id",
        "mapping_confidence": "medium",
        "review_status": "ready_for_external_enrichment",
        "review_note": (
            "Query the TCMSP molecule page; retain returned Molecule ID, "
            "InChIKey and PubChem CID."
            if entity_type == "compound" else
            "Query the TCMSP target page; resolve returned target metadata "
            "to a species-specific UniProt accession."
        ),
    })


def anonymous_row(row, secondary_id):
    has_crosswalk = bool(secondary_id)
    row.update({
        "matrix_entity_id": secondary_id,
        "mapping_method": (
            "anonymous_numeric_crosswalk" if has_crosswalk
            else "anonymous_numeric_id_only"
        ),
        "mapping_confidence": "unresolved",
        "review_status": (
            "blocked_unknown_identifier_namespace" if has_crosswalk
            else "blocked_anonymous_id_only"
        ),
        "review_note": (
            "Identify the namespace of the secondary numeric ID from upstream data."
            if has_crosswalk
            else "Recover a pre-anonymization mapping from the upstream dataset."
        ),
    })


def missing_row(row):
    row.update({
        "mapping_method": "missing",
        "mapping_confidence": "unresolved",
        "review_status": "blocked_mapping_missing",
        "review_note": "Recover a pre-anonymization mapping from the upstream dataset.",
    })


def verified_external_row(row, mapping):
    entity_type = row["entity_type"]
    raw_canonical = str(mapping.get("canonical_identifier") or "").strip()
    canonical = (
        first_value(raw_canonical) if entity_type == "compound"
        else valid_uniprot(raw_canonical)
    )
    row.update({
        "source_entity_id": str(mapping.get("source_entity_id") or "").strip(),
        "source_identifier_namespace": str(
            mapping.get("source_identifier_namespace") or ""
        ).strip(),
        "canonical_identifier": canonical,
        "canonical_name": str(mapping.get("canonical_name") or "").strip(),
        "molecular_formula": str(
            mapping.get("molecular_formula") or ""
        ).strip(),
        "organism": str(mapping.get("organism") or "").strip(),
        "mapping_method": "verified_external_%s" % str(
            mapping.get("match_method") or "alignment"
        ).strip(),
        "mapping_confidence": (
            "high" if mapping.get("match_method") == "exact" else "medium"
        ),
        "review_status": (
            "canonical_identifier_available" if canonical else
            "ready_for_external_enrichment" if entity_type == "compound" else
            "needs_uniprot_resolution"
        ),
        "review_note": (
            "Fetch canonical SMILES and verify formula/source conflicts."
            if entity_type == "compound" else
            "Fetch sequence by UniProt; otherwise resolve Ensembl/gene symbol."
        ),
    })


def build_entity_rows(
        dataset_name, dataset_dir, entity_type, degrees, external_path=None):
    ranks = rank_entities(degrees)
    rich_path, anonymous_path = mapping_paths(dataset_dir, entity_type)
    rich = {}
    anonymous = {}
    anonymous_labels = set()
    source_path = None
    external = read_external_alignment(external_path)
    if external:
        source_path = external_path
    elif rich_path is not None:
        id_column = entity_type + "_id"
        rich = read_rich_mapping(rich_path, id_column)
        source_path = rich_path
    elif anonymous_path is not None:
        anonymous, anonymous_labels = read_anonymous_crosswalk(anonymous_path)
        source_path = anonymous_path

    named_namespace = ""
    if entity_type == "compound" and "molecule_ID" in anonymous_labels:
        named_namespace = "TCMSP:molecule_query"
    elif entity_type == "protein" and "target_ID" in anonymous_labels:
        named_namespace = "TCMSP:target_query"

    rows = []
    for entity_id in sorted(degrees, key=id_sort_key):
        row = blank_row(
            dataset_name, entity_type, entity_id,
            degrees[entity_id], ranks[entity_id]
        )
        row["mapping_source_file"] = repository_relative(source_path)
        if entity_id in external:
            verified_external_row(row, external[entity_id])
        elif entity_id in rich:
            if entity_type == "compound":
                rich_compound_row(row, rich[entity_id])
            else:
                rich_protein_row(row, rich[entity_id])
        elif entity_id in anonymous:
            if named_namespace:
                named_source_crosswalk_row(
                    row, anonymous[entity_id], named_namespace
                )
            else:
                anonymous_row(row, anonymous[entity_id])
        else:
            missing_row(row)
        rows.append(row)
    return rows, source_path


def summarize_rows(rows):
    statuses = Counter(row["review_status"] for row in rows)
    source_ready = sum(
        row["review_status"] in {
            "ready_for_external_enrichment",
            "canonical_identifier_available",
            "needs_uniprot_resolution",
        }
        for row in rows
    )
    canonical = sum(bool(row["canonical_identifier"]) for row in rows)
    return {
        "entities": len(rows),
        "source_ready_entities": source_ready,
        "source_ready_coverage": float(source_ready) / len(rows) if rows else 0.0,
        "canonical_identifier_entities": canonical,
        "canonical_identifier_coverage": (
            float(canonical) / len(rows) if rows else 0.0
        ),
        "review_status_counts": dict(sorted(statuses.items())),
    }


def prepare_manifests(datasets, alignments=None):
    alignments = alignments or {}
    report_rows = []
    all_rows = []
    for dataset_name, dataset_dir in datasets.items():
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        cp_path = resolve_cp_path(dataset_dir)
        alignment_dir = alignments.get(dataset_name)
        compound_external = (
            Path(alignment_dir) / "compound_alignment.csv"
            if alignment_dir else None
        )
        protein_external = (
            Path(alignment_dir) / "protein_alignment.csv"
            if alignment_dir else None
        )
        edges, compound_degrees, protein_degrees, malformed = read_cp_degrees(cp_path)
        compound_rows, compound_mapping = build_entity_rows(
            dataset_name, dataset_dir, "compound", compound_degrees,
            compound_external
        )
        protein_rows, protein_mapping = build_entity_rows(
            dataset_name, dataset_dir, "protein", protein_degrees,
            protein_external
        )
        all_rows.extend(compound_rows)
        all_rows.extend(protein_rows)
        report_rows.append({
            "name": dataset_name,
            "path": str(dataset_dir),
            "cp_path": repository_relative(cp_path),
            "cp_sha256": sha256_file(cp_path),
            "cp_edges": len(edges),
            "cp_malformed_rows": malformed,
            "compound_mapping_path": repository_relative(compound_mapping),
            "protein_mapping_path": repository_relative(protein_mapping),
            "compound": summarize_rows(compound_rows),
            "protein": summarize_rows(protein_rows),
        })
    return {
        "audit_type": "entity_alignment_recovery_manifest",
        "created_at": datetime.now().astimezone().isoformat(),
        "network_accessed": False,
        "identity_guessing_performed": False,
        "datasets": report_rows,
    }, all_rows


def dataset_slug(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return slug or "dataset"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(report):
    lines = [
        "# 跨库实体映射恢复清单",
        "",
        "- 本步骤不访问网络，也不根据匿名数字 ID 猜测生物实体。",
        "- `source_ready` 表示已有带命名空间的来源 ID，或具备名称和分子式可供受控检索。",
        "- `canonical` 当前主要表示已验证格式的 UniProt accession；化合物需补全 PubChem 后才计入。",
        "",
        "| 数据集 | 实体 | 数量 | Source-ready | Canonical | 当前主要状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        for entity_type in ("compound", "protein"):
            summary = dataset[entity_type]
            main_status = max(
                summary["review_status_counts"].items(),
                key=lambda item: (item[1], item[0]),
            )[0]
            lines.append(
                "| %s | %s | %d | %.2f%% | %.2f%% | %s |" % (
                    dataset["name"], entity_type, summary["entities"],
                    100.0 * summary["source_ready_coverage"],
                    100.0 * summary["canonical_identifier_coverage"],
                    main_status,
                )
            )
    lines.extend([
        "",
        "## 下一处理顺序",
        "",
        "1. TCMSP 本地 ID 可作为官方 molecule/target 页面查询键，先批量补全并审查失败率。",
        "2. SymMap2.0 官方 V2 对齐通过后，按 PubChem 与 Ensembl/UniProt 补全实际属性。",
        "3. ETCM2.0 可并行进入 PubChem/UniProt 属性补全；TCM-Suite 继续恢复映射。",
        "4. 一对多、多对一和物种冲突保留为待审查，不自动合并。",
        "",
    ])
    return "\n".join(lines)


def write_outputs(output_dir, report, rows):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_entities.csv", rows)
    for dataset in report["datasets"]:
        slug = dataset_slug(dataset["name"])
        for entity_type in ("compound", "protein"):
            selected = [
                row for row in rows
                if row["dataset"] == dataset["name"]
                and row["entity_type"] == entity_type
            ]
            write_csv(output_dir / slug / (entity_type + "_alignment.csv"), selected)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = build_markdown(report)
    (output_dir / "report.md").write_text(markdown + "\n", encoding="utf-8")
    return output_dir, markdown


def main():
    args = parse_args()
    report, rows = prepare_manifests(
        parse_dataset_overrides(args.dataset),
        parse_alignment_overrides(args.alignment),
    )
    output_dir, markdown = write_outputs(args.output_dir, report, rows)
    print(markdown)
    print("Results written to: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
