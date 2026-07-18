#!/usr/bin/env python3
"""Prepare deterministic SymMap PubChem and UniProt enrichment worklists."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPOUND_FIELDS = (
    "local_entity_id", "official_entity_id", "canonical_name",
    "molecular_formula", "pubchem_id", "cas_id", "tcmsp_id",
    "enrichment_route", "query_identifier", "priority",
    "requires_manual_review", "status"
)
PROTEIN_FIELDS = (
    "local_entity_id", "official_entity_id", "canonical_name", "gene_symbol",
    "protein_name", "uniprot_id", "ensembl_id", "ncbi_id",
    "genbank_protein_id", "enrichment_route", "query_identifier", "priority",
    "requires_manual_review", "status"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Partition verified SymMap mappings into auditable PubChem and "
            "UniProt enrichment routes without accessing the network."
        )
    )
    parser.add_argument(
        "--alignment-dir", default="results/symmap_official_alignment"
    )
    parser.add_argument(
        "--output-dir", default="results/symmap_attribute_enrichment"
    )
    parser.add_argument("--minimum-smiles-coverage", type=float, default=0.70)
    parser.add_argument("--minimum-sequence-coverage", type=float, default=0.95)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_identifier(value):
    for separator in ("|", ";"):
        value = str(value or "").split(separator, 1)[0]
    return value.strip()


def id_sort_key(row):
    value = str(row.get("local_entity_id") or "")
    return int(value) if value.isdigit() else value


def compound_route(row):
    if row.get("pubchem_id", "").strip():
        return "direct_pubchem_cid", first_identifier(row["pubchem_id"]), 1, False
    if row.get("molecular_formula", "").strip():
        return "name_formula_pubchem_lookup", row["canonical_name"].strip(), 2, False
    if row.get("cas_id", "").strip():
        return "name_cas_pubchem_lookup", first_identifier(row["cas_id"]), 3, False
    if row.get("tcmsp_id", "").strip():
        return "tcmsp_cross_reference_lookup", first_identifier(row["tcmsp_id"]), 4, False
    if row.get("canonical_name", "").strip():
        return "name_only_manual_review", row["canonical_name"].strip(), 5, True
    return "unresolved", "", 6, True


def protein_route(row):
    if row.get("uniprot_id", "").strip():
        return "direct_uniprot", first_identifier(row["uniprot_id"]), 1, False
    if row.get("ensembl_id", "").strip():
        return "ensembl_to_uniprot", first_identifier(row["ensembl_id"]), 2, False
    if row.get("ncbi_id", "").strip():
        return "ncbi_to_uniprot", first_identifier(row["ncbi_id"]), 3, False
    if row.get("gene_symbol", "").strip():
        return "gene_symbol_manual_review", row["gene_symbol"].strip(), 4, True
    return "unresolved", "", 5, True


def build_worklist(rows, entity_type):
    routed = []
    route_function = compound_route if entity_type == "compound" else protein_route
    fields = COMPOUND_FIELDS if entity_type == "compound" else PROTEIN_FIELDS
    for source in rows:
        route, identifier, priority, manual = route_function(source)
        row = {field: source.get(field, "") for field in fields}
        row.update({
            "enrichment_route": route,
            "query_identifier": identifier,
            "priority": priority,
            "requires_manual_review": "yes" if manual else "no",
            "status": "pending",
        })
        routed.append(row)
    return sorted(routed, key=lambda row: (int(row["priority"]), id_sort_key(row)))


def coverage(part, total):
    return float(part) / float(total) if total else 0.0


def summarize(compounds, proteins, minimum_smiles, minimum_sequence):
    compound_routes = Counter(row["enrichment_route"] for row in compounds)
    protein_routes = Counter(row["enrichment_route"] for row in proteins)
    direct_compounds = compound_routes["direct_pubchem_cid"]
    direct_proteins = protein_routes["direct_uniprot"]
    ensembl_candidates = protein_routes["ensembl_to_uniprot"]
    required_compounds = math.ceil(minimum_smiles * len(compounds))
    required_proteins = math.ceil(minimum_sequence * len(proteins))
    return {
        "thresholds": {
            "minimum_smiles_coverage": minimum_smiles,
            "minimum_sequence_coverage": minimum_sequence,
        },
        "compound": {
            "entities": len(compounds),
            "route_counts": dict(sorted(compound_routes.items())),
            "direct_pubchem_entities": direct_compounds,
            "direct_pubchem_coverage": coverage(direct_compounds, len(compounds)),
            "required_entities": required_compounds,
            "additional_entities_needed": max(0, required_compounds - direct_compounds),
        },
        "protein": {
            "entities": len(proteins),
            "route_counts": dict(sorted(protein_routes.items())),
            "direct_uniprot_entities": direct_proteins,
            "direct_uniprot_coverage": coverage(direct_proteins, len(proteins)),
            "ensembl_candidates": ensembl_candidates,
            "direct_plus_ensembl_entities": direct_proteins + ensembl_candidates,
            "direct_plus_ensembl_coverage": coverage(
                direct_proteins + ensembl_candidates, len(proteins)
            ),
            "required_entities": required_proteins,
            "additional_entities_needed": max(0, required_proteins - direct_proteins),
            "minimum_ensembl_success_rate": coverage(
                max(0, required_proteins - direct_proteins), ensembl_candidates
            ),
        },
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(report):
    compound = report["compound"]
    protein = report["protein"]
    lines = [
        "# SymMap 属性补全工作清单",
        "",
        "本步骤不访问网络，只根据已验证官方映射分配查询路线。名称单独匹配不会自动成为确定映射。",
        "",
        "## 门槛缺口",
        "",
        "| 实体 | 总数 | 直接标准 ID | 直接覆盖 | 门槛所需 | 仍需补全 |",
        "|---|---:|---:|---:|---:|---:|",
        "| Compound | %d | %d | %.2f%% | %d | %d |" % (
            compound["entities"], compound["direct_pubchem_entities"],
            compound["direct_pubchem_coverage"] * 100,
            compound["required_entities"], compound["additional_entities_needed"],
        ),
        "| Protein | %d | %d | %.2f%% | %d | %d |" % (
            protein["entities"], protein["direct_uniprot_entities"],
            protein["direct_uniprot_coverage"] * 100,
            protein["required_entities"], protein["additional_entities_needed"],
        ),
        "",
        "## 查询路线",
        "",
        "### Compound",
        "",
    ]
    for route, count in compound["route_counts"].items():
        lines.append("* `%s`: %d" % (route, count))
    lines.extend(["", "### Protein", ""])
    for route, count in protein["route_counts"].items():
        lines.append("* `%s`: %d" % (route, count))
    lines.extend([
        "",
        "直接 UniProt 加 Ensembl 候选覆盖为 `%.2f%%`；要达到 95%% sequence 门槛，"
        "Ensembl 候选至少需要 `%.2f%%` 成功映射。" % (
            protein["direct_plus_ensembl_coverage"] * 100,
            protein["minimum_ensembl_success_rate"] * 100,
        ),
        "",
        "## 执行约束",
        "",
        "1. PubChem CID 与 UniProt accession 可直接批量查询并缓存原始响应。",
        "2. 名称查询只有在唯一命中且分子式或独立来源标识一致时才自动接受。",
        "3. Ensembl 一对多映射保留全部候选，并按物种、reviewed 状态和序列一致性审查。",
        "4. 所有失败、重试、冲突和响应哈希写入结果文件，不静默丢弃。",
        "",
    ])
    return "\n".join(lines)


def prepare_worklists(
        alignment_dir, output_dir, minimum_smiles=0.70, minimum_sequence=0.95):
    alignment_dir = Path(alignment_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compound_source = alignment_dir / "compound_alignment.csv"
    protein_source = alignment_dir / "protein_alignment.csv"
    compounds = build_worklist(read_rows(compound_source), "compound")
    proteins = build_worklist(read_rows(protein_source), "protein")
    report = summarize(compounds, proteins, minimum_smiles, minimum_sequence)
    report.update({
        "created_at": datetime.now().astimezone().isoformat(),
        "network_accessed": False,
        "compound_source": str(compound_source),
        "compound_source_sha256": sha256_file(compound_source),
        "protein_source": str(protein_source),
        "protein_source_sha256": sha256_file(protein_source),
    })
    write_csv(output_dir / "compound_worklist.csv", compounds, COMPOUND_FIELDS)
    write_csv(output_dir / "protein_worklist.csv", proteins, PROTEIN_FIELDS)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = build_markdown(report)
    (output_dir / "report.md").write_text(markdown + "\n", encoding="utf-8")
    return report, markdown


def main():
    args = parse_args()
    _, markdown = prepare_worklists(
        args.alignment_dir, args.output_dir,
        args.minimum_smiles_coverage, args.minimum_sequence_coverage
    )
    print(markdown)
    print("Results written to: %s" % Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
