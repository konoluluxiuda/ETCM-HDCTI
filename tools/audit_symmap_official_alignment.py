#!/usr/bin/env python3
"""Audit whether official SymMap exports recover local anonymous entity IDs."""

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare local SymMap C-P entity IDs with official SMIT/SMTT XLSX "
            "exports without modifying the training dataset."
        )
    )
    parser.add_argument("--smit", required=True, help="Official SMIT XLSX file")
    parser.add_argument("--smtt", required=True, help="Official SMTT XLSX file")
    parser.add_argument(
        "--dataset-dir", default=str(REPOSITORY_ROOT / "dataset" / "Symmap")
    )
    parser.add_argument(
        "--output-dir", default="results/symmap_official_alignment"
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_header(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def normalize_identifier(value):
    value = str(value or "").strip()
    if re.fullmatch(r"[+-]?\d+\.0+", value):
        return value.split(".", 1)[0]
    return value


def numeric_identifier(value, expected_prefix):
    value = normalize_identifier(value)
    if value.isdigit():
        return str(int(value))
    pattern = r"^%s[\s:_-]*0*(\d+)$" % re.escape(expected_prefix)
    match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
    return str(int(match.group(1))) if match else ""


def read_shared_strings(archive):
    try:
        source = archive.open("xl/sharedStrings.xml")
    except KeyError:
        return []
    strings = []
    with source:
        for _, element in ElementTree.iterparse(source, events=("end",)):
            if element.tag == XML_NS + "si":
                strings.append("".join(
                    node.text or "" for node in element.iter(XML_NS + "t")
                ))
                element.clear()
    return strings


def first_worksheet_path(archive):
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    first_sheet = workbook.find("%ssheets/%ssheet" % (XML_NS, XML_NS))
    if first_sheet is None:
        raise ValueError("XLSX workbook has no worksheets")
    relation_id = first_sheet.attrib[REL_NS + "id"]
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships.findall(PACKAGE_REL_NS + "Relationship"):
        if relationship.attrib.get("Id") == relation_id:
            target = relationship.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else "xl/" + target
    raise ValueError("Could not resolve first worksheet relationship")


def column_index(reference):
    letters = re.match(r"[A-Z]+", reference or "")
    if not letters:
        return 0
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(XML_NS + "t")
        )
    value = cell.find(XML_NS + "v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def read_first_sheet(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        worksheet_path = first_worksheet_path(archive)
        with archive.open(worksheet_path) as source:
            for _, element in ElementTree.iterparse(source, events=("end",)):
                if element.tag != XML_NS + "row":
                    continue
                values = {}
                for cell in element.findall(XML_NS + "c"):
                    values[column_index(cell.attrib.get("r", ""))] = cell_value(
                        cell, shared_strings
                    )
                if values:
                    width = max(values) + 1
                    rows.append([values.get(index, "") for index in range(width)])
                element.clear()
    if not rows:
        raise ValueError("XLSX worksheet is empty: %s" % path)
    headers = [str(value).strip() for value in rows[0]]
    records = []
    for row_number, values in enumerate(rows[1:], 2):
        record = {
            header: normalize_identifier(values[index] if index < len(values) else "")
            for index, header in enumerate(headers)
            if header
        }
        record["__row_number__"] = row_number
        records.append(record)
    return headers, records


def find_column(headers, aliases):
    normalized = {normalize_header(header): header for header in headers}
    for alias in aliases:
        if normalize_header(alias) in normalized:
            return normalized[normalize_header(alias)]
    raise ValueError(
        "Missing required column; expected one of %s, found %s"
        % (", ".join(aliases), ", ".join(headers))
    )


def optional_column(headers, aliases):
    try:
        return find_column(headers, aliases)
    except ValueError:
        return None


def read_relation_ids(path):
    compounds = set()
    proteins = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                compounds.add(parts[0])
                proteins.add(parts[1])
    return compounds, proteins


def read_local_ids(path):
    ids = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0].strip().isdigit():
                ids.add(str(int(row[0].strip())))
    return ids


def unique_index(records, id_column, key_function):
    candidates = {}
    for record in records:
        official_id = normalize_identifier(record.get(id_column, ""))
        key = key_function(official_id)
        if key:
            candidates.setdefault(key, []).append(record)
    return {
        key: values[0]
        for key, values in candidates.items()
        if len(values) == 1
    }, sum(len(values) > 1 for values in candidates.values())


def align_ids(local_ids, records, id_column, prefix):
    exact, exact_collisions = unique_index(
        records, id_column, lambda value: normalize_identifier(value).casefold()
    )
    numeric, numeric_collisions = unique_index(
        records, id_column, lambda value: numeric_identifier(value, prefix)
    )
    matched = []
    unmatched = []
    for local_id in sorted(local_ids, key=lambda value: int(value)):
        record = exact.get(local_id.casefold())
        method = "exact"
        if record is None:
            record = numeric.get(str(int(local_id)))
            method = "unique_numeric_suffix"
        if record is None:
            unmatched.append(local_id)
        else:
            matched.append((local_id, method, record))
    return {
        "matched": matched,
        "unmatched": unmatched,
        "exact_collisions": exact_collisions,
        "numeric_suffix_collisions": numeric_collisions,
    }


def ratio(part, total):
    return float(part) / float(total) if total else 0.0


def audit_entity(
        entity_type, prefix, headers, records, id_aliases, used_ids, all_ids):
    id_column = find_column(headers, id_aliases)
    used_alignment = align_ids(used_ids, records, id_column, prefix)
    all_alignment = align_ids(all_ids, records, id_column, prefix)
    return {
        "entity_type": entity_type,
        "official_id_column": id_column,
        "official_rows": len(records),
        "official_unique_ids": len({
            normalize_identifier(row.get(id_column, "")) for row in records
            if normalize_identifier(row.get(id_column, ""))
        }),
        "local_used_ids": len(used_ids),
        "local_all_ids": len(all_ids),
        "used_matched_ids": len(used_alignment["matched"]),
        "used_coverage": ratio(len(used_alignment["matched"]), len(used_ids)),
        "all_matched_ids": len(all_alignment["matched"]),
        "all_coverage": ratio(len(all_alignment["matched"]), len(all_ids)),
        "exact_collisions": used_alignment["exact_collisions"],
        "numeric_suffix_collisions": used_alignment["numeric_suffix_collisions"],
        "used_alignment": used_alignment,
        "all_alignment": all_alignment,
    }


def select_value(record, headers, aliases):
    column = optional_column(headers, aliases)
    return record.get(column, "") if column else ""


def standardized_row(entity_type, local_id, method, record, headers, id_column):
    base = {
        "entity_type": entity_type,
        "local_entity_id": local_id,
        "official_entity_id": record.get(id_column, ""),
        "source_entity_id": record.get(id_column, ""),
        "source_identifier_namespace": (
            "SymMap:Mol_id" if entity_type == "compound" else "SymMap:Gene_id"
        ),
        "match_method": method,
        "official_row_number": record.get("__row_number__", ""),
    }
    if entity_type == "compound":
        base.update({
            "canonical_name": select_value(
                record, headers, ("Molecule_name", "Ingredient_name", "Name")
            ),
            "molecular_formula": select_value(
                record, headers,
                ("Molecular_formula", "Molecule_formula", "Formula")
            ),
            "pubchem_id": select_value(
                record, headers, ("PubChem_id", "PubChem CID", "PubChem")
            ),
            "cas_id": select_value(record, headers, ("CAS_id", "CAS")),
            "tcmsp_id": select_value(record, headers, ("TCMSP_id",)),
            "uniprot_id": "",
            "gene_symbol": "",
            "protein_name": "",
            "ensembl_id": "",
            "ncbi_id": "",
            "genbank_protein_id": "",
        })
        base["canonical_identifier"] = base["pubchem_id"]
    else:
        base.update({
            "canonical_name": select_value(
                record, headers, ("Gene_name", "Protein_name", "Gene_symbol")
            ),
            "molecular_formula": "",
            "pubchem_id": "",
            "cas_id": "",
            "tcmsp_id": select_value(record, headers, ("TCMSP_id",)),
            "uniprot_id": select_value(
                record, headers, ("UniProt_id", "UniProt ID", "UniProt")
            ),
            "gene_symbol": select_value(record, headers, ("Gene_symbol",)),
            "protein_name": select_value(record, headers, ("Protein_name",)),
            "ensembl_id": select_value(record, headers, ("Ensembl_id",)),
            "ncbi_id": select_value(record, headers, ("NCBI_id",)),
            "genbank_protein_id": select_value(
                record, headers, ("GenBank_Protein_id", "GenBank Protein")
            ),
        })
        base["canonical_identifier"] = base["uniprot_id"]
    return base


def metadata_coverage(audit, headers):
    rows = [
        standardized_row(
            audit["entity_type"], local_id, method, record, headers,
            audit["official_id_column"]
        )
        for local_id, method, record in audit["used_alignment"]["matched"]
    ]
    fields = (
        ("canonical_name", "molecular_formula", "pubchem_id", "cas_id", "tcmsp_id")
        if audit["entity_type"] == "compound" else
        (
            "canonical_name", "gene_symbol", "protein_name", "uniprot_id",
            "ensembl_id", "ncbi_id", "genbank_protein_id"
        )
    )
    return {
        field: {
            "count": sum(bool(str(row.get(field, "")).strip()) for row in rows),
            "coverage": ratio(
                sum(bool(str(row.get(field, "")).strip()) for row in rows),
                len(rows),
            ),
        }
        for field in fields
    }


def serializable_summary(audit):
    return {key: value for key, value in audit.items()
            if key not in {"used_alignment", "all_alignment"}}


def write_alignment(path, audit, headers):
    rows = [
        standardized_row(
            audit["entity_type"], local_id, method, record, headers,
            audit["official_id_column"]
        )
        for local_id, method, record in audit["used_alignment"]["matched"]
    ]
    fieldnames = (
        "entity_type", "local_entity_id", "official_entity_id",
        "source_entity_id", "source_identifier_namespace",
        "canonical_identifier", "match_method",
        "official_row_number", "canonical_name", "molecular_formula",
        "pubchem_id", "cas_id", "tcmsp_id", "uniprot_id", "gene_symbol",
        "protein_name", "ensembl_id", "ncbi_id", "genbank_protein_id"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_unmatched(path, audit):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("entity_type", "local_entity_id", "scope"))
        for local_id in audit["used_alignment"]["unmatched"]:
            writer.writerow((audit["entity_type"], local_id, "cp_used"))


def build_markdown(report):
    lines = [
        "# SymMap 官方映射覆盖审计",
        "",
        "本报告只核验本地匿名 ID 与官方导出 ID 的可追溯对应关系，"
        "不修改训练数据。数字后缀匹配仅在官方 ID 后缀唯一时接受。",
        "",
        "| 实体 | C-P 使用数 | 已匹配 | 使用覆盖率 | 全量本地覆盖率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("compound", "protein"):
        item = report[key]
        lines.append(
            "| %s | %s | %s | %.2f%% | %.2f%% |" % (
                item["entity_type"], item["local_used_ids"],
                item["used_matched_ids"], item["used_coverage"] * 100,
                item["all_coverage"] * 100,
            )
        )
    lines.extend([
        "",
        "## 已有生物属性",
        "",
        "| 实体 | 字段 | 数量 | 覆盖率 |",
        "|---|---|---:|---:|",
    ])
    for key in ("compound", "protein"):
        item = report[key]
        for field, coverage in item["metadata_coverage"].items():
            lines.append(
                "| %s | `%s` | %s | %.2f%% |" % (
                    item["entity_type"], field, coverage["count"],
                    coverage["coverage"] * 100,
                )
            )
    lines.extend([
        "",
        "## 决策",
        "",
        "* Compound Go 门槛：C-P 使用实体覆盖率至少 70%。",
        "* Protein Go 门槛：C-P 使用实体覆盖率至少 95%。",
        "* 当前结论：`%s`。" % report["decision"],
        "",
        "## 注意",
        "",
        "本地全量实体数与官网当前 V2 导出规模不一致。即使 C-P 使用实体覆盖达标，"
        "也必须保留官方文件哈希、匹配方法和未匹配清单，不能将当前 V2 静默当作论文预处理快照。",
        "",
    ])
    return "\n".join(lines)


def run_audit(smit_path, smtt_path, dataset_dir, output_dir):
    smit_path = Path(smit_path).expanduser().resolve()
    smtt_path = Path(smtt_path).expanduser().resolve()
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compound_ids, protein_ids = read_relation_ids(dataset_dir / "IT.txt")
    all_compounds = read_local_ids(dataset_dir / "compound_id_all.txt")
    all_proteins = read_local_ids(dataset_dir / "protein_id_all.txt")
    smit_headers, smit_records = read_first_sheet(smit_path)
    smtt_headers, smtt_records = read_first_sheet(smtt_path)

    compound = audit_entity(
        "compound", "SMIT", smit_headers, smit_records,
        ("Ingredient_id", "Ingredient ID", "SMIT_id", "Mol_id"),
        compound_ids, all_compounds
    )
    protein = audit_entity(
        "protein", "SMTT", smtt_headers, smtt_records,
        ("Target_id", "Target ID", "SMTT_id", "Gene_id"),
        protein_ids, all_proteins
    )
    compound["metadata_coverage"] = metadata_coverage(compound, smit_headers)
    protein["metadata_coverage"] = metadata_coverage(protein, smtt_headers)
    decision = (
        "go_version_aware_alignment"
        if compound["used_coverage"] >= 0.70
        and protein["used_coverage"] >= 0.95
        else "no_go_current_official_exports"
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "relation_file": str(dataset_dir / "IT.txt"),
        "relation_sha256": sha256_file(dataset_dir / "IT.txt"),
        "smit_file": str(smit_path),
        "smit_sha256": sha256_file(smit_path),
        "smtt_file": str(smtt_path),
        "smtt_sha256": sha256_file(smtt_path),
        "compound": serializable_summary(compound),
        "protein": serializable_summary(protein),
        "decision": decision,
    }
    write_alignment(output_dir / "compound_alignment.csv", compound, smit_headers)
    write_alignment(output_dir / "protein_alignment.csv", protein, smtt_headers)
    write_unmatched(output_dir / "compound_unmatched.csv", compound)
    write_unmatched(output_dir / "protein_unmatched.csv", protein)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        build_markdown(report), encoding="utf-8"
    )
    return report


def main():
    args = parse_args()
    report = run_audit(
        args.smit, args.smtt, args.dataset_dir, args.output_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
