#!/usr/bin/env python3
"""Small, deterministic TCMSP/ETCM molecular-attribute feasibility audit."""

import argparse
import csv
import html
import json
import math
import re
import sys
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.enrich_symmap_attributes import (
    FORMULA_VERIFIED_STATUSES,
    HttpClient,
    ensure_pubchem_property,
    enrich_proteins,
    fetch_pubchem_candidates,
    formula_match,
    parse_uniprot_tsv,
    select_uniprot_candidate,
    sha256_bytes,
    write_bytes,
)


TCMSP_DIR = REPOSITORY_ROOT / "dataset" / "TCMSP"
ETCM_DIR = REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_mention10"
UNIPROT_BASE = "https://rest.uniprot.org"
COMPOUND_FIELDS = (
    "dataset", "entity_id", "degree", "source_name", "source_formula",
    "source_identifier", "source_inchikey", "pubchem_cid", "pubchem_title",
    "canonical_smiles", "pubchem_formula", "pubchem_inchikey",
    "identity_status", "formula_status", "resolution_status", "source_url",
)
PROTEIN_FIELDS = (
    "dataset", "entity_id", "degree", "source_name", "source_identifier",
    "uniprot_accession", "sequence", "organism_id", "reviewed",
    "resolution_status", "source_url",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument(
        "--dataset", choices=("all", "tcmsp", "etcm"), default="all"
    )
    parser.add_argument(
        "--entity", choices=("all", "compound", "protein"), default="all"
    )
    parser.add_argument(
        "--output-dir", default="results/tcmsp_etcm_attribute_feasibility"
    )
    parser.add_argument(
        "--cache-dir", default="results/tcmsp_etcm_attribute_feasibility/cache"
    )
    parser.add_argument("--minimum-smiles", type=float, default=0.70)
    parser.add_argument("--minimum-sequence", type=float, default=0.95)
    parser.add_argument("--minimum-identity", type=float, default=0.95)
    parser.add_argument("--organism-id", default="9606")
    parser.add_argument("--delay", type=float, default=0.20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relation_degrees(path):
    compounds = Counter()
    proteins = Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                compounds[parts[0]] += 1
                proteins[parts[1]] += 1
    return compounds, proteins


def id_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def systematic_degree_sample(degrees, sample_size):
    population = sorted(degrees, key=lambda value: (degrees[value], id_key(value)))
    if sample_size <= 0:
        raise ValueError("sample-size must be positive")
    if sample_size >= len(population):
        return population
    return [
        population[min(len(population) - 1, int((index + 0.5) * len(population) / sample_size))]
        for index in range(sample_size)
    ]


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def metric(successes, total, threshold):
    lower, upper = wilson_interval(successes, total)
    rate = successes / total if total else 0.0
    if not total:
        decision = "no_data"
    elif lower >= threshold:
        decision = "go"
    elif upper < threshold:
        decision = "no_go"
    elif rate >= threshold:
        decision = "promising_inconclusive"
    else:
        decision = "at_risk_inconclusive"
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "threshold": threshold,
        "decision": decision,
    }


def clean_html(value):
    value = re.sub(r"<br\s*/?>", "; ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip(" ;")


def table_value(page, label):
    pattern = (
        r"<th[^>]*>\s*%s\s*</th>\s*<td[^>]*>(.*?)</td>" % re.escape(label)
    )
    match = re.search(pattern, page, flags=re.IGNORECASE | re.DOTALL)
    return clean_html(match.group(1)) if match else ""


def cached_request(client, cache_path, url, accept="text/html"):
    if not cache_path.exists():
        if client.offline:
            return b""
        payload = client.request(url, headers={"Accept": accept}, allow_not_found=True)
        if payload is None:
            payload = b""
        write_bytes(cache_path, payload)
    return cache_path.read_bytes()


def print_progress(label, index, total):
    if index == 1 or index == total or index % 10 == 0:
        print("%s: %d/%d" % (label, index, total), flush=True)


def tcmsp_page(entity_type, entity_id, cache_dir, client):
    if entity_type == "compound":
        page = "molecule"
        parameter = "qn"
    else:
        page = "target"
        parameter = "qt"
    url = "https://tcmsp-e.com/%s.php?%s=%s" % (
        page, parameter, urllib.parse.quote(str(entity_id), safe="")
    )
    path = cache_dir / "tcmsp" / ("%s_%s.html" % (page, entity_id))
    payload = cached_request(client, path, url)
    return payload.decode("utf-8", errors="replace"), url


def normalize_inchikey(value):
    return re.sub(r"\s+", "", str(value or "")).upper()


def tcmsp_compound_rows(sample, degrees, cache_dir, client):
    rows = []
    for index, entity_id in enumerate(sample, 1):
        page, url = tcmsp_page("compound", entity_id, cache_dir, client)
        source_name = table_value(page, "Molecule name")
        source_inchikey = table_value(page, "InChIKey")
        pubchem_cid = table_value(page, "Pubchem Cid")
        if not pubchem_cid.isdigit() and source_inchikey:
            cids, _ = fetch_pubchem_candidates(
                "inchikey", source_inchikey, cache_dir, client
            )
            pubchem_cid = cids[0] if len(cids) == 1 else ""
        item = None
        if pubchem_cid.isdigit():
            item, _ = ensure_pubchem_property(pubchem_cid, cache_dir, client)
        pubchem_inchikey = item.get("InChIKey", "") if item else ""
        source_key = normalize_inchikey(source_inchikey)
        fetched_key = normalize_inchikey(pubchem_inchikey)
        identity_status = (
            "exact_inchikey" if source_key and source_key == fetched_key
            else "missing_source_inchikey" if not source_key
            else "missing_pubchem_record" if not fetched_key
            else "inchikey_mismatch"
        )
        rows.append({
            "dataset": "TCMSP", "entity_id": entity_id,
            "degree": degrees[entity_id], "source_name": source_name,
            "source_formula": "", "source_identifier": entity_id,
            "source_inchikey": source_inchikey, "pubchem_cid": pubchem_cid,
            "pubchem_title": item.get("Title", "") if item else "",
            "canonical_smiles": item.get("ConnectivitySMILES", "") if item else "",
            "pubchem_formula": item.get("MolecularFormula", "") if item else "",
            "pubchem_inchikey": pubchem_inchikey,
            "identity_status": identity_status, "formula_status": "not_available",
            "resolution_status": (
                "resolved_verified" if item and identity_status == "exact_inchikey"
                else "unresolved"
            ),
            "source_url": url,
        })
        print_progress("TCMSP compound sample", index, len(sample))
    return rows


def etcm_compound_rows(sample, degrees, mapping, cache_dir, client):
    rows = []
    for index, entity_id in enumerate(sample, 1):
        source = mapping[entity_id]
        source_name = source.get("ingredient_names", "").split("|", 1)[0].strip()
        source_formula = source.get("molecular_formula", "")
        cids, _ = fetch_pubchem_candidates("name", source_name, cache_dir, client)
        candidates = []
        for cid in cids[:10]:
            item, _ = ensure_pubchem_property(cid, cache_dir, client)
            if item:
                candidates.append((cid, item, formula_match(
                    source_formula, item.get("MolecularFormula", "")
                )))
        verified = [item for item in candidates if item[2] in FORMULA_VERIFIED_STATUSES]
        selected = verified[0] if len(verified) == 1 else None
        rows.append({
            "dataset": "ETCM2.0-mention10", "entity_id": entity_id,
            "degree": degrees[entity_id], "source_name": source_name,
            "source_formula": source_formula,
            "source_identifier": source.get("tcmip_ids", ""),
            "source_inchikey": "", "pubchem_cid": selected[0] if selected else "",
            "pubchem_title": selected[1].get("Title", "") if selected else "",
            "canonical_smiles": (
                selected[1].get("ConnectivitySMILES", "") if selected else ""
            ),
            "pubchem_formula": (
                selected[1].get("MolecularFormula", "") if selected else ""
            ),
            "pubchem_inchikey": selected[1].get("InChIKey", "") if selected else "",
            "identity_status": (
                "formula_verified" if selected else
                "ambiguous_formula_verified" if len(verified) > 1 else
                "formula_conflict" if candidates else "not_found"
            ),
            "formula_status": selected[2] if selected else (
                verified[0][2] if verified else
                candidates[0][2] if candidates else "not_available"
            ),
            "resolution_status": "resolved_verified" if selected else "unresolved",
            "source_url": "https://pubchem.ncbi.nlm.nih.gov/#query=" + urllib.parse.quote(source_name),
        })
        print_progress("ETCM compound sample", index, len(sample))
    return rows


def normalize_name(value):
    value = re.sub(r"\([^)]*\)", " ", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def uniprot_name_cache(cache_dir, target_name):
    key = sha256_bytes(target_name.encode("utf-8"))
    return cache_dir / "uniprot_name" / (key + ".tsv")


def search_uniprot_name(target_name, cache_dir, client, organism_id):
    path = uniprot_name_cache(cache_dir, target_name)
    query = '(protein_name:"%s") AND (organism_id:%s)' % (
        target_name.replace('"', ""), organism_id
    )
    fields = (
        "accession,id,reviewed,protein_name,gene_names,organism_id,"
        "organism_name,length,sequence"
    )
    url = "%s/uniprotkb/search?%s" % (UNIPROT_BASE, urllib.parse.urlencode({
        "query": query, "format": "tsv", "fields": fields, "size": 25,
    }))
    payload = cached_request(client, path, url, accept="text/tab-separated-values")
    candidates = parse_uniprot_tsv(payload)
    target = normalize_name(target_name)
    exact = [
        row for row in candidates
        if normalize_name(row.get("protein_name", "")) == target
        or normalize_name(row.get("protein_name", "")).startswith(target + " ")
    ]
    return select_uniprot_candidate(exact, organism_id), url


def tcmsp_protein_rows(sample, degrees, cache_dir, client, organism_id):
    rows = []
    for index, entity_id in enumerate(sample, 1):
        page, url = tcmsp_page("protein", entity_id, cache_dir, client)
        source_name = table_value(page, "Target name")
        selected = None
        status = "missing_target_name"
        if source_name:
            (selected, status, _), _ = search_uniprot_name(
                source_name, cache_dir, client, organism_id
            )
        rows.append({
            "dataset": "TCMSP", "entity_id": entity_id,
            "degree": degrees[entity_id], "source_name": source_name,
            "source_identifier": table_value(page, "Target ID") or entity_id,
            "uniprot_accession": selected.get("accession", "") if selected else "",
            "sequence": selected.get("sequence", "") if selected else "",
            "organism_id": selected.get("organism_id", "") if selected else "",
            "reviewed": selected.get("reviewed", "") if selected else "",
            "resolution_status": status,
            "source_url": url,
        })
        print_progress("TCMSP protein sample", index, len(sample))
    return rows


def etcm_protein_rows(
        sample, degrees, mapping, cache_dir, client, organism_id):
    worklist = []
    for entity_id in sample:
        source = mapping[entity_id]
        worklist.append({
            "local_entity_id": entity_id,
            "enrichment_route": "direct_uniprot",
            "query_identifier": source.get("uniprot_accessions", "").split("|", 1)[0],
        })
    enriched = enrich_proteins(
        worklist, cache_dir, client, organism_id, poll_interval=1.0,
        poll_attempts=100,
    )
    by_id = {row["entity_id"]: row for row in enriched}
    rows = []
    for entity_id in sample:
        source = mapping[entity_id]
        item = by_id[entity_id]
        rows.append({
            "dataset": "ETCM2.0-mention10", "entity_id": entity_id,
            "degree": degrees[entity_id],
            "source_name": source.get("target_names", ""),
            "source_identifier": source.get("uniprot_accessions", ""),
            "uniprot_accession": item.get("uniprot_accession", ""),
            "sequence": item.get("sequence", ""),
            "organism_id": item.get("organism_id", ""),
            "reviewed": item.get("reviewed", ""),
            "resolution_status": item.get("resolution_status", ""),
            "source_url": "https://www.uniprot.org/uniprotkb/" + urllib.parse.quote(
                item.get("uniprot_accession", ""), safe=""
            ),
        })
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_dataset(compounds, proteins, args):
    verified_compounds = sum(
        row["resolution_status"] == "resolved_verified" for row in compounds
    )
    identity_checkable = [
        row for row in compounds
        if row["identity_status"] not in {"missing_source_inchikey", "not_found"}
    ]
    identity_verified = sum(
        row["identity_status"] in {"exact_inchikey", "formula_verified"}
        for row in identity_checkable
    )
    sequences = sum(bool(row["sequence"]) for row in proteins)
    return {
        "compound_sample_size": len(compounds),
        "protein_sample_size": len(proteins),
        "verified_smiles": metric(
            verified_compounds, len(compounds), args.minimum_smiles
        ),
        "identity_verification": metric(
            identity_verified, len(identity_checkable), args.minimum_identity
        ),
        "protein_sequence": metric(
            sequences, len(proteins), args.minimum_sequence
        ),
        "compound_status_counts": dict(sorted(Counter(
            row["resolution_status"] for row in compounds
        ).items())),
        "protein_status_counts": dict(sorted(Counter(
            row["resolution_status"] for row in proteins
        ).items())),
    }


def overall_decision(dataset_reports):
    metrics = [
        report[key] for report in dataset_reports.values()
        for key in ("verified_smiles", "identity_verification", "protein_sequence")
        if report[key]["total"]
    ]
    if any(item["decision"] == "no_go" for item in metrics):
        return "stop_shared_multimodal_enrichment"
    if metrics and all(item["rate"] >= item["threshold"] for item in metrics):
        return "promising_for_full_enrichment"
    return "inconclusive_extend_or_stop"


def build_markdown(report):
    lines = [
        "# TCMSP 与 ETCM 属性补全小样本可行性审计", "",
        "总体判定：**%s**" % report["decision"], "",
        "抽样方法：按 C-P degree 排序后的确定性系统抽样；区间为 Wilson 95% CI。", "",
        "| 数据集 | 指标 | 成功/样本 | 覆盖率 | 95% CI | 门槛 | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    labels = {
        "verified_smiles": "可信 SMILES",
        "identity_verification": "身份核验",
        "protein_sequence": "蛋白序列",
    }
    for dataset, dataset_report in report["datasets"].items():
        for key, label in labels.items():
            item = dataset_report[key]
            lines.append(
                "| %s | %s | %d/%d | %.2f%% | %.2f%%–%.2f%% | %.0f%% | %s |" % (
                    dataset, label, item["successes"], item["total"],
                    item["rate"] * 100, item["wilson_95_lower"] * 100,
                    item["wilson_95_upper"] * 100, item["threshold"] * 100,
                    item["decision"],
                )
            )
    lines.extend([
        "", "## 解释边界", "",
        "- TCMSP 成分身份以官网 InChIKey 与 PubChem InChIKey 一致为准。",
        "- ETCM 成分身份以原始分子式与唯一 PubChem 候选一致为准。",
        "- TCMSP 蛋白只接受唯一的人类 UniProt 名称匹配；模糊名称按未解析处理。",
        "- 本审计只决定是否值得继续全量补全，不生成训练输入。", "",
    ])
    return "\n".join(lines)


def run(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = HttpClient(
        timeout=args.timeout, attempts=args.attempts, delay=args.delay,
        offline=args.offline,
    )
    datasets = {}
    selected_datasets = ("tcmsp", "etcm") if args.dataset == "all" else (args.dataset,)
    for dataset in selected_datasets:
        if dataset == "tcmsp":
            cp_degrees, protein_degrees = relation_degrees(
                TCMSP_DIR / "compound-protein.txt"
            )
            compound_mapping = protein_mapping = None
            display_name = "TCMSP"
        else:
            cp_degrees, protein_degrees = relation_degrees(ETCM_DIR / "C_P.txt")
            compound_mapping = {
                row["compound_id"]: row for row in read_csv(
                    ETCM_DIR / "mappings" / "compound_id_map.csv"
                )
            }
            protein_mapping = {
                row["protein_id"]: row for row in read_csv(
                    ETCM_DIR / "mappings" / "protein_id_map.csv"
                )
            }
            display_name = "ETCM2.0-mention10"
        compounds = []
        proteins = []
        if args.entity in {"all", "compound"}:
            sample = systematic_degree_sample(cp_degrees, args.sample_size)
            compounds = (
                tcmsp_compound_rows(sample, cp_degrees, cache_dir, client)
                if dataset == "tcmsp" else
                etcm_compound_rows(
                    sample, cp_degrees, compound_mapping, cache_dir, client
                )
            )
            write_csv(
                output_dir / (dataset + "_compound_sample.csv"),
                compounds, COMPOUND_FIELDS,
            )
        if args.entity in {"all", "protein"}:
            sample = systematic_degree_sample(protein_degrees, args.sample_size)
            proteins = (
                tcmsp_protein_rows(
                    sample, protein_degrees, cache_dir, client, args.organism_id
                ) if dataset == "tcmsp" else
                etcm_protein_rows(
                    sample, protein_degrees, protein_mapping, cache_dir, client,
                    args.organism_id,
                )
            )
            write_csv(
                output_dir / (dataset + "_protein_sample.csv"),
                proteins, PROTEIN_FIELDS,
            )
        datasets[display_name] = summarize_dataset(compounds, proteins, args)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "audit_type": "tcmsp_etcm_attribute_feasibility_sample",
        "offline": args.offline,
        "sample_size_per_entity": args.sample_size,
        "sampling_method": "systematic_midpoint_after_cp_degree_then_entity_id_sort",
        "thresholds": {
            "minimum_smiles": args.minimum_smiles,
            "minimum_identity": args.minimum_identity,
            "minimum_sequence": args.minimum_sequence,
        },
        "datasets": datasets,
    }
    report["decision"] = overall_decision(datasets)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = build_markdown(report)
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Results written to: %s" % output_dir)
    return report


def main():
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
