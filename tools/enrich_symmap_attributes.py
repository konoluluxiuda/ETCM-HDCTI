#!/usr/bin/env python3
"""Enrich verified SymMap entities with PubChem and UniProt attributes."""

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
UNIPROT_BASE = "https://rest.uniprot.org"
USER_AGENT = "HDCTI-entity-enrichment/1.0"
PUBCHEM_PROPERTIES = (
    "ConnectivitySMILES,SMILES,MolecularFormula,InChIKey,Title"
)
COMPOUND_OUTPUT_FIELDS = (
    "entity_id", "pubchem_cid", "canonical_smiles", "isomeric_smiles",
    "molecular_formula", "source_formula", "formula_match", "inchikey",
    "title", "mapping_method", "source_identifier", "resolution_status",
    "candidate_cids", "response_sha256"
)
PROTEIN_OUTPUT_FIELDS = (
    "entity_id", "uniprot_accession", "sequence", "entry_name", "reviewed",
    "protein_name", "gene_names", "organism", "organism_id", "length",
    "mapping_method", "source_identifier", "resolution_status",
    "candidate_accessions", "response_sha256"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch auditable SymMap molecular/protein attributes with raw "
            "response caching and explicit conflict states."
        )
    )
    parser.add_argument(
        "--worklist-dir", default="results/symmap_attribute_enrichment"
    )
    parser.add_argument(
        "--output-root", default="results/multidataset_attributes/SymMap2.0"
    )
    parser.add_argument(
        "--cache-dir", default="results/symmap_attribute_enrichment/cache"
    )
    parser.add_argument(
        "--entity", choices=("compound", "protein", "all"), default="all"
    )
    parser.add_argument(
        "--compound-route", action="append", default=[],
        help="Process only the named compound enrichment route; may be repeated."
    )
    parser.add_argument(
        "--protein-route", action="append", default=[],
        help="Process only the named protein enrichment route; may be repeated."
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Limit each selected entity worklist for a smoke test; 0 means all."
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.34)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--poll-attempts", type=int, default=100)
    parser.add_argument("--organism-id", default="9606")
    parser.add_argument(
        "--include-manual-compounds", action="store_true",
        help="Fetch name-only PubChem candidates but never auto-accept them."
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use existing cache only and never access the network."
    )
    return parser.parse_args()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def chunks(values, size):
    if size <= 0:
        raise ValueError("batch-size must be positive")
    for start in range(0, len(values), size):
        yield values[start:start + size]


class HttpClient:
    def __init__(self, timeout=60.0, attempts=4, delay=0.34, offline=False):
        self.timeout = timeout
        self.attempts = attempts
        self.delay = delay
        self.offline = offline

    def request(self, url, data=None, headers=None, allow_not_found=False):
        if self.offline:
            raise RuntimeError("Network access disabled in offline mode: %s" % url)
        request_headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        request_headers.update(headers or {})
        request = urllib.request.Request(url, data=data, headers=request_headers)
        error = None
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                if self.delay:
                    time.sleep(self.delay)
                return payload
            except urllib.error.HTTPError as exc:
                if allow_not_found and exc.code == 404:
                    return None
                error = exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                wait = float(retry_after) if retry_after else 2.0 ** attempt
            except Exception as exc:
                error = exc
                wait = 2.0 ** attempt
            if attempt + 1 < self.attempts:
                time.sleep(wait)
        raise RuntimeError("Failed request %s: %s" % (url, error))


FORMULA_VERIFIED_STATUSES = {
    "exact",
    "composition_equivalent",
    "composition_match_charge_diff",
}


def normalize_formula(value):
    return re.sub(r"\s+", "", str(value or ""))


def parse_formula(value):
    """Parse a simple molecular formula without normalizing salts or hydrates."""
    normalized = normalize_formula(value)
    if not normalized:
        return None
    charge = ""
    charge_match = re.search(r"([+-])(\d*)$", normalized)
    if charge_match:
        charge = charge_match.group(0)
        normalized = normalized[:charge_match.start()]
    if not normalized:
        return None
    composition = {}
    cursor = 0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", normalized):
        if match.start() != cursor:
            return None
        element, raw_count = match.groups()
        count = int(raw_count) if raw_count else 1
        if count <= 0:
            return None
        composition[element] = composition.get(element, 0) + count
        cursor = match.end()
    if cursor != len(normalized) or not composition:
        return None
    return composition, charge


def split_identifiers(value):
    return [
        token.strip() for token in re.split(r"[|;]", str(value or ""))
        if token.strip()
    ]


def formula_match(source_formula, fetched_formula):
    if not str(source_formula or "").strip():
        return "not_available"
    if not str(fetched_formula or "").strip():
        return "not_available"
    fetched_normalized = normalize_formula(fetched_formula)
    fetched_parsed = parse_formula(fetched_formula)
    if fetched_parsed is None:
        return "invalid_fetched_formula"
    valid_source = False
    best_status = None
    status_priority = {
        "exact": 3,
        "composition_equivalent": 2,
        "composition_match_charge_diff": 1,
    }
    for source_variant in split_identifiers(source_formula):
        if normalize_formula(source_variant).casefold() == fetched_normalized.casefold():
            return "exact"
        source_parsed = parse_formula(source_variant)
        if source_parsed is None:
            continue
        valid_source = True
        if source_parsed[0] == fetched_parsed[0]:
            status = (
                "composition_equivalent"
                if source_parsed[1] == fetched_parsed[1]
                else "composition_match_charge_diff"
            )
        else:
            continue
        if best_status is None or status_priority[status] > status_priority[best_status]:
            best_status = status
    if best_status:
        return best_status
    return "mismatch" if valid_source else "invalid_source_formula"


def pubchem_property_map(payload):
    document = json.loads(payload.decode("utf-8"))
    properties = document.get("PropertyTable", {}).get("Properties", [])
    return {str(item["CID"]): item for item in properties if item.get("CID")}


def pubchem_candidate_cids(payload):
    document = json.loads(payload.decode("utf-8"))
    return [str(value) for value in document.get("IdentifierList", {}).get("CID", [])]


def pubchem_property_cache(cache_dir, cid):
    return cache_dir / "pubchem" / "properties" / ("cid_%s.json" % cid)


def fetch_pubchem_property_batches(rows, cache_dir, client, batch_size):
    requested = sorted({
        str(row["query_identifier"]).strip()
        for row in rows if row["enrichment_route"] == "direct_pubchem_cid"
        and str(row.get("query_identifier", "")).strip().isdigit()
        and not pubchem_property_cache(cache_dir, row["query_identifier"]).exists()
    }, key=int)
    if client.offline:
        return
    batches = list(chunks(requested, batch_size))
    for batch_index, batch in enumerate(batches, 1):
        print(
            "PubChem direct CID batch %d/%d (%d identifiers)" % (
                batch_index, len(batches), len(batch)
            )
        )
        endpoint = "%s/compound/cid/property/%s/JSON" % (
            PUBCHEM_BASE, PUBCHEM_PROPERTIES
        )
        body = urllib.parse.urlencode({"cid": ",".join(batch)}).encode("utf-8")
        payload = client.request(
            endpoint, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        raw_key = sha256_bytes(",".join(batch).encode("utf-8"))
        write_bytes(
            cache_dir / "pubchem" / "raw" / ("cid_batch_%s.json" % raw_key),
            payload,
        )
        properties = pubchem_property_map(payload)
        for cid in batch:
            item = properties.get(cid)
            if item is not None:
                write_bytes(
                    pubchem_property_cache(cache_dir, cid),
                    json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                )


def ensure_pubchem_property(cid, cache_dir, client):
    path = pubchem_property_cache(cache_dir, cid)
    if not path.exists():
        if client.offline:
            return None, ""
        endpoint = "%s/compound/cid/%s/property/%s/JSON" % (
            PUBCHEM_BASE, urllib.parse.quote(cid, safe=""), PUBCHEM_PROPERTIES
        )
        payload = client.request(endpoint, allow_not_found=True)
        if payload is None:
            return None, ""
        raw_key = sha256_bytes(("cid\n" + cid).encode("utf-8"))
        write_bytes(
            cache_dir / "pubchem" / "raw" / ("cid_%s.json" % raw_key),
            payload,
        )
        properties = pubchem_property_map(payload)
        item = properties.get(str(cid))
        if item is None:
            return None, sha256_bytes(payload)
        write_bytes(
            path,
            json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8")), sha256_bytes(payload)


def pubchem_candidate_cache(cache_dir, namespace, identifier):
    key = sha256_bytes((namespace + "\n" + identifier).encode("utf-8"))
    return cache_dir / "pubchem" / "candidates" / (key + ".json")


def fetch_pubchem_candidates(namespace, identifier, cache_dir, client):
    path = pubchem_candidate_cache(cache_dir, namespace, identifier)
    if not path.exists():
        if client.offline:
            return [], ""
        endpoint = "%s/compound/%s/%s/cids/JSON" % (
            PUBCHEM_BASE, namespace,
            urllib.parse.quote(identifier, safe=""),
        )
        payload = client.request(endpoint, allow_not_found=True)
        if payload is not None:
            raw_key = sha256_bytes(
                (namespace + "\n" + identifier).encode("utf-8")
            )
            write_bytes(
                cache_dir / "pubchem" / "raw" / ("lookup_%s.json" % raw_key),
                payload,
            )
        wrapper = {
            "namespace": namespace,
            "identifier": identifier,
            "payload": json.loads(payload.decode("utf-8")) if payload else None,
        }
        write_bytes(
            path,
            json.dumps(wrapper, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
    raw = path.read_bytes()
    wrapper = json.loads(raw.decode("utf-8"))
    payload = wrapper.get("payload")
    if payload is None:
        return [], sha256_bytes(raw)
    cids = [
        str(value) for value in payload.get("IdentifierList", {}).get("CID", [])
    ]
    return cids, sha256_bytes(raw)


def blank_compound(row):
    output = {field: "" for field in COMPOUND_OUTPUT_FIELDS}
    output.update({
        "entity_id": row["local_entity_id"],
        "source_formula": row.get("molecular_formula", ""),
        "mapping_method": row.get("enrichment_route", ""),
        "source_identifier": row.get("query_identifier", ""),
        "resolution_status": "pending",
    })
    return output


def apply_pubchem_property(output, item, response_sha):
    output.update({
        "pubchem_cid": str(item.get("CID", "")),
        "canonical_smiles": item.get("ConnectivitySMILES", ""),
        "isomeric_smiles": item.get("SMILES", ""),
        "molecular_formula": item.get("MolecularFormula", ""),
        "formula_match": formula_match(
            output.get("source_formula"), item.get("MolecularFormula", "")
        ),
        "inchikey": item.get("InChIKey", ""),
        "title": item.get("Title", ""),
        "response_sha256": response_sha,
    })


def enrich_compounds(rows, cache_dir, client, batch_size, include_manual=False):
    fetch_pubchem_property_batches(rows, cache_dir, client, batch_size)
    outputs = []
    for row_index, row in enumerate(rows, 1):
        output = blank_compound(row)
        route = row["enrichment_route"]
        if route == "direct_pubchem_cid":
            item, response_sha = ensure_pubchem_property(
                row["query_identifier"], cache_dir, client
            )
            if item:
                apply_pubchem_property(output, item, response_sha)
                output["resolution_status"] = "resolved_direct_cid"
            else:
                output["resolution_status"] = "not_found_direct_cid"
        elif route in {"name_formula_pubchem_lookup", "name_cas_pubchem_lookup"}:
            namespace = "name"
            identifier = row["query_identifier"]
            try:
                cids, candidate_sha = fetch_pubchem_candidates(
                    namespace, identifier, cache_dir, client
                )
            except RuntimeError:
                output["resolution_status"] = "request_error_pubchem_lookup"
                outputs.append(output)
                continue
            output["candidate_cids"] = "|".join(cids)
            output["response_sha256"] = candidate_sha
            if len(cids) == 1:
                item, property_sha = ensure_pubchem_property(cids[0], cache_dir, client)
                if item:
                    apply_pubchem_property(output, item, property_sha)
                    if route == "name_formula_pubchem_lookup":
                        output["resolution_status"] = (
                            "resolved_name_formula"
                            if output["formula_match"] in FORMULA_VERIFIED_STATUSES
                            else "conflict_formula"
                        )
                        if output["formula_match"] not in FORMULA_VERIFIED_STATUSES:
                            output["canonical_smiles"] = ""
                            output["isomeric_smiles"] = ""
                    else:
                        output["resolution_status"] = "resolved_cas"
            elif len(cids) > 1:
                output["resolution_status"] = "ambiguous_pubchem_candidates"
            else:
                output["resolution_status"] = "not_found_pubchem"
        elif route == "name_only_manual_review" and include_manual:
            try:
                cids, response_sha = fetch_pubchem_candidates(
                    "name", row["query_identifier"], cache_dir, client
                )
            except RuntimeError:
                output["resolution_status"] = "request_error_pubchem_lookup"
                outputs.append(output)
                continue
            output["candidate_cids"] = "|".join(cids)
            output["response_sha256"] = response_sha
            output["resolution_status"] = (
                "manual_review_unique_name" if len(cids) == 1
                else "manual_review_ambiguous_name" if cids
                else "not_found_pubchem"
            )
        elif route == "tcmsp_cross_reference_lookup":
            output["resolution_status"] = "pending_tcmsp_cross_reference"
        elif route == "name_only_manual_review":
            output["resolution_status"] = "pending_manual_name_review"
        else:
            output["resolution_status"] = "unresolved"
        outputs.append(output)
        if row_index % 100 == 0 or row_index == len(rows):
            print("Compound enrichment progress: %d/%d" % (row_index, len(rows)))
    return outputs


def cache_key(prefix, identifiers):
    digest = sha256_bytes((prefix + "\n" + "\n".join(identifiers)).encode("utf-8"))
    return "%s_%s" % (re.sub(r"[^A-Za-z0-9]+", "_", prefix), digest)


def submit_uniprot_mapping(identifiers, from_database, client):
    body = urllib.parse.urlencode({
        "ids": ",".join(identifiers),
        "from": from_database,
        "to": "UniProtKB",
    }).encode("utf-8")
    payload = client.request(
        UNIPROT_BASE + "/idmapping/run", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return json.loads(payload.decode("utf-8"))["jobId"]


def wait_for_uniprot_job(job_id, client, poll_interval, poll_attempts):
    endpoint = UNIPROT_BASE + "/idmapping/status/" + job_id
    for _ in range(poll_attempts):
        payload = client.request(endpoint)
        status = json.loads(payload.decode("utf-8"))
        job_status = status.get("jobStatus")
        if job_status == "FAILED":
            raise RuntimeError("UniProt mapping job failed: %s" % job_id)
        if job_status not in {"NEW", "RUNNING"}:
            return status
        time.sleep(poll_interval)
    raise RuntimeError("UniProt mapping job timed out: %s" % job_id)


def fetch_uniprot_mapping_tsv(job_id, client):
    fields = (
        "accession,id,reviewed,protein_name,gene_names,organism_id,"
        "organism_name,length,sequence"
    )
    endpoint = (
        UNIPROT_BASE + "/idmapping/uniprotkb/results/stream/" + job_id
        + "?format=tsv&fields=" + urllib.parse.quote(fields, safe=",")
    )
    return client.request(endpoint, headers={"Accept": "text/tab-separated-values"})


def ensure_uniprot_mapping(
        identifiers, from_database, cache_dir, client, poll_interval,
        poll_attempts):
    identifiers = sorted(set(identifiers))
    key = cache_key(from_database, identifiers)
    tsv_path = cache_dir / "uniprot" / (key + ".tsv")
    metadata_path = cache_dir / "uniprot" / (key + ".json")
    if not tsv_path.exists():
        if client.offline:
            return b"", ""
        job_id = submit_uniprot_mapping(identifiers, from_database, client)
        status = wait_for_uniprot_job(
            job_id, client, poll_interval, poll_attempts
        )
        payload = fetch_uniprot_mapping_tsv(job_id, client)
        write_bytes(tsv_path, payload)
        write_bytes(metadata_path, json.dumps({
            "job_id": job_id,
            "from_database": from_database,
            "identifier_count": len(identifiers),
            "identifiers_sha256": sha256_bytes(
                "\n".join(identifiers).encode("utf-8")
            ),
            "job_status": status.get("jobStatus", "FINISHED"),
            "failed_ids": status.get("failedIds", []),
            "status_result_count": len(status.get("results", [])),
            "retrieved_at": datetime.now().astimezone().isoformat(),
        }, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    payload = tsv_path.read_bytes() if tsv_path.exists() else b""
    return payload, sha256_bytes(payload) if payload else ""


def normalized_header(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def parse_uniprot_tsv(payload):
    if not payload:
        return []
    text = payload.decode("utf-8", errors="replace").splitlines()
    reader = csv.DictReader(text, delimiter="\t")
    rows = []
    for source in reader:
        normalized = {normalized_header(key): value for key, value in source.items()}
        rows.append({
            "from": normalized.get("from", ""),
            "accession": normalized.get("entry", normalized.get("accession", "")),
            "entry_name": normalized.get("entryname", ""),
            "reviewed": normalized.get("reviewed", ""),
            "protein_name": normalized.get("proteinnames", ""),
            "gene_names": normalized.get("genenames", ""),
            "organism": normalized.get("organism", ""),
            "organism_id": normalized.get("organismid", ""),
            "length": normalized.get("length", ""),
            "sequence": normalized.get("sequence", ""),
        })
    return rows


def reviewed_rank(value):
    return 0 if str(value or "").casefold() in {"reviewed", "yes", "true"} else 1


def select_uniprot_candidate(candidates, organism_id):
    candidates = [row for row in candidates if row.get("sequence")]
    if not candidates:
        return None, "not_found_uniprot", []
    organism_matches = [
        row for row in candidates if str(row.get("organism_id")) == str(organism_id)
    ]
    if organism_matches:
        candidates = organism_matches
    reviewed = [row for row in candidates if reviewed_rank(row.get("reviewed")) == 0]
    if reviewed:
        candidates = reviewed
    unique = {}
    for row in candidates:
        unique[(row.get("accession"), row.get("sequence"))] = row
    candidates = sorted(
        unique.values(),
        key=lambda row: (reviewed_rank(row.get("reviewed")), row.get("accession", "")),
    )
    accessions = [row.get("accession", "") for row in candidates]
    if len(candidates) == 1:
        return candidates[0], "resolved_uniprot", accessions
    sequences = {row.get("sequence") for row in candidates}
    if len(sequences) == 1:
        return candidates[0], "resolved_identical_sequence", accessions
    return None, "ambiguous_uniprot_candidates", accessions


def blank_protein(row):
    output = {field: "" for field in PROTEIN_OUTPUT_FIELDS}
    output.update({
        "entity_id": row["local_entity_id"],
        "mapping_method": row.get("enrichment_route", ""),
        "source_identifier": row.get("query_identifier", ""),
        "resolution_status": "pending",
    })
    return output


def enrich_proteins(
        rows, cache_dir, client, organism_id, poll_interval, poll_attempts):
    route_database = {
        "direct_uniprot": "UniProtKB_AC-ID",
        "ensembl_to_uniprot": "Ensembl",
        "ncbi_to_uniprot": "GeneID",
    }
    mapped = {}
    response_shas = {}
    route_errors = {}
    for route, database in route_database.items():
        identifiers = [
            row["query_identifier"] for row in rows
            if row["enrichment_route"] == route and row["query_identifier"]
        ]
        if not identifiers:
            continue
        print(
            "UniProt mapping route %s (%d identifiers)" % (
                route, len(identifiers)
            )
        )
        try:
            payload, response_sha = ensure_uniprot_mapping(
                identifiers, database, cache_dir, client,
                poll_interval, poll_attempts
            )
        except RuntimeError as exc:
            route_errors[route] = str(exc)
            mapped[route] = {}
            response_shas[route] = ""
            print("UniProt route failed: %s: %s" % (route, exc))
            continue
        grouped = defaultdict(list)
        for candidate in parse_uniprot_tsv(payload):
            grouped[candidate["from"]].append(candidate)
        mapped[route] = grouped
        response_shas[route] = response_sha

    outputs = []
    for row in rows:
        output = blank_protein(row)
        route = row["enrichment_route"]
        if route in route_errors:
            output["resolution_status"] = "request_error_uniprot_mapping"
        elif route in mapped:
            candidates = mapped[route].get(row["query_identifier"], [])
            selected, status, accessions = select_uniprot_candidate(
                candidates, organism_id
            )
            output["candidate_accessions"] = "|".join(accessions)
            output["resolution_status"] = status
            output["response_sha256"] = response_shas.get(route, "")
            if selected is not None:
                output.update({
                    "uniprot_accession": selected["accession"],
                    "sequence": selected["sequence"],
                    "entry_name": selected["entry_name"],
                    "reviewed": selected["reviewed"],
                    "protein_name": selected["protein_name"],
                    "gene_names": selected["gene_names"],
                    "organism": selected["organism"],
                    "organism_id": selected["organism_id"],
                    "length": selected["length"],
                })
        elif route == "gene_symbol_manual_review":
            output["resolution_status"] = "pending_gene_symbol_review"
        else:
            output["resolution_status"] = "unresolved"
        outputs.append(output)
    return outputs


def limit_rows(rows, maximum):
    return rows[:maximum] if maximum and maximum > 0 else rows


def filter_routes(rows, selected_routes):
    if not selected_routes:
        return rows
    selected_routes = set(selected_routes)
    return [row for row in rows if row.get("enrichment_route") in selected_routes]


def status_summary(rows):
    return dict(sorted(Counter(row["resolution_status"] for row in rows).items()))


def build_report(
        args, worklist_dir, cache_dir, compound_rows, protein_rows,
        compound_source, protein_source):
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "offline": args.offline,
        "entity_selection": args.entity,
        "compound_routes": args.compound_route,
        "protein_routes": args.protein_route,
        "max_records": args.max_records,
        "organism_id": args.organism_id,
        "worklist_dir": str(worklist_dir),
        "cache_dir": str(cache_dir),
        "compound_worklist_sha256": (
            sha256_file(compound_source) if compound_source.exists() else None
        ),
        "protein_worklist_sha256": (
            sha256_file(protein_source) if protein_source.exists() else None
        ),
        "compound": {
            "processed": len(compound_rows),
            "status_counts": status_summary(compound_rows),
            "smiles_entities": sum(bool(row["canonical_smiles"]) for row in compound_rows),
            "formula_status_counts": dict(sorted(Counter(
                row["formula_match"] or "not_recorded" for row in compound_rows
            ).items())),
            "formula_verified": sum(
                row["formula_match"] in FORMULA_VERIFIED_STATUSES
                for row in compound_rows
            ),
        },
        "protein": {
            "processed": len(protein_rows),
            "status_counts": status_summary(protein_rows),
            "sequence_entities": sum(bool(row["sequence"]) for row in protein_rows),
        },
    }


def run(args):
    worklist_dir = Path(args.worklist_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = HttpClient(
        timeout=args.timeout, attempts=args.attempts,
        delay=args.delay, offline=args.offline
    )
    compound_source = worklist_dir / "compound_worklist.csv"
    protein_source = worklist_dir / "protein_worklist.csv"
    compound_outputs = []
    protein_outputs = []
    if args.entity in {"compound", "all"}:
        compound_rows = limit_rows(
            filter_routes(read_csv(compound_source), args.compound_route),
            args.max_records,
        )
        compound_outputs = enrich_compounds(
            compound_rows, cache_dir, client, args.batch_size,
            include_manual=args.include_manual_compounds
        )
        write_csv(
            output_root / "compound_attributes.csv",
            compound_outputs, COMPOUND_OUTPUT_FIELDS
        )
    if args.entity in {"protein", "all"}:
        protein_rows = limit_rows(
            filter_routes(read_csv(protein_source), args.protein_route),
            args.max_records,
        )
        protein_outputs = enrich_proteins(
            protein_rows, cache_dir, client, args.organism_id,
            args.poll_interval, args.poll_attempts
        )
        write_csv(
            output_root / "protein_attributes.csv",
            protein_outputs, PROTEIN_OUTPUT_FIELDS
        )
    report = build_report(
        args, worklist_dir, cache_dir, compound_outputs, protein_outputs,
        compound_source, protein_source
    )
    (output_root / "enrichment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Results written to: %s" % output_root)
    return report


def main():
    return 0 if run(parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
