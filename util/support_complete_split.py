"""Load frozen target-cold and double-cold evaluation units."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def stable_int(seed, role, value):
    content = "%d|%s|%s" % (int(seed), str(role), str(value))
    return int(hashlib.sha256(content.encode("utf-8")).hexdigest()[:16], 16)


def record_lines(records):
    return [
        "%s\t%s\t%d" % (
            str(left_id), str(right_id), int(float(label) > 0)
        )
        for left_id, right_id, label in records
    ]


def records_sha256(records):
    content = "\n".join(sorted(record_lines(records))) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path):
    pairs = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(
                    "Invalid relation row %d in %s." % (line_number, path)
                )
            pairs.add((str(parts[0]), str(parts[1])))
    return pairs


def read_records(path):
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    "Invalid test record row %d in %s." % (
                        line_number, path
                    )
                )
            record = [str(parts[0]), str(parts[1]), float(parts[2])]
            pair = (record[0], record[1])
            if pair in seen:
                raise ValueError("Duplicate test pair %s in %s." % (pair, path))
            seen.add(pair)
            records.append(record)
    return records


def read_groups(path, expected_entity_type):
    groups = {}
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        header = next(handle, "").strip().split("\t")
        if header != ["entity_type", "entity_id", "group"]:
            raise ValueError("Invalid entity group header in %s." % path)
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(
                    "Invalid entity group row %d in %s." % (
                        line_number, path
                    )
                )
            entity_type, entity_id, group_value = parts
            if entity_type != expected_entity_type:
                raise ValueError(
                    "Expected %s group in %s; found %s."
                    % (expected_entity_type, path, entity_type)
                )
            if entity_id in seen:
                raise ValueError(
                    "Entity %s appears in multiple groups in %s."
                    % (entity_id, path)
                )
            seen.add(entity_id)
            groups.setdefault(int(group_value), set()).add(entity_id)
    return groups


def _coprime_stride(size, seed_value):
    if size <= 1:
        return 1
    stride = 1 + seed_value % (size - 1)
    while math.gcd(stride, size) != 1:
        stride += 1
        if stride >= size:
            stride = 1
    return stride


def build_compound_matched_training_negatives(
        training_positive_edges, allowed_proteins, all_positive_pairs,
        seed, unit_key):
    """Build deterministic 1:1 negatives without materializing a Cartesian pool."""
    positives_by_compound = {}
    for compound_id, protein_id in training_positive_edges:
        positives_by_compound.setdefault(compound_id, set()).add(protein_id)
    allowed_proteins = sorted(str(value) for value in allowed_proteins)
    if not allowed_proteins:
        raise ValueError("Training negative generation has no allowed proteins.")

    negative_records = []
    for compound_id in sorted(positives_by_compound):
        required = len(positives_by_compound[compound_id])
        available = sum(
            (compound_id, protein_id) not in all_positive_pairs
            for protein_id in allowed_proteins
        )
        if available < required:
            raise ValueError(
                "Compound %s has %d allowed unobserved proteins; %d are "
                "required for unit %s."
                % (compound_id, available, required, unit_key)
            )
        base_seed = stable_int(
            seed, "training_negative", "%s|%s" % (unit_key, compound_id)
        )
        start = base_seed % len(allowed_proteins)
        stride = _coprime_stride(
            len(allowed_proteins),
            stable_int(
                seed,
                "training_negative_stride",
                "%s|%s" % (unit_key, compound_id),
            ),
        )
        selected = 0
        for offset in range(len(allowed_proteins)):
            protein_id = allowed_proteins[
                (start + offset * stride) % len(allowed_proteins)
            ]
            if (compound_id, protein_id) in all_positive_pairs:
                continue
            negative_records.append([compound_id, protein_id, 0.0])
            selected += 1
            if selected == required:
                break
        if selected != required:
            raise AssertionError(
                "Deterministic negative traversal ended before capacity."
            )
    return negative_records


def _verify_manifest_files(manifest_path, manifest):
    root = Path(manifest_path).resolve().parent
    for relation, metadata in manifest["sources"].items():
        path = Path(metadata["path"])
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(
                "Support-complete source file changed: %s." % relation
            )
    for name, metadata in manifest["artifacts"].items():
        path = root / metadata["path"]
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(
                "Support-complete artifact changed: %s." % name
            )


def _unit_entry(manifest, mode, fold, compound_group, protein_group):
    if mode == "target_cold":
        if fold is None:
            raise ValueError("target_cold requires fold.")
        matches = [
            row for row in manifest["target_cold"]["folds"]
            if int(row["fold"]) == int(fold)
        ]
    elif mode == "double_cold":
        if compound_group is None or protein_group is None:
            raise ValueError(
                "double_cold requires compound_group and protein_group."
            )
        matches = [
            row for row in manifest["double_cold"]["cells"]
            if int(row["compound_group"]) == int(compound_group)
            and int(row["protein_group"]) == int(protein_group)
        ]
    else:
        raise ValueError(
            "Unsupported support-complete mode %r." % mode
        )
    if len(matches) != 1:
        raise ValueError(
            "Expected one manifest entry for %s; found %d."
            % (mode, len(matches))
        )
    return matches[0]


def load_support_complete_unit(
        manifest_path, mode, fold=None,
        compound_group=None, protein_group=None):
    """Reconstruct one explicit train/test unit and verify all frozen hashes."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise ValueError(
            "Support-complete loader requires manifest version 2."
        )
    if manifest.get("protocol") != "support_complete_cold_start":
        raise ValueError("Unexpected support-complete protocol.")
    _verify_manifest_files(manifest_path, manifest)
    root = manifest_path.parent
    entry = _unit_entry(
        manifest, mode, fold, compound_group, protein_group
    )
    cp_edges = read_pairs(manifest["sources"]["C_P"]["path"])
    cp_compounds = {compound for compound, _ in cp_edges}
    cp_proteins = {protein for _, protein in cp_edges}
    protein_groups = read_groups(
        root / manifest["artifacts"]["cold_target_groups"]["path"],
        "protein",
    )

    if mode == "target_cold":
        heldout_proteins = protein_groups[int(fold)]
        heldout_compounds = set()
        training_positive_edges = {
            edge for edge in cp_edges
            if edge[1] not in heldout_proteins
        }
        unit_key = "target_fold_%d" % int(fold)
    else:
        compound_groups = read_groups(
            root / manifest["artifacts"][
                "double_cold_compound_groups"
            ]["path"],
            "compound",
        )
        heldout_compounds = compound_groups[int(compound_group)]
        heldout_proteins = protein_groups[int(protein_group)]
        training_positive_edges = {
            edge for edge in cp_edges
            if edge[0] not in heldout_compounds
            and edge[1] not in heldout_proteins
        }
        unit_key = "double_c%d_p%d" % (
            int(compound_group), int(protein_group)
        )

    training_positive_records = [
        [compound_id, protein_id, 1.0]
        for compound_id, protein_id in training_positive_edges
    ]
    if (
        len(training_positive_records) != entry["training_positive_count"]
        or records_sha256(training_positive_records)
        != entry["training_positives_sha256"]
    ):
        raise ValueError(
            "Reconstructed training positives do not match manifest for %s."
            % unit_key
        )
    allowed_proteins = cp_proteins - heldout_proteins
    training_negative_records = build_compound_matched_training_negatives(
        training_positive_edges,
        allowed_proteins,
        cp_edges,
        manifest["seed"],
        unit_key,
    )
    if (
        len(training_negative_records) != entry["training_negative_count"]
        or records_sha256(training_negative_records)
        != entry["training_negatives_sha256"]
    ):
        raise ValueError(
            "Reconstructed training negatives do not match manifest for %s."
            % unit_key
        )

    test_path = root / entry["test_path"]
    test_records = read_records(test_path)
    if records_sha256(test_records) != entry["test_records_sha256"]:
        raise ValueError(
            "Loaded test records do not match manifest for %s." % unit_key
        )
    test_positive_count = sum(float(row[2]) > 0 for row in test_records)
    test_negative_count = len(test_records) - test_positive_count
    if (
        test_positive_count != entry["test_positive_count"]
        or test_negative_count != entry["test_negative_count"]
    ):
        raise ValueError("Test label counts do not match manifest.")

    training_records = training_positive_records + training_negative_records
    train_pairs = {(row[0], row[1]) for row in training_records}
    test_pairs = {(row[0], row[1]) for row in test_records}
    if train_pairs & test_pairs:
        raise ValueError("Support-complete unit contains train/test pair overlap.")
    if any(
            float(row[2]) <= 0 and (row[0], row[1]) in cp_edges
            for row in training_records + test_records):
        raise ValueError("A sampled negative is a known C-P positive.")

    train_compounds = {row[0] for row in training_records}
    train_proteins = {row[1] for row in training_records}
    if train_proteins & heldout_proteins:
        raise ValueError("Held-out protein leaked into training records.")
    if mode == "target_cold":
        if any(row[1] not in heldout_proteins for row in test_records):
            raise ValueError("Target-cold test contains a non-held-out protein.")
        positive_test_compounds = {
            row[0] for row in test_records if float(row[2]) > 0
        }
        if not positive_test_compounds <= train_compounds:
            raise ValueError(
                "Target-cold test contains a cold compound positive."
            )
    else:
        if train_compounds & heldout_compounds:
            raise ValueError("Held-out compound leaked into training records.")
        if any(
                row[0] not in heldout_compounds
                or row[1] not in heldout_proteins
                for row in test_records):
            raise ValueError(
                "Double-cold test contains an endpoint outside its cell."
            )

    metadata = {
        "mode": mode,
        "unit_key": unit_key,
        "seed": int(manifest["seed"]),
        "training_positive_count": len(training_positive_records),
        "training_negative_count": len(training_negative_records),
        "test_positive_count": test_positive_count,
        "test_negative_count": test_negative_count,
        "heldout_compounds": len(heldout_compounds),
        "heldout_proteins": len(heldout_proteins),
        "train_test_pair_overlap": 0,
        "train_test_compound_overlap": len(
            train_compounds & heldout_compounds
        ),
        "train_test_protein_overlap": len(
            train_proteins & heldout_proteins
        ),
        "training_positives_sha256": records_sha256(
            training_positive_records
        ),
        "training_negatives_sha256": records_sha256(
            training_negative_records
        ),
        "test_records_sha256": records_sha256(test_records),
    }
    return training_records, test_records, metadata
