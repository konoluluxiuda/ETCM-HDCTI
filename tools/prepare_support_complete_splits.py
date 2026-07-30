#!/usr/bin/env python3
"""Create frozen target-cold and double-cold evaluation manifests."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_support_complete_cold_start import (
    DEFAULT_DATASETS,
    RELATION_CANDIDATES,
    balanced_entity_folds,
    read_edges,
    resolve_file,
    sha256_file,
    stable_hash,
)
from util.support_complete_split import (
    build_compound_matched_training_negatives,
    records_sha256,
)


MANIFEST_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze entity groups and test candidates for target-cold and "
            "double-cold protocols. No model training is performed."
        )
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-root",
        help=(
            "Optional common output root. By default each manifest is written "
            "under DATASET/splits/support_complete_seed_SEED_kK."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing manifest instead of validating and reusing it.",
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


def safe_name(value):
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")


def record_lines(records):
    return [
        "%s\t%s\t%d" % (
            str(left_id), str(right_id), int(float(label) > 0)
        )
        for left_id, right_id, label in records
    ]


def ids_sha256(values):
    content = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def write_records(path, records, seed, role):
    ordered = sorted(
        records,
        key=lambda row: (
            stable_hash(seed, role, "%s|%s|%d" % (
                row[0], row[1], int(float(row[2]) > 0)
            )),
            str(row[0]),
            str(row[1]),
        ),
    )
    write_atomic(path, "".join(line + "\n" for line in record_lines(ordered)))


def write_groups(path, groups, entity_type):
    lines = ["entity_type\tentity_id\tgroup\n"]
    for group_index, entity_ids in enumerate(groups):
        for entity_id in sorted(entity_ids):
            lines.append(
                "%s\t%s\t%d\n" % (entity_type, entity_id, group_index)
            )
    write_atomic(path, "".join(lines))


def sample_target_matched_negatives(
        positive_edges, candidate_compounds, all_positive_pairs,
        seed, fold_index):
    positives_by_target = {}
    for compound_id, protein_id in positive_edges:
        positives_by_target.setdefault(protein_id, set()).add(compound_id)
    negative_records = []
    candidate_compounds = sorted(candidate_compounds)
    for protein_id in sorted(positives_by_target):
        required = len(positives_by_target[protein_id])
        candidates = [
            compound_id for compound_id in candidate_compounds
            if (compound_id, protein_id) not in all_positive_pairs
        ]
        if len(candidates) < required:
            raise ValueError(
                "Target %s in fold %d has %d unobserved candidates; %d are "
                "required for target-matched sampling."
                % (protein_id, fold_index, len(candidates), required)
            )
        rng = random.Random(
            int(stable_hash(
                seed, "target_negative",
                "%d|%s" % (fold_index, protein_id),
            )[:16], 16)
        )
        rng.shuffle(candidates)
        negative_records.extend(
            [compound_id, protein_id, 0.0]
            for compound_id in candidates[:required]
        )
    return negative_records


def sample_block_negatives(
        heldout_compounds, heldout_proteins, all_positive_pairs,
        required, seed, compound_group, protein_group):
    candidates = [
        (compound_id, protein_id)
        for compound_id in sorted(heldout_compounds)
        for protein_id in sorted(heldout_proteins)
        if (compound_id, protein_id) not in all_positive_pairs
    ]
    if len(candidates) < required:
        raise ValueError(
            "Double-cold cell C%d/P%d has %d unobserved pairs; %d are required."
            % (
                compound_group,
                protein_group,
                len(candidates),
                required,
            )
        )
    rng = random.Random(
        int(stable_hash(
            seed, "double_negative",
            "%d|%d" % (compound_group, protein_group),
        )[:16], 16)
    )
    rng.shuffle(candidates)
    return [
        [compound_id, protein_id, 0.0]
        for compound_id, protein_id in candidates[:required]
    ]


def artifact_metadata(root, path):
    path = Path(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def source_metadata(dataset_dir):
    paths = {
        relation: resolve_file(dataset_dir, candidates)
        for relation, candidates in RELATION_CANDIDATES.items()
    }
    return paths, {
        relation: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for relation, path in paths.items()
    }


def validate_reusable_manifest(manifest_path, expected):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for key, value in expected.items():
        if manifest.get(key) != value:
            mismatches.append(key)
    root = manifest_path.parent
    for name, metadata in manifest.get("artifacts", {}).items():
        path = root / metadata["path"]
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            mismatches.append("artifact:%s" % name)
    if mismatches:
        raise ValueError(
            "Existing support-complete manifest does not match %s. "
            "Use --force or a new output directory."
            % ", ".join(mismatches)
        )
    return manifest


def prepare_dataset_manifest(name, dataset_dir, output_dir,
                             folds=5, seed=2026, force=False):
    if folds < 2 or folds > 10:
        raise ValueError("folds must be between 2 and 10")
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    paths, sources = source_metadata(dataset_dir)
    expected = {
        "version": MANIFEST_VERSION,
        "protocol": "support_complete_cold_start",
        "dataset": name,
        "dataset_path": str(dataset_dir),
        "seed": int(seed),
        "folds": int(folds),
        "sources": sources,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = validate_reusable_manifest(manifest_path, expected)
        print("Reusing support-complete manifest: %s" % manifest_path)
        return manifest

    hc_edges, hc_malformed = read_edges(paths["H_C"])
    cp_edges, cp_malformed = read_edges(paths["C_P"])
    pd_edges, pd_malformed = read_edges(paths["P_D"])
    if any((hc_malformed, cp_malformed, pd_malformed)):
        raise ValueError(
            "Malformed relation rows found in %s: H_C=%d C_P=%d P_D=%d"
            % (name, hc_malformed, cp_malformed, pd_malformed)
        )

    cp_compounds = {compound for compound, _ in cp_edges}
    cp_proteins = {protein for _, protein in cp_edges}
    supported_compounds = cp_compounds & {
        compound for _, compound in hc_edges
    }
    supported_proteins = cp_proteins & {
        protein for protein, _ in pd_edges
    }
    supported_edges = {
        edge for edge in cp_edges
        if edge[0] in supported_compounds
        and edge[1] in supported_proteins
    }
    compound_degrees = Counter(compound for compound, _ in supported_edges)
    protein_degrees = Counter(protein for _, protein in supported_edges)
    compound_groups, compound_loads = balanced_entity_folds(
        compound_degrees, folds, seed, "compound"
    )
    protein_groups, protein_loads = balanced_entity_folds(
        protein_degrees, folds, seed, "protein"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    compound_groups_path = output_dir / "double_cold_compound_groups.tsv"
    protein_groups_path = output_dir / "cold_target_groups.tsv"
    write_groups(compound_groups_path, compound_groups, "compound")
    write_groups(protein_groups_path, protein_groups, "protein")
    artifacts["double_cold_compound_groups"] = artifact_metadata(
        output_dir, compound_groups_path
    )
    artifacts["cold_target_groups"] = artifact_metadata(
        output_dir, protein_groups_path
    )

    target_folds = []
    for fold_index, heldout_proteins in enumerate(protein_groups):
        training_positive_edges = {
            edge for edge in cp_edges if edge[1] not in heldout_proteins
        }
        training_compounds = {
            compound for compound, _ in training_positive_edges
        }
        raw_test_edges = {
            edge for edge in supported_edges
            if edge[1] in heldout_proteins
        }
        test_positive_edges = {
            edge for edge in raw_test_edges
            if edge[0] in training_compounds
        }
        candidate_compounds = supported_compounds & training_compounds
        unit_key = "target_fold_%d" % fold_index
        training_negative_records = (
            build_compound_matched_training_negatives(
                training_positive_edges,
                cp_proteins - heldout_proteins,
                cp_edges,
                seed,
                unit_key,
            )
        )
        negative_records = sample_target_matched_negatives(
            test_positive_edges,
            candidate_compounds,
            cp_edges,
            seed,
            fold_index,
        )
        test_records = (
            [[compound, protein, 1.0]
             for compound, protein in test_positive_edges]
            + negative_records
        )
        test_path = output_dir / "target_cold" / (
            "test_fold_%d.tsv" % fold_index
        )
        write_records(
            test_path,
            test_records,
            seed,
            "target_test_%d" % fold_index,
        )
        artifact_name = "target_cold_test_fold_%d" % fold_index
        artifacts[artifact_name] = artifact_metadata(output_dir, test_path)
        target_folds.append({
            "fold": fold_index,
            "heldout_proteins": len(heldout_proteins),
            "heldout_proteins_sha256": ids_sha256(heldout_proteins),
            "training_positive_count": len(training_positive_edges),
            "training_positives_sha256": records_sha256([
                [compound, protein, 1.0]
                for compound, protein in training_positive_edges
            ]),
            "training_negative_count": len(training_negative_records),
            "training_negatives_sha256": records_sha256(
                training_negative_records
            ),
            "raw_supported_test_positive_count": len(raw_test_edges),
            "test_positive_count": len(test_positive_edges),
            "test_negative_count": len(negative_records),
            "test_records_sha256": records_sha256(test_records),
            "test_path": str(test_path.relative_to(output_dir)),
            "state_purity": (
                len(test_positive_edges) / float(len(raw_test_edges))
                if raw_test_edges else 0.0
            ),
        })

    double_cells = []
    covered_double_positives = set()
    for compound_group, heldout_compounds in enumerate(compound_groups):
        for protein_group, heldout_proteins in enumerate(protein_groups):
            training_positive_edges = {
                edge for edge in cp_edges
                if edge[0] not in heldout_compounds
                and edge[1] not in heldout_proteins
            }
            test_positive_edges = {
                edge for edge in supported_edges
                if edge[0] in heldout_compounds
                and edge[1] in heldout_proteins
            }
            unit_key = "double_c%d_p%d" % (
                compound_group, protein_group
            )
            training_negative_records = (
                build_compound_matched_training_negatives(
                    training_positive_edges,
                    cp_proteins - heldout_proteins,
                    cp_edges,
                    seed,
                    unit_key,
                )
            )
            covered_double_positives.update(test_positive_edges)
            negative_records = sample_block_negatives(
                heldout_compounds,
                heldout_proteins,
                cp_edges,
                len(test_positive_edges),
                seed,
                compound_group,
                protein_group,
            )
            test_records = (
                [[compound, protein, 1.0]
                 for compound, protein in test_positive_edges]
                + negative_records
            )
            test_path = output_dir / "double_cold" / (
                "test_c%d_p%d.tsv" % (compound_group, protein_group)
            )
            write_records(
                test_path,
                test_records,
                seed,
                "double_test_%d_%d" % (
                    compound_group, protein_group
                ),
            )
            artifact_name = "double_cold_test_c%d_p%d" % (
                compound_group, protein_group
            )
            artifacts[artifact_name] = artifact_metadata(
                output_dir, test_path
            )
            double_cells.append({
                "compound_group": compound_group,
                "protein_group": protein_group,
                "heldout_compounds": len(heldout_compounds),
                "heldout_compounds_sha256": ids_sha256(
                    heldout_compounds
                ),
                "heldout_proteins": len(heldout_proteins),
                "heldout_proteins_sha256": ids_sha256(
                    heldout_proteins
                ),
                "training_positive_count": len(training_positive_edges),
                "training_positives_sha256": records_sha256([
                    [compound, protein, 1.0]
                    for compound, protein in training_positive_edges
                ]),
                "training_negative_count": len(
                    training_negative_records
                ),
                "training_negatives_sha256": records_sha256(
                    training_negative_records
                ),
                "test_positive_count": len(test_positive_edges),
                "test_negative_count": len(negative_records),
                "test_records_sha256": records_sha256(test_records),
                "test_path": str(test_path.relative_to(output_dir)),
            })

    manifest = dict(expected)
    manifest.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_algorithm": {
            "entity_groups": "degree_greedy_hash_tie_break_v1",
            "target_negatives": "per_target_matched_cartesian_v1",
            "double_negatives": "cell_global_matched_cartesian_v1",
            "training_negatives": (
                "compound_matched_coprime_traversal_v1"
            ),
        },
        "universe": {
            "cp_compounds": len(cp_compounds),
            "cp_proteins": len(cp_proteins),
            "cp_positive_edges": len(cp_edges),
            "hc_supported_compounds": len(supported_compounds),
            "pd_supported_proteins": len(supported_proteins),
            "both_supported_positive_edges": len(supported_edges),
        },
        "group_loads": {
            "compound_supported_positive_edges": compound_loads,
            "protein_supported_positive_edges": protein_loads,
        },
        "target_cold": {
            "folds": target_folds,
            "test_positives_total": sum(
                row["test_positive_count"] for row in target_folds
            ),
        },
        "double_cold": {
            "grid_shape": [folds, folds],
            "cells": double_cells,
            "test_positives_total": sum(
                row["test_positive_count"] for row in double_cells
            ),
            "covered_supported_positive_count": len(
                covered_double_positives
            ),
            "covered_supported_positives_sha256": records_sha256([
                [compound, protein, 1.0]
                for compound, protein in covered_double_positives
            ]),
        },
        "artifacts": artifacts,
        "strict_guarantees": {
            "source_files_hashed": True,
            "entity_groups_fixed": True,
            "test_candidates_fixed": True,
            "target_cold_protein_disjoint": True,
            "double_cold_compound_and_protein_disjoint": True,
            "all_known_cp_positives_excluded_from_test_negatives": True,
            "double_grid_covers_each_supported_positive_once": (
                covered_double_positives == supported_edges
            ),
            "model_training_performed": False,
        },
    })
    write_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print("Created support-complete manifest: %s" % manifest_path)
    return manifest


def prepare_manifests(datasets, folds=5, seed=2026,
                      output_root=None, force=False):
    manifests = {}
    for name, dataset_dir in datasets.items():
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        if output_root:
            output_dir = (
                Path(output_root).expanduser().resolve() / safe_name(name)
            )
        else:
            output_dir = dataset_dir / "splits" / (
                "support_complete_seed_%d_k%d" % (seed, folds)
            )
        manifests[name] = prepare_dataset_manifest(
            name,
            dataset_dir,
            output_dir,
            folds=folds,
            seed=seed,
            force=force,
        )
    return manifests


def main():
    args = parse_args()
    manifests = prepare_manifests(
        parse_dataset_overrides(args.dataset),
        folds=args.folds,
        seed=args.seed,
        output_root=args.output_root,
        force=args.force,
    )
    print("")
    print("Support-complete split manifests:")
    for name, manifest in manifests.items():
        print(
            "- %s: target positives=%d, double positives=%d, cells=%d"
            % (
                name,
                manifest["target_cold"]["test_positives_total"],
                manifest["double_cold"]["test_positives_total"],
                len(manifest["double_cold"]["cells"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
