#!/usr/bin/env python3
"""Freeze one shared four-state support unit as verified TSV artifacts."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from util.support_complete_split import (  # noqa: E402
    build_four_state_support_unit,
    load_four_state_support_artifact,
    records_sha256,
    sha256_file,
)


STATES = ("warm_warm", "cold_warm", "warm_cold", "cold_cold")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Freeze a shared four-state support-complete unit.'
    )
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--compound-group', type=int, default=0)
    parser.add_argument('--protein-group', type=int, default=0)
    parser.add_argument('--warm-holdout-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir')
    return parser.parse_args()


def write_records(path, records):
    lines = sorted(
        "%s\t%s\t%d\n" % (
            str(left), str(right), int(float(label) > 0)
        )
        for left, right, label in records
    )
    path.write_text("".join(lines), encoding="utf-8")


def artifact_entry(path, root, records):
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "records": len(records),
        "records_sha256": records_sha256(records),
    }


def prepare_four_state_artifact(
        source_manifest, output_dir, compound_group=0, protein_group=0,
        warm_holdout_ratio=0.1, seed=2026):
    source_manifest = Path(source_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    training, states, metadata = build_four_state_support_unit(
        source_manifest,
        compound_group,
        protein_group,
        warm_holdout_ratio=warm_holdout_ratio,
        seed=seed,
    )

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_parameters = {
            "compound_group": int(compound_group),
            "protein_group": int(protein_group),
            "warm_holdout_ratio": float(warm_holdout_ratio),
            "seed": int(seed),
        }
        if existing.get("parameters") != expected_parameters:
            raise ValueError(
                "Existing four-state artifact parameters do not match."
            )
        if (
            existing.get("metadata", {}).get("assignments_sha256")
            != metadata["assignments_sha256"]
        ):
            raise ValueError(
                "Existing four-state assignment hash does not match."
            )
        load_four_state_support_artifact(manifest_path)
        print("Reusing four-state support artifact: %s" % manifest_path)
        return existing

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Output directory is non-empty but has no manifest: %s."
            % output_dir
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training.tsv"
    write_records(training_path, training)
    artifacts = {
        "training": artifact_entry(training_path, output_dir, training)
    }
    for state in STATES:
        path = output_dir / ("test_%s.tsv" % state)
        write_records(path, states[state])
        artifacts["test_%s" % state] = artifact_entry(
            path, output_dir, states[state]
        )

    manifest = {
        "version": 1,
        "protocol": "support_complete_four_state",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "parameters": {
            "compound_group": int(compound_group),
            "protein_group": int(protein_group),
            "warm_holdout_ratio": float(warm_holdout_ratio),
            "seed": int(seed),
        },
        "artifacts": artifacts,
        "metadata": metadata,
    }
    manifest_path.write_text(
        json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    load_four_state_support_artifact(manifest_path)
    print("Created four-state support artifact: %s" % manifest_path)
    return manifest


def main():
    args = parse_args()
    source_manifest = Path(args.source_manifest).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        source_manifest.parent /
        (
            "four_state_seed_%d_c%d_p%d" % (
                args.seed, args.compound_group, args.protein_group
            )
        )
    )
    manifest = prepare_four_state_artifact(
        source_manifest,
        output_dir,
        compound_group=args.compound_group,
        protein_group=args.protein_group,
        warm_holdout_ratio=args.warm_holdout_ratio,
        seed=args.seed,
    )
    print(json.dumps({
        "output_manifest": str(output_dir / "manifest.json"),
        "unit_key": manifest["metadata"]["unit_key"],
        "assignments_sha256": manifest["metadata"][
            "assignments_sha256"
        ],
        "training_positive_count": manifest["metadata"][
            "training_positive_count"
        ],
        "state_positive_counts": {
            state: manifest["metadata"]["states"][state]["positive_count"]
            for state in STATES
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
