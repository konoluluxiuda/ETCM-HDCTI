#!/usr/bin/env python3
"""Prepare unseen four-state units and frozen NoContext configs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.prepare_four_state_support_unit import (  # noqa: E402
    prepare_four_state_artifact,
)
from util.support_complete_split import sha256_file  # noqa: E402


EXPECTED_PROTOCOL = (
    "frozen_base_hctx_router_repeated_outer_preregistration_v1"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="configs/frozen_base_hctx_router_repeated_outer_plan.json",
    )
    parser.add_argument(
        "--output-manifest",
        default=(
            "configs/frozen_base_hctx_router_repeated_units_manifest.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def repository_path(path):
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def verify_file(path, expected_hash, label):
    if not path.is_file():
        raise FileNotFoundError("%s not found: %s" % (label, path))
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(
            "%s hash mismatch: expected=%s actual=%s"
            % (label, expected_hash, actual)
        )
    return actual


def replace_setting(text, key, value):
    prefix = key + "="
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines)
               if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one %s setting; found %d."
            % (key, len(matches))
        )
    lines[matches[0]] = "%s=%s" % (key, value)
    return "\n".join(lines) + "\n"


def render_config(template_text, dataset_key, group, artifact_manifest):
    variant = "%s_four_state_no_context_c%dp%d_v1" % (
        dataset_key, group, group
    )
    text = replace_setting(
        template_text, "model.variant", variant
    )
    text = replace_setting(
        text,
        "support.four.state.manifest",
        "./" + repository_path(artifact_manifest),
    )
    return text


def deterministic_json_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare(plan_path, output_manifest_path, dry_run=False):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Unexpected repeated-outer plan protocol.")

    jobs = []
    for dataset_key, spec in plan["datasets"].items():
        source_manifest = resolve_path(spec["source_manifest"])
        template_config = resolve_path(spec["template_config"])
        verify_file(
            source_manifest,
            spec["source_manifest_sha256"],
            "%s source manifest" % dataset_key,
        )
        verify_file(
            template_config,
            spec["template_config_sha256"],
            "%s template config" % dataset_key,
        )
        template_text = template_config.read_text(encoding="utf-8")

        for unit in plan["confirmatory_units"]:
            compound_group = int(unit["compound_group"])
            protein_group = int(unit["protein_group"])
            if compound_group != protein_group:
                raise ValueError("Only diagonal cN/pN units are allowed.")
            group = compound_group
            artifact_dir = (
                source_manifest.parent
                / (
                    "four_state_seed_%d_c%d_p%d"
                    % (
                        int(plan["support_unit"]["seed"]),
                        group,
                        group,
                    )
                )
            )
            artifact_manifest = artifact_dir / "manifest.json"
            config_path = (
                REPOSITORY_ROOT
                / "configs"
                / "frozen_base_hctx_router_repeated_units"
                / ("%s_c%dp%d.conf" % (dataset_key, group, group))
            )
            if dry_run:
                jobs.append({
                    "job_key": "%s_c%dp%d" % (
                        dataset_key, group, group
                    ),
                    "artifact_manifest": repository_path(
                        artifact_manifest
                    ),
                    "config": repository_path(config_path),
                })
                continue

            prepare_four_state_artifact(
                source_manifest,
                artifact_dir,
                compound_group=group,
                protein_group=group,
                warm_holdout_ratio=float(
                    plan["support_unit"]["warm_holdout_ratio"]
                ),
                seed=int(plan["support_unit"]["seed"]),
            )
            rendered = render_config(
                template_text, dataset_key, group, artifact_manifest
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                existing = config_path.read_text(encoding="utf-8")
                if existing != rendered:
                    raise ValueError(
                        "Existing generated config differs: %s" % config_path
                    )
            else:
                config_path.write_text(rendered, encoding="utf-8")

            artifact = json.loads(
                artifact_manifest.read_text(encoding="utf-8")
            )
            jobs.append({
                "job_key": "%s_c%dp%d" % (
                    dataset_key, group, group
                ),
                "dataset": dataset_key,
                "display_name": spec["display_name"],
                "compound_group": group,
                "protein_group": group,
                "artifact_manifest": repository_path(artifact_manifest),
                "artifact_manifest_sha256": sha256_file(
                    artifact_manifest
                ),
                "assignments_sha256": artifact["metadata"][
                    "assignments_sha256"
                ],
                "config": repository_path(config_path),
                "config_sha256": sha256_file(config_path),
            })

    prepared = {
        "protocol": "frozen_base_hctx_router_repeated_units_prepared_v1",
        "plan": repository_path(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "plan_semantic_sha256": deterministic_json_sha256(plan),
        "outer_metrics_read": False,
        "training_started": False,
        "jobs": jobs,
    }
    if not dry_run:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            prepared, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        if output_manifest_path.exists():
            existing = output_manifest_path.read_text(encoding="utf-8")
            if existing != serialized:
                raise ValueError(
                    "Existing prepared manifest differs: %s"
                    % output_manifest_path
                )
        else:
            output_manifest_path.write_text(serialized, encoding="utf-8")
    return prepared


def main():
    args = parse_args()
    plan_path = resolve_path(args.plan)
    output_manifest_path = resolve_path(args.output_manifest)
    prepared = prepare(
        plan_path, output_manifest_path, dry_run=args.dry_run
    )
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print("Prepared manifest: %s" % output_manifest_path)


if __name__ == "__main__":
    main()
