#!/usr/bin/env python3
"""Freeze matched Hctx-P + SDIS configs for the repeated support units."""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_four_state_checkpoint import sha256_file  # noqa: E402


EXPECTED_PROTOCOL = 'frozen_base_hctx_router_vs_sdis_preregistration_v1'


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def repository_path(path):
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_file(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError('%s not found: %s' % (label, path))
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError('%s hash mismatch.' % label)


def replace_setting(text, key, value):
    prefix = key + '='
    lines = text.splitlines()
    positions = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(positions) != 1:
        raise ValueError('Expected one %s setting.' % key)
    lines[positions[0]] = '%s=%s' % (key, value)
    return '\n'.join(lines) + '\n'


def render_sdis_config(base_text, job, overrides):
    text = replace_setting(
        base_text, 'model.variant', job['job_key'] + '_joint_sdis_v1'
    )
    for key in ('context.interaction', 'context.herb_protein'):
        text = replace_setting(text, key, overrides[key])
    text = replace_setting(text, 'inductive.context', overrides[
        'inductive.context'
    ])
    marker = 'inductive.context=True\n'
    additions = (
        'inductive.context.suppress.base.zero.support=%s\n'
        'inductive.context.self.excluded=%s\n'
        % (
            overrides['inductive.context.suppress.base.zero.support'],
            overrides['inductive.context.self.excluded'],
        )
    )
    if marker not in text:
        raise ValueError('Rendered SDIS config is missing inductive.context.')
    text = text.replace(marker, marker + additions, 1)
    return text


def prepare(plan_path, output_manifest, dry_run=False):
    plan = load_json(plan_path)
    if plan.get('protocol') != EXPECTED_PROTOCOL:
        raise ValueError('Unexpected V3/SDIS plan protocol.')
    prepared_path = resolve_path(plan['prepared_units_manifest'])
    v3_summary_path = resolve_path(plan['v3_repeated_summary'])
    verify_file(
        prepared_path, plan['prepared_units_manifest_sha256'],
        'Prepared repeated-unit manifest',
    )
    verify_file(
        v3_summary_path, plan['v3_repeated_summary_sha256'],
        'Frozen V3 repeated summary',
    )
    prepared = load_json(prepared_path)
    v3_summary = load_json(v3_summary_path)
    if len(prepared.get('jobs', [])) != 16:
        raise ValueError('Expected 16 prepared repeated units.')
    if not v3_summary.get('passed') or v3_summary.get('outer_unit_count') != 16:
        raise ValueError('Frozen V3 repeated confirmation is invalid.')

    jobs = []
    overrides = plan['comparator']['config_overrides']
    for job in prepared['jobs']:
        base_path = resolve_path(job['config'])
        verify_file(base_path, job['config_sha256'], job['job_key'] + ' base config')
        output_path = (
            REPOSITORY_ROOT / 'configs'
            / 'frozen_base_hctx_router_vs_sdis_units'
            / (job['job_key'] + '.conf')
        )
        rendered = render_sdis_config(
            base_path.read_text(encoding='utf-8'), job, overrides
        )
        entry = dict(job)
        entry['base_config'] = job['config']
        entry['base_config_sha256'] = job['config_sha256']
        entry['config'] = repository_path(output_path)
        if dry_run:
            entry['config_sha256'] = None
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and output_path.read_text(
                    encoding='utf-8') != rendered:
                raise ValueError('Existing generated config differs: %s' % output_path)
            output_path.write_text(rendered, encoding='utf-8')
            entry['config_sha256'] = sha256_file(output_path)
        jobs.append(entry)

    manifest = {
        'protocol': 'frozen_base_hctx_router_vs_sdis_units_v1',
        'plan': repository_path(plan_path),
        'plan_sha256': sha256_file(plan_path),
        'prepared_units_manifest': repository_path(prepared_path),
        'prepared_units_manifest_sha256': sha256_file(prepared_path),
        'v3_repeated_summary': repository_path(v3_summary_path),
        'v3_repeated_summary_sha256': sha256_file(v3_summary_path),
        'outer_metrics_read_for_comparator_selection': False,
        'jobs': jobs,
    }
    if not dry_run:
        serialized = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + '\n'
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        if output_manifest.exists() and output_manifest.read_text(
                encoding='utf-8') != serialized:
            raise ValueError('Existing comparison manifest differs.')
        output_manifest.write_text(serialized, encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plan', default='configs/frozen_base_hctx_router_vs_sdis_plan.json'
    )
    parser.add_argument(
        '--output-manifest',
        default='configs/frozen_base_hctx_router_vs_sdis_units_manifest.json',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = prepare(
        resolve_path(args.plan), resolve_path(args.output_manifest), args.dry_run
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
