#!/usr/bin/env python3
"""Evaluate one frozen base + Hctx-P composite on the untouched outer unit."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_four_state_cc_residual import restore_model  # noqa: E402
from tools.evaluate_four_state_checkpoint import (  # noqa: E402
    STATE_NAMES,
    compare_to_baseline,
    normalize_checkpoint,
    sha256_file,
)
from tools.train_frozen_base_hctx_router import (  # noqa: E402
    checkpoint_hashes,
    evaluate_router,
    records_to_arrays,
    recompute_frozen_base_metrics,
)


SINGLE_UNIT_PROTOCOL = 'frozen_base_hctx_router_outer_evaluation_v3'
REPEATED_UNIT_PROTOCOL = (
    'frozen_base_hctx_router_repeated_outer_evaluation_v1'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_file(path, expected_hash, label):
    if not path.is_file():
        raise FileNotFoundError('%s not found: %s' % (label, path))
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(
            '%s hash mismatch: expected=%s actual=%s.'
            % (label, expected_hash, actual)
        )
    return actual


def markdown_report(report):
    lines = [
        '# Frozen-Base Hctx-P Outer Evaluation',
        '',
        '- Dataset: `%s`' % report['dataset'],
        '- Protocol: `%s`' % report['protocol'],
        '- Training/optimizer steps: `0`',
        '- Parameter selection on outer unit: `False`',
        '- Assignment hash: `%s`' %
        report['support_unit']['assignments_sha256'],
        '',
        '| State | NoContext AUPR | V3 AUPR | Delta | NoContext AUC | '
        'V3 AUC | Delta | Records |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for state_name in STATE_NAMES:
        baseline = report['baseline_metrics'][state_name]
        candidate = report['metrics'][state_name]
        delta = report['comparison']['deltas'][state_name]
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %.6f | %.6f | '
            '%+.6f | %d |' % (
                state_name,
                baseline['AUPR'],
                candidate['AUPR'],
                delta['AUPR'],
                baseline['AUC'],
                candidate['AUC'],
                delta['AUC'],
                candidate['records'],
            )
        )
    baseline = report['baseline_metrics']['macro']
    candidate = report['metrics']['macro']
    delta = report['comparison']['deltas']['macro']
    lines.extend([
        '| macro | %.6f | %.6f | %+.6f | %.6f | %.6f | %+.6f | - |'
        % (
            baseline['AUPR'], candidate['AUPR'], delta['AUPR'],
            baseline['AUC'], candidate['AUC'], delta['AUC'],
        ),
        '',
        'Gate: `%s`' % (
            'PASS' if report['comparison']['passed'] else 'FAIL'
        ),
        '',
        '- Frozen WC exact preservation: `%s`' %
        report['preservation_checks']['warm_cold_exact'],
        '- Frozen CC exact preservation: `%s`' %
        report['preservation_checks']['cold_cold_exact'],
        '- Base checkpoint hashes unchanged: `%s`' %
        report['preservation_checks']['checkpoint_hashes_unchanged'],
        '- Hctx-P head hash unchanged: `%s`' %
        report['preservation_checks']['head_hash_unchanged'],
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    manifest = load_json(manifest_path)
    protocol = manifest.get('protocol')
    if protocol not in (SINGLE_UNIT_PROTOCOL, REPEATED_UNIT_PROTOCOL):
        raise ValueError('Unexpected outer-evaluation manifest protocol.')
    if args.dataset not in manifest['datasets']:
        raise ValueError('Dataset is absent from manifest: %s' % args.dataset)
    spec = manifest['datasets'][args.dataset]

    if protocol == SINGLE_UNIT_PROTOCOL:
        parent_manifest_path = resolve_path(
            manifest['parent_gate_manifest']
        )
        parent_summary_path = resolve_path(
            manifest['parent_gate_summary']
        )
        verify_file(
            parent_manifest_path,
            manifest['parent_gate_manifest_sha256'],
            'Parent Gate manifest',
        )
        verify_file(
            parent_summary_path,
            manifest['parent_gate_summary_sha256'],
            'Parent Gate summary',
        )
        parent_summary = load_json(parent_summary_path)
        if not parent_summary.get('all_passed'):
            raise ValueError('Parent four-dataset Gate did not pass.')
        expected_training_manifest_hash = manifest[
            'parent_gate_manifest_sha256'
        ]
        require_inner_gate_pass = True
    else:
        prepared_manifest_path = resolve_path(
            manifest['prepared_units_manifest']
        )
        head_manifest_path = resolve_path(
            manifest['head_training_manifest']
        )
        verify_file(
            prepared_manifest_path,
            manifest['prepared_units_manifest_sha256'],
            'Prepared repeated-unit manifest',
        )
        verify_file(
            head_manifest_path,
            manifest['head_training_manifest_sha256'],
            'Frozen head-training manifest',
        )
        prepared = load_json(prepared_manifest_path)
        prepared_jobs = {
            row['job_key']: row for row in prepared['jobs']
        }
        job_key = spec['job_key']
        if job_key not in prepared_jobs:
            raise ValueError('Repeated outer job is not preregistered.')
        prepared_job = prepared_jobs[job_key]
        for name in ('dataset', 'compound_group', 'protein_group'):
            if prepared_job[name] != spec[name]:
                raise ValueError(
                    'Repeated outer job metadata mismatch: %s.' % name
                )
        if prepared_job['assignments_sha256'] != spec[
                'assignments_sha256']:
            raise ValueError('Repeated outer assignment hash mismatch.')
        expected_training_manifest_hash = manifest[
            'head_training_manifest_sha256'
        ]
        require_inner_gate_pass = False

    training_report_path = resolve_path(spec['training_report'])
    head_path = resolve_path(spec['head'])
    verify_file(
        training_report_path,
        spec['training_report_sha256'],
        'Training report',
    )
    head_hash_before = verify_file(
        head_path, spec['head_sha256'], 'Hctx-P head'
    )
    training_report = load_json(training_report_path)
    if training_report.get('protocol') != (
            'frozen_base_hctx_router_pilot_v3'):
        raise ValueError('Unexpected training report protocol.')
    if (
            require_inner_gate_pass
            and not training_report['comparison']['passed']):
        raise ValueError('Training report did not pass its inner Gate.')
    if not all(training_report['preservation_checks'].values()):
        raise ValueError('Training report failed a preservation check.')
    if training_report['manifest']['sha256'] != (
            expected_training_manifest_hash):
        raise ValueError(
            'Training report did not use the frozen head manifest.'
        )

    config_path = resolve_path(training_report['config']['path'])
    checkpoint = normalize_checkpoint(
        training_report['base_checkpoint']['prefix']
    )
    if sha256_file(config_path) != training_report['config']['sha256']:
        raise ValueError('Training report/config hash mismatch.')
    hashes_before = checkpoint_hashes(checkpoint)
    if hashes_before != training_report['base_checkpoint']['files_after']:
        raise ValueError('Base checkpoint differs from the training report.')

    with np.load(head_path) as payload:
        head = np.asarray(
            payload['context_herb_protein'], dtype=np.float64
        )
    conf, experiment, model = restore_model(config_path, checkpoint)
    state = model.fetchModelState()
    if head.shape != (state['compound'].shape[1],):
        model.sess.close()
        raise ValueError('Hctx-P head dimension does not match the model.')
    outer_arrays = {
        state_name: records_to_arrays(
            model,
            state,
            experiment.supportTestDataByState[state_name],
        )
        for state_name in STATE_NAMES
    }
    baseline_metrics = recompute_frozen_base_metrics(outer_arrays)
    preserved_metrics = {
        state_name: baseline_metrics[state_name]
        for state_name in ('warm_cold', 'cold_cold')
    }
    metrics = evaluate_router(
        outer_arrays,
        head,
        preserved_metrics=preserved_metrics,
    )
    model.sess.close()
    hashes_after = checkpoint_hashes(checkpoint)
    head_hash_after = sha256_file(head_path)
    comparison = compare_to_baseline(metrics, baseline_metrics)
    preservation_checks = {
        'warm_cold_exact': (
            metrics['warm_cold'] == baseline_metrics['warm_cold']
        ),
        'cold_cold_exact': (
            metrics['cold_cold'] == baseline_metrics['cold_cold']
        ),
        'checkpoint_hashes_unchanged': hashes_before == hashes_after,
        'head_hash_unchanged': head_hash_before == head_hash_after,
    }
    if not all(preservation_checks.values()):
        raise ValueError('Outer frozen-artifact preservation check failed.')

    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'evaluation': 'frozen_base_hctx_router_outer_pure_inference',
        'protocol': protocol,
        'dataset': spec['display_name'],
        'job': {
            'key': spec.get('job_key', args.dataset),
            'dataset': spec.get('dataset', args.dataset),
            'compound_group': spec.get('compound_group'),
            'protein_group': spec.get('protein_group'),
        },
        'manifest': {
            'path': str(manifest_path),
            'sha256': sha256_file(manifest_path),
        },
        'parent_training_report': str(training_report_path),
        'base_checkpoint': {
            'prefix': str(checkpoint),
            'files_before': hashes_before,
            'files_after': hashes_after,
        },
        'head': {
            'path': str(head_path),
            'sha256_before': head_hash_before,
            'sha256_after': head_hash_after,
        },
        'support_unit': experiment.supportUnitMetadata,
        'routing': training_report['routing'],
        'training_optimizer_steps': 0,
        'parameter_selection_on_outer': False,
        'baseline_metrics': baseline_metrics,
        'metrics': metrics,
        'comparison': comparison,
        'preservation_checks': preservation_checks,
    }
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    markdown = markdown_report(report)
    (output_dir / 'report.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()
