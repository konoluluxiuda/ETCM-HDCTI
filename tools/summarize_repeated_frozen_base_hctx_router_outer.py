#!/usr/bin/env python3
"""Summarize preregistered repeated frozen-router outer units."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


STATE_NAMES = (
    'warm_warm',
    'cold_warm',
    'warm_cold',
    'cold_cold',
)
EXPECTED_PROTOCOL = (
    'frozen_base_hctx_router_repeated_outer_evaluation_v1'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--prepared-manifest', required=True)
    parser.add_argument('--report', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--require-pass', action='store_true')
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }


def summarize(plan, prepared, report_paths):
    expected_jobs = {row['job_key']: row for row in prepared['jobs']}
    rows = []
    seen = set()
    for value in report_paths:
        path = Path(value).expanduser().resolve()
        report = load_json(path)
        if report.get('protocol') != EXPECTED_PROTOCOL:
            raise ValueError('Unexpected repeated outer report: %s' % path)
        job_key = report['job']['key']
        if job_key not in expected_jobs or job_key in seen:
            raise ValueError('Unexpected or duplicate job: %s' % job_key)
        seen.add(job_key)
        expected = expected_jobs[job_key]
        if report['support_unit']['assignments_sha256'] != (
                expected['assignments_sha256']):
            raise ValueError('Assignment mismatch for %s.' % job_key)
        if report['training_optimizer_steps'] != 0:
            raise ValueError('Outer training detected for %s.' % job_key)
        if report['parameter_selection_on_outer']:
            raise ValueError('Outer parameter selection detected for %s.' % job_key)
        if not all(report['preservation_checks'].values()):
            raise ValueError('Preservation check failed for %s.' % job_key)
        rows.append({
            'job_key': job_key,
            'dataset': expected['dataset'],
            'display_name': expected['display_name'],
            'compound_group': expected['compound_group'],
            'protein_group': expected['protein_group'],
            'report': str(path),
            'baseline_macro_aupr': report[
                'baseline_metrics']['macro']['AUPR'],
            'candidate_macro_aupr': report['metrics']['macro']['AUPR'],
            'deltas': {
                name: report['comparison']['deltas'][name]['AUPR']
                for name in STATE_NAMES + ('macro',)
            },
        })
    missing = sorted(set(expected_jobs) - seen)
    if missing:
        raise ValueError('Missing repeated outer reports: %s' % missing)

    rows.sort(key=lambda row: (row['dataset'], row['compound_group']))
    by_dataset = []
    for dataset_key, spec in plan['datasets'].items():
        dataset_rows = [
            row for row in rows if row['dataset'] == dataset_key
        ]
        expected_count = len(plan['confirmatory_units'])
        if len(dataset_rows) != expected_count:
            raise ValueError(
                '%s has %d/%d reports.' % (
                    dataset_key, len(dataset_rows), expected_count
                )
            )
        by_dataset.append({
            'dataset': dataset_key,
            'display_name': spec['display_name'],
            'units': len(dataset_rows),
            'baseline_macro_aupr': mean_std([
                row['baseline_macro_aupr'] for row in dataset_rows
            ]),
            'candidate_macro_aupr': mean_std([
                row['candidate_macro_aupr'] for row in dataset_rows
            ]),
            'delta': {
                name: mean_std([
                    row['deltas'][name] for row in dataset_rows
                ])
                for name in STATE_NAMES + ('macro',)
            },
            'positive_macro_units': sum(
                row['deltas']['macro'] > 0.0 for row in dataset_rows
            ),
        })

    gate = plan['confirmatory_gate']
    overall_delta = float(np.mean([
        row['deltas']['macro'] for row in rows
    ]))
    positive_units = sum(row['deltas']['macro'] > 0.0 for row in rows)
    checks = {
        'expected_new_outer_unit_count': (
            len(rows) == int(gate['new_outer_unit_count'])
        ),
        'overall_mean_macro_aupr_delta_minimum': (
            overall_delta
            >= float(gate['overall_mean_macro_aupr_delta_minimum'])
        ),
        'minimum_positive_units': (
            positive_units >= int(gate['minimum_positive_units'])
        ),
        'each_dataset_mean_macro_aupr_delta_not_lower': all(
            row['delta']['macro']['mean'] >= 0.0 for row in by_dataset
        ),
        'maximum_dataset_mean_state_aupr_drop': all(
            row['delta'][state]['mean']
            >= -float(gate['maximum_dataset_mean_state_aupr_drop'])
            for row in by_dataset for state in STATE_NAMES
        ),
        'warm_cold_and_cold_cold_exact_preservation': all(
            row['deltas']['warm_cold'] == 0.0
            and row['deltas']['cold_cold'] == 0.0
            for row in rows
        ),
        'no_parameter_selection_on_outer_units': True,
    }
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': 'frozen_base_hctx_router_repeated_outer_gate_v1',
        'primary_analysis': 'new_c1p1_through_c4p4_outer_units',
        'outer_unit_count': len(rows),
        'overall_mean_macro_aupr_delta': overall_delta,
        'positive_macro_units': positive_units,
        'gate_checks': checks,
        'passed': all(checks.values()),
        'datasets': by_dataset,
        'units': rows,
    }


def markdown(summary):
    lines = [
        '# Repeated Frozen-Base Hctx-P Outer Gate',
        '',
        '- Protocol: `%s`' % summary['protocol'],
        '- New outer units: `%d`' % summary['outer_unit_count'],
        '- Positive Macro-AUPR units: `%d/%d`' % (
            summary['positive_macro_units'], summary['outer_unit_count']
        ),
        '- Overall mean Macro-AUPR delta: `%+.6f`' % (
            summary['overall_mean_macro_aupr_delta']
        ),
        '- Gate: `%s`' % ('PASS' if summary['passed'] else 'FAIL'),
        '',
        '| Dataset | Units | NoContext AUPR | V3 AUPR | Delta | '
        'Positive | WW Delta | CW Delta | WC Delta | CC Delta |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary['datasets']:
        lines.append(
            '| %s | %d | %.6f (±%.6f) | %.6f (±%.6f) | '
            '%+.6f (±%.6f) | %d/%d | %+.6f | %+.6f | %+.6f | '
            '%+.6f |' % (
                row['display_name'], row['units'],
                row['baseline_macro_aupr']['mean'],
                row['baseline_macro_aupr']['std'],
                row['candidate_macro_aupr']['mean'],
                row['candidate_macro_aupr']['std'],
                row['delta']['macro']['mean'],
                row['delta']['macro']['std'],
                row['positive_macro_units'], row['units'],
                row['delta']['warm_warm']['mean'],
                row['delta']['cold_warm']['mean'],
                row['delta']['warm_cold']['mean'],
                row['delta']['cold_cold']['mean'],
            )
        )
    lines.extend(['', '## Gate Checks', ''])
    for name, passed in summary['gate_checks'].items():
        lines.append('- `%s`: `%s`' % (name, passed))
    lines.extend([
        '',
        'All heads were frozen before any new outer metric was read. '
        'No outer-unit parameter selection was performed.',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    plan = load_json(Path(args.plan).expanduser().resolve())
    prepared = load_json(
        Path(args.prepared_manifest).expanduser().resolve()
    )
    summary = summarize(plan, prepared, args.report)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    text = markdown(summary)
    (output_dir / 'summary.md').write_text(text, encoding='utf-8')
    print(text)
    print('Summary written to: %s' % (output_dir / 'summary.md'))
    if args.require_pass and not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
