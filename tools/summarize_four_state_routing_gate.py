#!/usr/bin/env python3
"""Combine frozen four-state routing reports across datasets."""

import argparse
import json
from datetime import datetime
from pathlib import Path


STATE_NAMES = (
    'warm_warm',
    'cold_warm',
    'warm_cold',
    'cold_cold',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize four-state routing Gate reports.'
    )
    parser.add_argument(
        '--report',
        action='append',
        required=True,
        metavar='DATASET=REPORT_JSON',
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--require-all-pass',
        action='store_true',
        help='Return a non-zero exit code when any dataset Gate fails.',
    )
    return parser.parse_args()


def parse_report_spec(value):
    if '=' not in value:
        raise ValueError(
            'Report specification must use DATASET=REPORT_JSON: %s' % value
        )
    dataset, path = value.split('=', 1)
    dataset = dataset.strip()
    if not dataset:
        raise ValueError('Dataset label cannot be empty.')
    return dataset, Path(path).expanduser().resolve()


def load_report(path):
    with open(path, encoding='utf-8') as handle:
        report = json.load(handle)
    if report.get('evaluation') != 'four_state_checkpoint_pure_inference':
        raise ValueError('Unexpected report type: %s' % path)
    comparison = report.get('comparison')
    if not comparison:
        raise ValueError('Candidate report has no baseline comparison: %s' % path)
    for state_name in STATE_NAMES + ('macro',):
        if state_name not in report['metrics']:
            raise ValueError(
                'Report is missing %s metrics: %s' % (state_name, path)
            )
        if state_name not in comparison['deltas']:
            raise ValueError(
                'Report is missing %s deltas: %s' % (state_name, path)
            )
    return report


def summarize_reports(report_specs):
    rows = []
    labels = set()
    for value in report_specs:
        dataset, path = parse_report_spec(value)
        if dataset in labels:
            raise ValueError('Duplicate dataset label: %s' % dataset)
        labels.add(dataset)
        report = load_report(path)
        metrics = report['metrics']
        deltas = report['comparison']['deltas']
        candidate_macro = float(metrics['macro']['AUPR'])
        macro_delta = float(deltas['macro']['AUPR'])
        rows.append({
            'dataset': dataset,
            'report': str(path),
            'assignment_sha256': report['support_unit'][
                'assignments_sha256'
            ],
            'baseline_macro_aupr': candidate_macro - macro_delta,
            'candidate_macro_aupr': candidate_macro,
            'macro_aupr_delta': macro_delta,
            'state_aupr_deltas': {
                name: float(deltas[name]['AUPR']) for name in STATE_NAMES
            },
            'gate_checks': report['comparison']['gate_checks'],
            'passed': bool(report['comparison']['passed']),
        })

    all_passed = all(row['passed'] for row in rows)
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': 'frozen_four_state_support_routing_gate',
        'dataset_count': len(rows),
        'all_passed': all_passed,
        'mean_macro_aupr_delta': (
            sum(row['macro_aupr_delta'] for row in rows) / len(rows)
        ),
        'datasets': rows,
    }


def markdown_summary(summary):
    lines = [
        '# Four-State Support Routing Gate',
        '',
        '- Protocol: `%s`' % summary['protocol'],
        '- Dataset count: `%d`' % summary['dataset_count'],
        '- All dataset Gates passed: `%s`' % summary['all_passed'],
        '- Mean Macro-AUPR delta: `%+.6f`' %
        summary['mean_macro_aupr_delta'],
        '',
        '| Dataset | Baseline Macro-AUPR | V2 Macro-AUPR | Delta | '
        'WW Delta | CW Delta | WC Delta | CC Delta | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in summary['datasets']:
        deltas = row['state_aupr_deltas']
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %+.6f | %+.6f | '
            '%+.6f | %+.6f | %s |' % (
                row['dataset'],
                row['baseline_macro_aupr'],
                row['candidate_macro_aupr'],
                row['macro_aupr_delta'],
                deltas['warm_warm'],
                deltas['cold_warm'],
                deltas['warm_cold'],
                deltas['cold_cold'],
                'PASS' if row['passed'] else 'FAIL',
            )
        )
    lines.extend([
        '',
        'Per-dataset Gate requires:',
        '',
        '1. Macro-AUPR delta >= `0.005`;',
        '2. cold-cold AUPR does not decrease;',
        '3. no state AUPR decreases by more than `0.020`.',
        '',
        'Machine-readable details are stored in `summary.json`.',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    summary = summarize_reports(args.report)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'summary.json'
    markdown_path = output_dir / 'summary.md'
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    markdown = markdown_summary(summary)
    markdown_path.write_text(markdown, encoding='utf-8')
    print(markdown)
    print('Summary written to: %s' % markdown_path)
    if args.require_all_pass and not summary['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
