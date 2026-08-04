#!/usr/bin/env python3
"""Summarize four frozen outer-unit Hctx-P evaluations."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--require-all-pass', action='store_true')
    return parser.parse_args()


def summarize(report_paths):
    rows = []
    for value in report_paths:
        path = Path(value).expanduser().resolve()
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('evaluation') != (
                'frozen_base_hctx_router_outer_pure_inference'):
            raise ValueError('Unexpected outer report: %s' % path)
        passed = bool(
            report['comparison']['passed']
            and all(report['preservation_checks'].values())
            and report['training_optimizer_steps'] == 0
            and not report['parameter_selection_on_outer']
        )
        rows.append({
            'dataset': report['dataset'],
            'report': str(path),
            'baseline_macro_aupr': report['baseline_metrics']['macro']['AUPR'],
            'candidate_macro_aupr': report['metrics']['macro']['AUPR'],
            'deltas': {
                name: report['comparison']['deltas'][name]['AUPR']
                for name in STATE_NAMES + ('macro',)
            },
            'records': {
                name: report['metrics'][name]['records']
                for name in STATE_NAMES
            },
            'passed': passed,
        })
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': 'frozen_base_hctx_router_outer_four_dataset_gate',
        'all_passed': all(row['passed'] for row in rows),
        'datasets': rows,
    }


def markdown(summary):
    lines = [
        '# Frozen-Base Hctx-P Outer Four-Dataset Gate',
        '',
        '- Protocol: `%s`' % summary['protocol'],
        '- All datasets passed: `%s`' % summary['all_passed'],
        '- Parameter selection on outer units: `False`',
        '',
        '| Dataset | NoContext Macro-AUPR | V3 Macro-AUPR | Delta | '
        'WW Delta | CW Delta | WC Delta | CC Delta | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in summary['datasets']:
        delta = row['deltas']
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %+.6f | %+.6f | '
            '%+.6f | %+.6f | %s |' % (
                row['dataset'],
                row['baseline_macro_aupr'],
                row['candidate_macro_aupr'],
                delta['macro'],
                delta['warm_warm'],
                delta['cold_warm'],
                delta['warm_cold'],
                delta['cold_cold'],
                'PASS' if row['passed'] else 'FAIL',
            )
        )
    lines.extend([
        '',
        'All model parameters and Hctx-P heads were frozen before this '
        'outer-unit evaluation.',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    summary = summarize(args.report)
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
    if args.require_all_pass and not summary['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
