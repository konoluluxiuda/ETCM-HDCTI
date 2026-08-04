#!/usr/bin/env python3
"""Combine historical c0p0 and confirmed c1p1-c4p4 outer units."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path


STATE_NAMES = (
    'warm_warm',
    'cold_warm',
    'warm_cold',
    'cold_cold',
)
HISTORICAL_PROTOCOL = 'frozen_base_hctx_router_outer_four_dataset_gate'
REPEATED_PROTOCOL = 'frozen_base_hctx_router_repeated_outer_gate_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-summary', required=True)
    parser.add_argument('--repeated-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values):
    return {
        'mean': float(statistics.fmean(values)),
        'std': float(statistics.pstdev(values)),
    }


def validate_outer_report(path):
    path = Path(path).expanduser().resolve()
    report = load_json(path)
    if report.get('evaluation') != (
            'frozen_base_hctx_router_outer_pure_inference'):
        raise ValueError('Unexpected outer report: %s' % path)
    if report['training_optimizer_steps'] != 0:
        raise ValueError('Outer report contains optimizer steps: %s' % path)
    if report['parameter_selection_on_outer']:
        raise ValueError('Outer report selected parameters: %s' % path)
    if not all(report['preservation_checks'].values()):
        raise ValueError('Outer report failed preservation: %s' % path)
    return path, report


def summarize(historical, repeated, historical_path, repeated_path):
    if historical.get('protocol') != HISTORICAL_PROTOCOL:
        raise ValueError('Unexpected historical summary protocol.')
    if not historical.get('all_passed'):
        raise ValueError('Historical c0p0 Gate did not pass.')
    if repeated.get('protocol') != REPEATED_PROTOCOL:
        raise ValueError('Unexpected repeated summary protocol.')
    if not repeated.get('passed'):
        raise ValueError('Repeated c1p1-c4p4 Gate did not pass.')
    if repeated.get('outer_unit_count') != 16:
        raise ValueError('Repeated summary must contain exactly 16 units.')

    repeated_specs = {
        row['display_name']: row for row in repeated['datasets']
    }
    historical_specs = {
        row['dataset']: row for row in historical['datasets']
    }
    if set(repeated_specs) != set(historical_specs):
        raise ValueError('Historical and repeated dataset sets differ.')

    units_by_dataset = {name: [] for name in repeated_specs}
    for display_name, row in historical_specs.items():
        report_path, report = validate_outer_report(row['report'])
        units_by_dataset[display_name].append({
            'unit': 'c0p0',
            'report': str(report_path),
            'report_sha256': sha256_file(report_path),
            'baseline_macro_aupr': row['baseline_macro_aupr'],
            'candidate_macro_aupr': row['candidate_macro_aupr'],
            'deltas': dict(row['deltas']),
        })
        if report['comparison']['deltas']['macro']['AUPR'] != (
                row['deltas']['macro']):
            raise ValueError('Historical summary/report delta mismatch.')

    for row in repeated['units']:
        display_name = row['display_name']
        if display_name not in units_by_dataset:
            raise ValueError('Unexpected repeated dataset: %s' % display_name)
        report_path, report = validate_outer_report(row['report'])
        unit = 'c%dp%d' % (
            int(row['compound_group']), int(row['protein_group'])
        )
        units_by_dataset[display_name].append({
            'unit': unit,
            'report': str(report_path),
            'report_sha256': sha256_file(report_path),
            'baseline_macro_aupr': row['baseline_macro_aupr'],
            'candidate_macro_aupr': row['candidate_macro_aupr'],
            'deltas': dict(row['deltas']),
        })
        if report['comparison']['deltas']['macro']['AUPR'] != (
                row['deltas']['macro']):
            raise ValueError('Repeated summary/report delta mismatch.')

    datasets = []
    all_units = []
    for display_name in repeated_specs:
        units = sorted(
            units_by_dataset[display_name], key=lambda row: row['unit']
        )
        if [row['unit'] for row in units] != [
                'c0p0', 'c1p1', 'c2p2', 'c3p3', 'c4p4']:
            raise ValueError(
                '%s does not contain c0p0-c4p4.' % display_name
            )
        datasets.append({
            'display_name': display_name,
            'units': 5,
            'baseline_macro_aupr': mean_std([
                row['baseline_macro_aupr'] for row in units
            ]),
            'candidate_macro_aupr': mean_std([
                row['candidate_macro_aupr'] for row in units
            ]),
            'delta': {
                name: mean_std([
                    row['deltas'][name] for row in units
                ])
                for name in STATE_NAMES + ('macro',)
            },
            'positive_macro_units': sum(
                row['deltas']['macro'] > 0.0 for row in units
            ),
            'unit_results': units,
        })
        all_units.extend(units)

    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': 'frozen_base_hctx_router_five_unit_descriptive_v1',
        'analysis_role': (
            'descriptive_c0p0_through_c4p4; confirmatory_gate_uses_only_'
            'c1p1_through_c4p4'
        ),
        'sources': {
            'historical_summary': str(historical_path),
            'historical_summary_sha256': sha256_file(historical_path),
            'repeated_summary': str(repeated_path),
            'repeated_summary_sha256': sha256_file(repeated_path),
        },
        'dataset_count': len(datasets),
        'unit_count': len(all_units),
        'overall_macro_aupr_delta': mean_std([
            row['deltas']['macro'] for row in all_units
        ]),
        'positive_macro_units': sum(
            row['deltas']['macro'] > 0.0 for row in all_units
        ),
        'datasets': datasets,
    }


def markdown(summary):
    lines = [
        '# Frozen-Base Hctx-P Five-Unit Descriptive Summary',
        '',
        '- Protocol: `%s`' % summary['protocol'],
        '- Role: historical `c0p0` plus confirmed `c1p1-c4p4`;',
        '  the preregistered Gate remains based only on the 16 new units.',
        '- Units: `%d`' % summary['unit_count'],
        '- Positive Macro-AUPR units: `%d/%d`' % (
            summary['positive_macro_units'], summary['unit_count']
        ),
        '- Overall Macro-AUPR delta: `%.6f (±%.6f)`' % (
            summary['overall_macro_aupr_delta']['mean'],
            summary['overall_macro_aupr_delta']['std'],
        ),
        '',
        '| Dataset | Units | NoContext Macro-AUPR | V3 Macro-AUPR | '
        'Delta | Positive | WW | CW | WC | CC |',
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
    lines.extend([
        '',
        'This table is descriptive. It does not redefine or rerun the '
        'preregistered 16-new-unit Gate.',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    historical_path = Path(args.historical_summary).expanduser().resolve()
    repeated_path = Path(args.repeated_summary).expanduser().resolve()
    summary = summarize(
        load_json(historical_path),
        load_json(repeated_path),
        historical_path,
        repeated_path,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'five_unit_summary.json'
    markdown_path = output_dir / 'five_unit_summary.md'
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    text = markdown(summary)
    markdown_path.write_text(text, encoding='utf-8')
    print(text)
    print('Five-unit summary written to: %s' % markdown_path)


if __name__ == '__main__':
    main()
