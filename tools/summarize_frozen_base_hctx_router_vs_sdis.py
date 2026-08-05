#!/usr/bin/env python3
"""Summarize matched repeated-unit V3 versus joint Hctx-P + SDIS."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

STATE_NAMES = ('warm_warm', 'cold_warm', 'warm_cold', 'cold_cold')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mean_std(values):
    values = [float(value) for value in values]
    return {
        'mean': statistics.fmean(values),
        'std': statistics.pstdev(values),
    }


def validate_v3_report(report, job):
    if report.get('protocol') != (
            'frozen_base_hctx_router_repeated_outer_evaluation_v1'):
        raise ValueError('Unexpected V3 outer report protocol.')
    if report.get('training_optimizer_steps') != 0:
        raise ValueError('V3 outer evaluation performed training.')
    if report.get('parameter_selection_on_outer'):
        raise ValueError('V3 outer evaluation selected parameters.')
    if report['support_unit']['assignments_sha256'] != job[
            'assignments_sha256']:
        raise ValueError('V3 support assignment mismatch: %s' % job['job_key'])


def summarize(plan, manifest, v3_summary_path, sdis_report_paths):
    if plan.get('protocol') != (
            'frozen_base_hctx_router_vs_sdis_preregistration_v1'):
        raise ValueError('Unexpected V3/SDIS plan protocol.')
    if manifest.get('protocol') != (
            'frozen_base_hctx_router_vs_sdis_units_v1'):
        raise ValueError('Unexpected V3/SDIS unit manifest protocol.')
    v3 = load_json(v3_summary_path)
    if v3.get('protocol') != (
            'frozen_base_hctx_router_repeated_outer_gate_v1'):
        raise ValueError('Unexpected frozen V3 summary protocol.')
    if not v3.get('passed') or v3.get('outer_unit_count') != 16:
        raise ValueError('Frozen V3 confirmation did not pass its Gate.')
    v3_units = {row['job_key']: row for row in v3['units']}
    jobs = {row['job_key']: row for row in manifest['jobs']}
    reports = {}
    report_paths = {}
    for path in sdis_report_paths:
        path = Path(path)
        job_key = path.parent.name
        if job_key not in jobs or job_key in reports:
            raise ValueError('Unexpected or duplicate SDIS report: %s' % path)
        report = load_json(path)
        if report.get('records') != 'outer':
            raise ValueError('SDIS report did not evaluate outer records.')
        if report.get('training_optimizer_steps') != 0:
            raise ValueError('SDIS outer evaluation performed training.')
        if report.get('parameter_selection_on_records'):
            raise ValueError('SDIS outer evaluation selected parameters.')
        if report['support_unit']['assignments_sha256'] != jobs[job_key][
                'assignments_sha256']:
            raise ValueError('SDIS support assignment mismatch: %s' % job_key)
        reports[job_key] = report
        report_paths[job_key] = str(path.resolve())
    if set(reports) != set(jobs) or set(v3_units) != set(jobs):
        raise ValueError('V3/SDIS comparison is not complete for all units.')

    rows = []
    for job_key, job in jobs.items():
        v3_report = load_json(v3_units[job_key]['report'])
        validate_v3_report(v3_report, job)
        sdis_report = reports[job_key]
        deltas = {}
        for state in STATE_NAMES + ('macro',):
            deltas[state] = {
                metric: (
                    v3_report['metrics'][state][metric]
                    - sdis_report['metrics'][state][metric]
                )
                for metric in ('AUPR', 'AUC')
            }
        rows.append({
            'job_key': job_key,
            'dataset': job['dataset'],
            'display_name': job['display_name'],
            'compound_group': job['compound_group'],
            'protein_group': job['protein_group'],
            'v3_report': v3_units[job_key]['report'],
            'sdis_report': report_paths[job_key],
            'v3_macro_aupr': v3_report['metrics']['macro']['AUPR'],
            'sdis_macro_aupr': sdis_report['metrics']['macro']['AUPR'],
            'deltas': deltas,
        })
    rows.sort(key=lambda row: (row['dataset'], row['compound_group']))

    dataset_rows = []
    for dataset in sorted({row['dataset'] for row in rows}):
        selected = [row for row in rows if row['dataset'] == dataset]
        state_deltas = {
            state: mean_std([
                row['deltas'][state]['AUPR'] for row in selected
            ])
            for state in STATE_NAMES + ('macro',)
        }
        dataset_rows.append({
            'dataset': dataset,
            'display_name': selected[0]['display_name'],
            'units': len(selected),
            'v3_macro_aupr': mean_std([
                row['v3_macro_aupr'] for row in selected
            ]),
            'sdis_macro_aupr': mean_std([
                row['sdis_macro_aupr'] for row in selected
            ]),
            'delta': state_deltas,
            'positive_units': sum(
                row['deltas']['macro']['AUPR'] > 0 for row in selected
            ),
        })

    gate = plan['decision_gate']
    overall = statistics.fmean(
        row['deltas']['macro']['AUPR'] for row in rows
    )
    positive = sum(row['deltas']['macro']['AUPR'] > 0 for row in rows)
    noninferiority = {
        'expected_unit_count': len(rows) == int(gate['unit_count']),
        'overall_mean_delta': overall >= float(
            gate['noninferiority']['overall_mean_macro_aupr_delta_minimum']
        ),
        'each_dataset_mean_delta': all(
            row['delta']['macro']['mean'] >= float(
                gate['noninferiority'][
                    'each_dataset_mean_macro_aupr_delta_minimum'
                ]
            ) for row in dataset_rows
        ),
        'maximum_dataset_state_drop': all(
            row['delta'][state]['mean'] >= -float(
                gate['noninferiority']['maximum_dataset_mean_state_aupr_drop']
            )
            for row in dataset_rows for state in STATE_NAMES
        ),
    }
    superiority = {
        'overall_mean_delta': overall >= float(
            gate['superiority']['overall_mean_macro_aupr_delta_minimum']
        ),
        'minimum_positive_units': positive >= int(
            gate['superiority']['minimum_positive_units']
        ),
        'each_dataset_mean_delta': all(
            row['delta']['macro']['mean'] >= float(
                gate['superiority'][
                    'each_dataset_mean_macro_aupr_delta_minimum'
                ]
            ) for row in dataset_rows
        ),
    }
    noninferiority_passed = all(noninferiority.values())
    superiority_passed = noninferiority_passed and all(superiority.values())
    if superiority_passed:
        decision = 'superiority_pass'
    elif noninferiority_passed:
        decision = 'noninferiority_only'
    else:
        decision = 'noninferiority_fail'
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': 'frozen_base_hctx_router_vs_sdis_outer_gate_v1',
        'unit_count': len(rows),
        'overall_mean_macro_aupr_delta_v3_minus_sdis': overall,
        'positive_units': positive,
        'noninferiority_checks': noninferiority,
        'superiority_checks': superiority,
        'noninferiority_passed': noninferiority_passed,
        'superiority_passed': superiority_passed,
        'decision': decision,
        'decision_text': plan['decision_rule'][decision],
        'datasets': dataset_rows,
        'units': rows,
    }


def markdown(summary):
    lines = [
        '# Frozen-Base Router versus Hctx-P + SDIS', '',
        '- Units: `%d`' % summary['unit_count'],
        '- Overall Macro-AUPR delta (V3 - SDIS): `%+.6f`' % summary[
            'overall_mean_macro_aupr_delta_v3_minus_sdis'
        ],
        '- Positive units: `%d/%d`' % (
            summary['positive_units'], summary['unit_count']
        ),
        '- Noninferiority: `%s`' % summary['noninferiority_passed'],
        '- Superiority: `%s`' % summary['superiority_passed'],
        '- Decision: `%s`' % summary['decision'], '',
        '| Dataset | Units | Hctx-P+SDIS Macro-AUPR | V3 Macro-AUPR | Delta | Positive | WW | CW | WC | CC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary['datasets']:
        lines.append(
            '| %s | %d | %.6f (±%.6f) | %.6f (±%.6f) | %+.6f (±%.6f) | %d/%d | %+.6f | %+.6f | %+.6f | %+.6f |'
            % (
                row['display_name'], row['units'],
                row['sdis_macro_aupr']['mean'], row['sdis_macro_aupr']['std'],
                row['v3_macro_aupr']['mean'], row['v3_macro_aupr']['std'],
                row['delta']['macro']['mean'], row['delta']['macro']['std'],
                row['positive_units'], row['units'],
                row['delta']['warm_warm']['mean'],
                row['delta']['cold_warm']['mean'],
                row['delta']['warm_cold']['mean'],
                row['delta']['cold_cold']['mean'],
            )
        )
    lines.extend(['', summary['decision_text'], '', '## Gate checks', ''])
    for group in ('noninferiority_checks', 'superiority_checks'):
        lines.append('### ' + group.replace('_', ' ').title())
        lines.append('')
        for name, passed in summary[group].items():
            lines.append('- `%s`: `%s`' % (name, passed))
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sdis-report', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan = load_json(args.plan)
    repository_root = Path(__file__).resolve().parents[1]
    v3_summary_path = Path(plan['v3_repeated_summary']).expanduser()
    if not v3_summary_path.is_absolute():
        v3_summary_path = repository_root / v3_summary_path
    summary = summarize(
        plan, load_json(args.manifest), v3_summary_path.resolve(),
        args.sdis_report,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    text = markdown(summary)
    (output / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
