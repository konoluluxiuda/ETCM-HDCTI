#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.summarize_schpt_pilot import (
    METADATA_PATTERN,
    parse_validation_aupr,
)


def summarize(run_dir, manifest_path):
    run_dir = Path(run_dir)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = []
    for index, dataset in enumerate(manifest['datasets']):
        baseline_log = run_dir / ('%02d_%s_baseline.log' % (
            index * 2 + 1, dataset['slug']
        ))
        candidate_log = run_dir / ('%02d_%s_candidate.log' % (
            index * 2 + 2, dataset['slug']
        ))
        baseline_aupr, _ = parse_validation_aupr(baseline_log)
        candidate_aupr, candidate_text = parse_validation_aupr(candidate_log)
        metadata_matches = METADATA_PATTERN.findall(candidate_text)
        if not metadata_matches:
            raise ValueError(
                'No SCHPT metadata path found for %s.' % dataset['name']
            )
        metadata_path = Path(metadata_matches[-1].strip())
        if not metadata_path.is_absolute():
            metadata_path = Path.cwd() / metadata_path
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        rows.append({
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'baseline_validation_aupr': baseline_aupr,
            'candidate_validation_aupr': candidate_aupr,
            'validation_aupr_delta': candidate_aupr - baseline_aupr,
            'evidence_coverage': float(
                metadata['validation']['evidence_coverage']
            ),
            'mean_abs_residual': float(
                metadata['validation']['mean_abs_residual']
            ),
            'learned_scale': float(metadata['learned_scale']),
            'positive_mean_residual': metadata['positive_mean_residual'],
            'negative_mean_residual': metadata['negative_mean_residual'],
            'metadata_path': str(metadata_path.resolve()),
        })

    deltas = np.asarray([
        row['validation_aupr_delta'] for row in rows
    ], dtype=np.float64)
    gate = manifest['gate']
    checks = {
        'mean_delta_at_least_0.003': float(np.mean(deltas)) >= float(
            gate['minimum_mean_validation_aupr_delta']
        ),
        'at_least_3_of_4_datasets_positive': int(np.sum(deltas > 0)) >= int(
            gate['minimum_positive_dataset_count']
        ),
        'no_dataset_drop_below_minus_0.003': float(np.min(deltas)) >= float(
            gate['minimum_dataset_validation_aupr_delta']
        ),
        'all_dataset_coverage_at_least_0.30': all(
            row['evidence_coverage'] >= float(
                gate['minimum_evidence_coverage_each_dataset']
            ) for row in rows
        ),
        'all_dataset_scales_nonzero': all(
            abs(row['learned_scale']) > float(
                gate['minimum_absolute_learned_scale_each_dataset']
            ) for row in rows
        ),
    }
    return {
        'protocol': manifest['protocol'],
        'manifest': str(manifest_path.resolve()),
        'rows': rows,
        'mean_validation_aupr_delta': float(np.mean(deltas)),
        'positive_dataset_count': int(np.sum(deltas > 0)),
        'checks': checks,
        'gate': 'PASS' if all(checks.values()) else 'NO-GO',
    }


def markdown(report):
    lines = [
        '# SCHPT Four-Dataset Gate 1',
        '',
        '- Protocol: frozen seed 52026, fold-1 inner validation only',
        '- Outer-test parameter selection: disabled',
        '- Mean Validation-AUPR delta: `%+.6f`' % report[
            'mean_validation_aupr_delta'
        ],
        '- Positive datasets: `%d/4`' % report['positive_dataset_count'],
        '- Gate: `%s`' % report['gate'],
        '',
        '| Dataset | Baseline AUPR | SCHPT AUPR | Delta | Coverage | Scale | Pos residual | Neg residual |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['rows']:
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %.4f | %.6f | %.6f | %.6f |' % (
                row['dataset'],
                row['baseline_validation_aupr'],
                row['candidate_validation_aupr'],
                row['validation_aupr_delta'],
                row['evidence_coverage'],
                row['learned_scale'],
                row['positive_mean_residual'],
                row['negative_mean_residual'],
            )
        )
    lines.extend(['', '## Gate Checks', ''])
    lines.extend(
        '- `%s`: `%s`' % (name, value)
        for name, value in report['checks'].items()
    )
    lines.extend([
        '',
        'A PASS permits frozen five-fold confirmation. A NO-GO stops SCHPT '
        'without dataset-specific tuning.',
        '',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument(
        '--manifest', default='configs/schpt_gate1_manifest.json'
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = summarize(run_dir, args.manifest)
    (run_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (run_dir / 'summary.md').write_text(
        markdown(report), encoding='utf-8'
    )
    print('SCHPT four-dataset Gate 1: %s' % report['gate'])
    print('Summary: %s' % (run_dir / 'summary.md').resolve())
    return 0 if report['gate'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
