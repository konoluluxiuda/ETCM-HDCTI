#!/usr/bin/env python3
"""Run a validation-only full-candidate ranking headroom audit."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.run_full_candidate_ranking_gate import (  # noqa: E402
    build_worklist,
    repository_path,
    run_command,
    select_jobs,
    sha256_file,
)


def run_job(python, job, audit, output_dir):
    job_dir = output_dir / 'ours' / job['slug'] / ('fold_%d' % job['fold'])
    command = [
        python,
        str(REPOSITORY_ROOT / 'tools' / 'evaluate_checkpoint_ranking.py'),
        '--config', job['config'],
        '--checkpoint', job['checkpoint'],
        '--fold', str(job['fold']),
        '--evaluation-split', 'validation',
        '--ks', *[str(value) for value in audit['ks']],
        '--export-top', str(audit['export_top']),
        '--output-dir', str(job_dir),
    ]
    status = run_command(
        command,
        output_dir / 'logs' / ('%s_fold_%d.log' % (job['slug'], job['fold'])),
    )
    if status:
        raise RuntimeError(
            'Inner ranking audit failed for %s fold %d: %d.' %
            (job['dataset'], job['fold'], status))


def render_markdown(report):
    lines = [
        '# Inner-Validation Full-Candidate Ranking Headroom',
        '',
        '- Protocol: `%s`' % report['protocol'],
        '- Outer test scored: `False`',
        '- Optimizer steps: `0`',
        '- Gate: `%s`' % report['gate'],
        '',
        '| Dataset | Fold | Ours MRR | Best heuristic MRR | Gap | Ours Recall@20 | Best heuristic Recall@20 | Gap | Headroom |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in report['rows']:
        lines.append(
            '| %s | %d | %.6f | %s %.6f | %+.6f | %.6f | %s %.6f | %+.6f | %s |' % (
                row['dataset'],
                row['fold'],
                row['ours']['MRR'],
                row['best_heuristic']['MRR']['method'],
                row['best_heuristic']['MRR']['value'],
                row['gaps']['MRR'],
                row['ours']['Recall@20'],
                row['best_heuristic']['Recall@20']['method'],
                row['best_heuristic']['Recall@20']['value'],
                row['gaps']['Recall@20'],
                row['headroom'],
            )
        )
    lines.extend([
        '',
        '## Gate Checks',
        '',
    ])
    for key, value in report['checks'].items():
        lines.append('- `%s`: `%s`' % (key, value))
    lines.extend([
        '',
        'A PASS only permits one fixed ranking-loss Pilot. It is not evidence of model improvement.',
        '',
    ])
    return '\n'.join(lines)


def summarize(audit, parent, worklist, output_dir):
    heuristic_path = output_dir / 'heuristics' / 'summary.json'
    heuristics = json.loads(heuristic_path.read_text(encoding='utf-8'))
    if heuristics.get('evaluation_split') != 'validation':
        raise ValueError('Heuristic report is not validation-only.')
    heuristic_by_slug = {row['slug']: row for row in heuristics['datasets']}
    rows = []
    for job in worklist['jobs']:
        if job['fold'] not in audit['folds']:
            continue
        report_path = (
            output_dir / 'ours' / job['slug'] /
            ('fold_%d' % job['fold']) / 'report.json'
        )
        if not report_path.exists():
            continue
        model_report = json.loads(report_path.read_text(encoding='utf-8'))
        if model_report['split'].get('outer_test_scored'):
            raise ValueError('Outer test was scored in %s.' % report_path)
        metrics = model_report['fixed_candidate_metrics']
        heuristic_fold = heuristic_by_slug[job['slug']]['folds'][job['fold'] - 1]
        best = {}
        gaps = {}
        for metric in ('MRR', 'Recall@20'):
            values = {
                method: float(heuristic_fold['methods'][method]['metrics'][metric])
                for method in heuristics['methods']
            }
            method = max(values, key=values.get)
            best[metric] = {'method': method, 'value': values[method]}
            gaps[metric] = float(metrics[metric]) - values[method]
        gate = audit['gate']
        headroom = (
            gaps['MRR'] <= -float(gate['minimum_mrr_gap'])
            or gaps['Recall@20'] <= -float(gate['minimum_recall_at_20_gap'])
        )
        rows.append({
            'dataset': job['dataset'],
            'slug': job['slug'],
            'fold': job['fold'],
            'ours': {
                'MRR': float(metrics['MRR']),
                'Recall@20': float(metrics['Recall@20']),
            },
            'best_heuristic': best,
            'gaps': gaps,
            'headroom': bool(headroom),
            'report': str(report_path),
        })
    expected = len(parent['datasets']) * len(audit['folds'])
    headroom_count = sum(row['headroom'] for row in rows)
    checks = {
        'expected_validation_units': len(rows) == expected,
        'all_outer_test_scored_false': all(
            not json.loads(Path(row['report']).read_text(encoding='utf-8'))[
                'split']['outer_test_scored']
            for row in rows
        ),
        'minimum_dataset_count_with_headroom': (
            headroom_count >=
            int(audit['gate']['minimum_dataset_count_with_headroom'])
        ),
    }
    status = 'PASS' if all(checks.values()) else 'FAIL'
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': audit['protocol'],
        'audit_manifest': str(repository_path(
            'configs/inner_full_candidate_ranking_audit.json')),
        'parent_manifest': str(repository_path(audit['parent_manifest'])),
        'heuristic_report': str(heuristic_path),
        'heuristic_report_sha256': sha256_file(heuristic_path),
        'rows': rows,
        'headroom_dataset_count': int(headroom_count),
        'checks': checks,
        'gate': status,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'summary.md').write_text(
        render_markdown(report), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--audit-manifest',
        default='configs/inner_full_candidate_ranking_audit.json',
    )
    parser.add_argument(
        '--output-dir', default='results/inner_full_candidate_ranking_audit')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dataset')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    audit_path = repository_path(args.audit_manifest)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    parent_path = repository_path(audit['parent_manifest'])
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    worklist = build_worklist(parent)
    jobs = select_jobs(
        worklist,
        dataset=args.dataset,
        fold=args.fold,
    )
    jobs = [job for job in jobs if job['fold'] in audit['folds']]
    if not jobs:
        raise ValueError('No selected jobs are included in the audit folds.')
    output_dir = repository_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'worklist.json').write_text(
        json.dumps(worklist, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    if args.dry_run:
        print(json.dumps({
            'protocol': audit['protocol'],
            'outer_test_scored': False,
            'selected_jobs': jobs,
        }, ensure_ascii=False, indent=2))
        return 0

    if not args.summarize_only:
        for position, job in enumerate(jobs, start=1):
            report_path = (
                output_dir / 'ours' / job['slug'] /
                ('fold_%d' % job['fold']) / 'report.json'
            )
            if report_path.exists() and not args.force:
                print('[%d/%d] Reusing %s fold %d.' % (
                    position, len(jobs), job['dataset'], job['fold']))
                continue
            print('[%d/%d] Auditing %s fold %d inner validation.' % (
                position, len(jobs), job['dataset'], job['fold']))
            run_job(args.python, job, audit, output_dir)

    report = summarize(audit, parent, worklist, output_dir)
    print(render_markdown(report))
    print('Summary written to: %s' % (output_dir / 'summary.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
