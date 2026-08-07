#!/usr/bin/env python3
"""Run and summarize the frozen full-candidate ranking Gate."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repository_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_prefix(metadata_path):
    prefix = metadata_path.parent / 'hdcti_model.ckpt'
    if not Path(str(prefix) + '.index').exists():
        raise FileNotFoundError('%s.index' % prefix)
    if not list(prefix.parent.glob(prefix.name + '.data-*')):
        raise FileNotFoundError('No data shard for %s.' % prefix)
    return prefix


def build_worklist(manifest):
    schpt_results_path = repository_path(manifest['schpt_results'])
    schpt_manifest_path = repository_path(manifest['schpt_manifest'])
    schpt_results = json.loads(schpt_results_path.read_text(encoding='utf-8'))
    schpt_manifest = json.loads(schpt_manifest_path.read_text(encoding='utf-8'))
    result_rows = {row['slug']: row for row in schpt_results['rows']}
    frozen_rows = {row['slug']: row for row in schpt_manifest['datasets']}
    jobs = []
    for dataset in manifest['datasets']:
        slug = dataset['slug']
        if slug not in result_rows or slug not in frozen_rows:
            raise ValueError('Missing frozen SCHPT metadata for %s.' % slug)
        config_path = repository_path(dataset['config'])
        expected_config_hash = frozen_rows[slug]['candidate_sha256']
        actual_config_hash = sha256_file(config_path)
        if actual_config_hash != expected_config_hash:
            raise ValueError('Config hash mismatch for %s.' % slug)
        metadata_paths = [
            repository_path(value) for value in result_rows[slug]['metadata_paths']
        ]
        if len(metadata_paths) != int(manifest['fold_count']):
            raise ValueError('Checkpoint count mismatch for %s.' % slug)
        for fold, metadata_path in enumerate(metadata_paths, start=1):
            prefix = checkpoint_prefix(metadata_path)
            jobs.append({
                'dataset': dataset['name'],
                'slug': slug,
                'fold': fold,
                'config': str(config_path),
                'config_sha256': actual_config_hash,
                'checkpoint': str(prefix),
                'checkpoint_index_sha256': sha256_file(str(prefix) + '.index'),
                'metadata': str(metadata_path),
                'metadata_sha256': sha256_file(metadata_path),
            })
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': manifest['protocol'],
        'source_results': str(schpt_results_path),
        'source_results_sha256': sha256_file(schpt_results_path),
        'source_manifest': str(schpt_manifest_path),
        'source_manifest_sha256': sha256_file(schpt_manifest_path),
        'jobs': jobs,
    }


def select_jobs(worklist, dataset=None, fold=None):
    jobs = worklist['jobs']
    if dataset:
        key = dataset.strip().lower()
        jobs = [
            job for job in jobs
            if job['slug'].lower() == key or job['dataset'].lower() == key
        ]
    if fold is not None:
        jobs = [job for job in jobs if job['fold'] == fold]
    if not jobs:
        raise ValueError('No jobs match the requested dataset/fold filter.')
    return jobs


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            handle.write(line)
        return process.wait()


def run_heuristics(python, manifest_path, output_dir):
    command = [
        python,
        str(REPOSITORY_ROOT / 'tools' / 'evaluate_full_candidate_heuristics.py'),
        '--manifest', str(manifest_path),
        '--output-dir', str(output_dir / 'heuristics'),
    ]
    status = run_command(command, output_dir / 'logs' / 'heuristics.log')
    if status:
        raise RuntimeError('Full-candidate heuristic evaluation failed: %d.' % status)


def run_ours_job(python, job, manifest, output_dir):
    job_dir = output_dir / 'ours' / job['slug'] / ('fold_%d' % job['fold'])
    command = [
        python,
        str(REPOSITORY_ROOT / 'tools' / 'evaluate_checkpoint_ranking.py'),
        '--config', job['config'],
        '--checkpoint', job['checkpoint'],
        '--fold', str(job['fold']),
        '--ks', *[str(value) for value in manifest['ks']],
        '--export-top', str(manifest['export_top']),
        '--output-dir', str(job_dir),
    ]
    status = run_command(
        command,
        output_dir / 'logs' / ('%s_fold_%d.log' % (job['slug'], job['fold'])),
    )
    if status:
        raise RuntimeError(
            'Checkpoint ranking failed for %s fold %d: %d.' %
            (job['dataset'], job['fold'], status))


def metric_mean(folds, metric):
    values = [float(row[metric]) for row in folds]
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'folds': values,
    }


def _value_summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'mean': float(np.mean(array)) if array.size else 0.0,
        'median': float(np.median(array)) if array.size else 0.0,
        'zero_rate': float(np.mean(array == 0)) if array.size else 0.0,
    }


def target_support_diagnostic(dataset, fold, report_dir):
    assignments_path = repository_path(dataset['split_dir']) / 'fold_assignments.tsv'
    training_degree = {}
    held_out_positive_degrees = []
    with open(assignments_path, encoding='utf-8') as handle:
        header = next(handle, '').rstrip('\n').split('\t')
        if header != ['left_id', 'right_id', 'label', 'fold']:
            raise ValueError('Invalid assignment header in %s.' % assignments_path)
        rows = [line.rstrip('\n').split('\t') for line in handle if line.strip()]
    test_fold = fold - 1
    for _, protein_id, label, fold_text in rows:
        if int(fold_text) != test_fold and int(label) == 1:
            training_degree[protein_id] = training_degree.get(protein_id, 0) + 1
    for _, protein_id, label, fold_text in rows:
        if int(fold_text) == test_fold and int(label) == 1:
            held_out_positive_degrees.append(training_degree.get(protein_id, 0))

    top_degrees = []
    top_unlabeled_degrees = []
    top_path = report_dir / 'top_candidates.tsv'
    with open(top_path, encoding='utf-8') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            degree = training_degree.get(row['protein_id'], 0)
            top_degrees.append(degree)
            if row['label_status'] == 'unlabeled':
                top_unlabeled_degrees.append(degree)
    return {
        'training_supported_proteins': len(training_degree),
        'held_out_positive_target_degree': _value_summary(
            held_out_positive_degrees),
        'exported_top_candidate_target_degree': _value_summary(top_degrees),
        'exported_top_unlabeled_target_degree': _value_summary(
            top_unlabeled_degrees),
    }


def summarize(manifest, worklist, output_dir):
    heuristic_path = output_dir / 'heuristics' / 'summary.json'
    if not heuristic_path.exists():
        raise FileNotFoundError('Missing heuristic report: %s' % heuristic_path)
    heuristic_report = json.loads(heuristic_path.read_text(encoding='utf-8'))
    heuristic_by_slug = {
        row['slug']: row for row in heuristic_report['datasets']
    }
    datasets = []
    complete = True
    for dataset in manifest['datasets']:
        slug = dataset['slug']
        ours_folds = []
        for fold in range(1, int(manifest['fold_count']) + 1):
            report_path = output_dir / 'ours' / slug / ('fold_%d' % fold) / 'report.json'
            if not report_path.exists():
                complete = False
                continue
            report = json.loads(report_path.read_text(encoding='utf-8'))
            metrics = report['fixed_candidate_metrics']
            support = target_support_diagnostic(
                dataset, fold, report_path.parent)
            ours_folds.append({
                'fold': fold,
                **{
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
                'report': str(report_path),
                'target_support_diagnostic': support,
            })
        if not ours_folds:
            continue
        ours_summary = {
            metric: metric_mean(ours_folds, metric)
            for metric in ('MRR', 'Recall@20', 'Hits@20', 'Recall@50', 'Hits@50')
        }
        heuristic = heuristic_by_slug[slug]
        comparisons = {}
        for metric in ('MRR', 'Recall@20', 'Hits@20', 'Recall@50', 'Hits@50'):
            completed_fold_indices = [row['fold'] - 1 for row in ours_folds]
            methods = {}
            for method in heuristic_report['methods']:
                method_values = [
                    float(heuristic['folds'][index]['methods'][method]['metrics'][metric])
                    for index in completed_fold_indices
                ]
                methods[method] = float(np.mean(method_values))
            best_method = max(methods, key=methods.get)
            comparisons[metric] = {
                'ours': ours_summary[metric]['mean'],
                'best_heuristic': best_method,
                'best_heuristic_value': float(methods[best_method]),
                'delta': ours_summary[metric]['mean'] - float(methods[best_method]),
            }
        datasets.append({
            'name': dataset['name'],
            'slug': slug,
            'completed_folds': len(ours_folds),
            'ours_folds': ours_folds,
            'ours_summary': ours_summary,
            'heuristic_summary': heuristic['summary'],
            'comparisons': comparisons,
            'target_support_summary': {
                'top_candidate_zero_rate': float(np.mean([
                    row['target_support_diagnostic'][
                        'exported_top_candidate_target_degree']['zero_rate']
                    for row in ours_folds
                ])),
                'held_out_positive_mean_degree': float(np.mean([
                    row['target_support_diagnostic'][
                        'held_out_positive_target_degree']['mean']
                    for row in ours_folds
                ])),
                'top_candidate_mean_degree': float(np.mean([
                    row['target_support_diagnostic'][
                        'exported_top_candidate_target_degree']['mean']
                    for row in ours_folds
                ])),
            },
        })

    expected_jobs = len(manifest['datasets']) * int(manifest['fold_count'])
    completed_jobs = sum(row['completed_folds'] for row in datasets)
    complete = complete and completed_jobs == expected_jobs
    gate_checks = None
    gate_status = 'INCOMPLETE'
    macro = None
    if complete:
        macro = {}
        for metric in ('MRR', 'Recall@20', 'Hits@20', 'Recall@50', 'Hits@50'):
            macro[metric] = {
                'ours': float(np.mean([
                    row['ours_summary'][metric]['mean'] for row in datasets
                ])),
                'best_heuristic_per_dataset': float(np.mean([
                    row['comparisons'][metric]['best_heuristic_value']
                    for row in datasets
                ])),
            }
            macro[metric]['delta'] = (
                macro[metric]['ours'] - macro[metric]['best_heuristic_per_dataset'])
        gate = manifest['gate']
        mrr_deltas = [row['comparisons']['MRR']['delta'] for row in datasets]
        gate_checks = {
            'macro_mrr_delta': (
                macro['MRR']['delta'] >=
                float(gate['minimum_macro_mrr_delta_vs_best_heuristic'])
            ),
            'macro_recall_at_20_delta': (
                macro['Recall@20']['delta'] >=
                float(gate['minimum_macro_recall_at_20_delta_vs_best_heuristic'])
            ),
            'dataset_mrr_wins': (
                sum(value > 0 for value in mrr_deltas) >=
                int(gate['minimum_dataset_mrr_wins'])
            ),
            'maximum_dataset_mrr_drop': (
                min(mrr_deltas) >= float(gate['minimum_dataset_mrr_delta'])
            ),
        }
        gate_status = 'PASS' if all(gate_checks.values()) else 'FAIL'

    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': manifest['protocol'],
        'worklist_sha256': sha256_file(output_dir / 'worklist.json'),
        'heuristic_report': str(heuristic_path),
        'heuristic_report_sha256': sha256_file(heuristic_path),
        'expected_jobs': expected_jobs,
        'completed_jobs': completed_jobs,
        'complete': complete,
        'datasets': datasets,
        'macro': macro,
        'gate_checks': gate_checks,
        'gate': gate_status,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'summary.md').write_text(
        render_markdown(report), encoding='utf-8')
    return report


def render_markdown(report):
    lines = [
        '# Frozen Full-Candidate Ranking Gate',
        '',
        '- Protocol: `%s`' % report['protocol'],
        '- Completed checkpoint jobs: `%d/%d`' % (
            report['completed_jobs'], report['expected_jobs']),
        '- Gate: `%s`' % report['gate'],
        '',
        '| Dataset | Folds | Ours MRR | Best heuristic MRR | Delta | Ours Recall@20 | Best heuristic Recall@20 | Delta |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in report['datasets']:
        mrr = dataset['comparisons']['MRR']
        recall = dataset['comparisons']['Recall@20']
        lines.append(
            '| %s | %d | %.6f | %s %.6f | %+.6f | %.6f | %s %.6f | %+.6f |' % (
                dataset['name'],
                dataset['completed_folds'],
                mrr['ours'],
                mrr['best_heuristic'],
                mrr['best_heuristic_value'],
                mrr['delta'],
                recall['ours'],
                recall['best_heuristic'],
                recall['best_heuristic_value'],
                recall['delta'],
            )
        )
    if report['macro'] is not None:
        lines.extend([
            '| **Macro** | 20 | %.6f | best-per-dataset %.6f | %+.6f | %.6f | best-per-dataset %.6f | %+.6f |' % (
                report['macro']['MRR']['ours'],
                report['macro']['MRR']['best_heuristic_per_dataset'],
                report['macro']['MRR']['delta'],
                report['macro']['Recall@20']['ours'],
                report['macro']['Recall@20']['best_heuristic_per_dataset'],
                report['macro']['Recall@20']['delta'],
            ),
            '',
            '## Gate Checks',
            '',
        ])
        for key, value in report['gate_checks'].items():
            lines.append('- `%s`: `%s`' % (key, value))
    lines.extend([
        '',
        'All checkpoint evaluations are pure inference. Unobserved pairs are unlabeled.',
        '',
    ])
    if report['datasets']:
        lines.extend([
            '## Target-Support Diagnostic',
            '',
            '| Dataset | Top candidate zero-support rate | Held-out positive mean degree | Top candidate mean degree |',
            '|---|---:|---:|---:|',
        ])
        for dataset in report['datasets']:
            support = dataset['target_support_summary']
            lines.append('| %s | %.2f%% | %.2f | %.2f |' % (
                dataset['name'],
                100.0 * support['top_candidate_zero_rate'],
                support['held_out_positive_mean_degree'],
                support['top_candidate_mean_degree'],
            ))
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', default='configs/full_candidate_ranking_gate.json')
    parser.add_argument(
        '--output-dir', default='results/full_candidate_ranking_gate')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dataset')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--heuristics-only', action='store_true')
    parser.add_argument('--ours-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.heuristics_only and args.ours_only:
        raise ValueError('--heuristics-only and --ours-only are mutually exclusive.')

    manifest_path = repository_path(args.manifest)
    output_dir = repository_path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    worklist = build_worklist(manifest)
    jobs = select_jobs(worklist, dataset=args.dataset, fold=args.fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'worklist.json').write_text(
        json.dumps(worklist, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.dry_run:
        print(json.dumps({
            'python': args.python,
            'output_dir': str(output_dir),
            'selected_jobs': jobs,
        }, ensure_ascii=False, indent=2))
        return 0

    if not args.summarize_only:
        heuristic_summary = output_dir / 'heuristics' / 'summary.json'
        if not args.ours_only and (args.force or not heuristic_summary.exists()):
            run_heuristics(args.python, manifest_path, output_dir)
        if not args.heuristics_only:
            for position, job in enumerate(jobs, start=1):
                report_path = (
                    output_dir / 'ours' / job['slug'] /
                    ('fold_%d' % job['fold']) / 'report.json'
                )
                if report_path.exists() and not args.force:
                    print('[%d/%d] Reusing %s fold %d report.' % (
                        position, len(jobs), job['dataset'], job['fold']))
                    continue
                print('[%d/%d] Evaluating %s fold %d.' % (
                    position, len(jobs), job['dataset'], job['fold']))
                run_ours_job(args.python, job, manifest, output_dir)

    heuristic_summary = output_dir / 'heuristics' / 'summary.json'
    if heuristic_summary.exists():
        report = summarize(manifest, worklist, output_dir)
        print(render_markdown(report))
        print('Summary written to: %s' % (output_dir / 'summary.md'))
    else:
        print('Heuristic summary is absent; combined summary was not generated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
