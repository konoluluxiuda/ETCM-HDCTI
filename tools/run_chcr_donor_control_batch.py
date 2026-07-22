#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / 'configs' / 'chcr_donor_control_checkpoints.json'
)
AUDIT_SCRIPT = REPOSITORY_ROOT / 'tools' / 'audit_chcr_donor_controls.py'
SUPPORT_DECISIONS = {
    'supports_context_specificity_beyond_degree_and_overlap',
    'supports_context_specificity_beyond_degree_disjoint_confirmed',
    'supports_context_specificity_beyond_degree_disjoint_confirmed_overlap_inconclusive',
    'supports_context_specificity_beyond_degree_disjoint_coverage_inconclusive',
    'supports_context_specificity_beyond_degree_disjoint_not_confirmed',
}
RESULT_FIELDS = (
    'dataset',
    'slug',
    'fold',
    'status',
    'decision',
    'common_coverage',
    'degree_control_coverage',
    'factual_AUPR',
    'degree_control_AUPR_drop',
    'degree_control_positive_margin',
    'degree_control_pair_win_rate',
    'disjoint_AUPR_drop',
    'disjoint_positive_margin',
    'disjoint_pair_win_rate',
    'degree_control_overlap',
    'report',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the frozen CHCR donor-control inference audit over a checked '
            'four-dataset checkpoint manifest.'
        )
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--output-dir')
    parser.add_argument('--dataset', action='append', default=[])
    parser.add_argument('--fold', action='append', type=int, default=[])
    parser.add_argument('--python-bin', default=os.environ.get(
        'PYTHON_BIN', sys.executable
    ))
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate manifest, configs, hashes, and checkpoint files only.',
    )
    parser.add_argument(
        '--protocol-dry-run', action='store_true',
        help='Also invoke each selected single-fold audit with --dry-run.',
    )
    parser.add_argument(
        '--summarize-only', action='store_true',
        help='Rebuild the cross-dataset summary from existing report.json files.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Replace an existing report that does not match the frozen job.',
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def executable_path(value):
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path('.'):
        resolved = repository_path(path)
        if not resolved.is_file():
            raise FileNotFoundError('Missing executable: %s' % resolved)
        return resolved
    resolved = shutil.which(str(value))
    if not resolved:
        raise FileNotFoundError('Executable not found on PATH: %s' % value)
    return Path(resolved).resolve()


def checkpoint_files(prefix):
    prefix = repository_path(prefix)
    index_path = Path(str(prefix) + '.index')
    data_paths = sorted(prefix.parent.glob(prefix.name + '.data-*'))
    return prefix, index_path, data_paths


def load_manifest(path):
    manifest_path = repository_path(path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint manifest schema.')
    fold_count = int(manifest.get('fold_count', 0))
    if fold_count <= 0:
        raise ValueError('Manifest fold_count must be positive.')
    if int(manifest.get('draws', 0)) <= 0:
        raise ValueError('Manifest draws must be positive.')
    datasets = manifest.get('datasets') or []
    if not datasets:
        raise ValueError('Manifest contains no datasets.')

    jobs = []
    seen_slugs = set()
    seen_jobs = set()
    for dataset in datasets:
        name = str(dataset['name'])
        slug = str(dataset['slug'])
        if slug in seen_slugs:
            raise ValueError('Duplicate dataset slug: %s' % slug)
        seen_slugs.add(slug)
        config = repository_path(dataset['config'])
        if not config.is_file():
            raise FileNotFoundError('Missing config: %s' % config)
        actual_hash = sha256_file(config)
        if actual_hash != dataset['config_sha256']:
            raise ValueError(
                'Config hash mismatch for %s: %s != %s' % (
                    name, actual_hash, dataset['config_sha256']
                )
            )
        checkpoints = dataset.get('checkpoints') or []
        source_log = repository_path(dataset['source_log'])
        if not source_log.is_file():
            raise FileNotFoundError('Missing source log: %s' % source_log)
        folds = sorted(int(item['fold']) for item in checkpoints)
        if folds != list(range(1, fold_count + 1)):
            raise ValueError(
                '%s checkpoint folds must be exactly 1..%d.' % (
                    name, fold_count
                )
            )
        for item in checkpoints:
            fold = int(item['fold'])
            job_key = (slug, fold)
            if job_key in seen_jobs:
                raise ValueError('Duplicate job: %s fold %d' % job_key)
            seen_jobs.add(job_key)
            prefix, index_path, data_paths = checkpoint_files(item['path'])
            if not index_path.is_file() or not data_paths:
                raise FileNotFoundError(
                    'Incomplete checkpoint for %s fold %d: %s' % (
                        name, fold, prefix
                    )
                )
            jobs.append({
                'dataset': name,
                'slug': slug,
                'fold': fold,
                'config': config,
                'config_sha256': actual_hash,
                'checkpoint': prefix,
                'source_log': source_log,
            })
    manifest['_path'] = manifest_path
    return manifest, jobs


def select_jobs(jobs, datasets=None, folds=None):
    dataset_filters = {value.lower() for value in (datasets or [])}
    fold_filters = set(folds or [])
    selected = []
    for job in jobs:
        dataset_match = not dataset_filters or (
            job['slug'].lower() in dataset_filters
            or job['dataset'].lower() in dataset_filters
        )
        fold_match = not fold_filters or job['fold'] in fold_filters
        if dataset_match and fold_match:
            selected.append(job)
    if not selected:
        raise ValueError('No jobs match the requested dataset/fold filters.')
    return selected


def report_matches_job(report, job, manifest):
    metadata = report.get('metadata', {})
    protocol = metadata.get('protocol', {})
    checkpoint = metadata.get('checkpoint', {})
    return all((
        metadata.get('evaluation_type') == 'chcr_donor_control_pure_inference',
        int(metadata.get('fold', -1)) == job['fold'],
        int(metadata.get('draws', -1)) == int(manifest['draws']),
        int(metadata.get('counterfactual_seed', -1)) == int(
            manifest['counterfactual_seed']
        ),
        protocol.get('config_sha256') == job['config_sha256'],
        Path(checkpoint.get('prefix', '')).resolve() == job['checkpoint'],
    ))


def strategy_metric(report, strategy):
    for row in report.get('strategy_metrics', []):
        if row.get('strategy') == strategy:
            return row
    raise ValueError('Missing strategy metrics: %s' % strategy)


def report_row(job, report, report_path):
    metadata = report['metadata']
    disjoint = strategy_metric(report, 'exact_degree_disjoint')
    degree = strategy_metric(report, 'exact_degree')
    primary_degree = report.get('primary_degree_control', {}).get('metrics')
    if primary_degree is None:
        primary_degree = degree
    return {
        'dataset': job['dataset'],
        'slug': job['slug'],
        'fold': job['fold'],
        'status': 'OK',
        'decision': report['comparison']['decision'],
        'common_coverage': metadata['common_coverage']['fraction'],
        'degree_control_coverage': metadata.get(
            'degree_control_coverage', metadata['common_coverage']
        )['fraction'],
        'factual_AUPR': primary_degree['factual_AUPR'],
        'degree_control_AUPR_drop': primary_degree['AUPR_drop'],
        'degree_control_positive_margin': primary_degree[
            'positive_mean_margin'
        ],
        'degree_control_pair_win_rate': primary_degree[
            'positive_pair_win_rate'
        ],
        'disjoint_AUPR_drop': disjoint['AUPR_drop'],
        'disjoint_positive_margin': disjoint['positive_mean_margin'],
        'disjoint_pair_win_rate': disjoint['positive_pair_win_rate'],
        'degree_control_overlap': degree['assignment_overlap_fraction'],
        'report': str(report_path),
    }


def mean_std(values):
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values)


def summarize_datasets(rows, manifest):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['slug']].append(row)
    summaries = []
    for dataset in manifest['datasets']:
        dataset_rows = grouped.get(dataset['slug'], [])
        supported = sum(
            row['decision'] in SUPPORT_DECISIONS for row in dataset_rows
        )
        drops = [row['degree_control_AUPR_drop'] for row in dataset_rows]
        margins = [
            row['degree_control_positive_margin'] for row in dataset_rows
        ]
        wins = [
            row['degree_control_pair_win_rate'] for row in dataset_rows
        ]
        drop_mean, drop_std = mean_std(drops)
        margin_mean, margin_std = mean_std(margins)
        win_mean, win_std = mean_std(wins)
        complete = len(dataset_rows) == int(manifest['fold_count'])
        passed = bool(
            complete
            and supported >= int(
                manifest['minimum_supported_folds_per_dataset']
            )
            and drop_mean >= float(
                manifest['minimum_dataset_mean_aupr_drop']
            )
            and win_mean >= float(
                manifest['minimum_dataset_mean_pair_win_rate']
            )
        )
        summaries.append({
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'completed_folds': len(dataset_rows),
            'supported_folds': supported,
            'AUPR_drop_mean': drop_mean,
            'AUPR_drop_std': drop_std,
            'positive_margin_mean': margin_mean,
            'positive_margin_std': margin_std,
            'pair_win_rate_mean': win_mean,
            'pair_win_rate_std': win_std,
            'overlap_inconclusive_folds': sum(
                'inconclusive' in row['decision']
                for row in dataset_rows
            ),
            'verdict': 'PASS' if passed else (
                'PENDING' if not complete else 'NO-GO'
            ),
        })
    return summaries


def formatted_mean_std(mean, std):
    if mean is None:
        return '-'
    return '%.6f(+-%.6f)' % (mean, std)


def write_outputs(output_dir, jobs, manifest):
    rows = []
    for job in jobs:
        report_path = output_dir / job['slug'] / (
            'fold_%d' % job['fold']
        ) / 'report.json'
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if not report_matches_job(report, job, manifest):
            continue
        rows.append(report_row(job, report, report_path))
    rows.sort(key=lambda row: (row['slug'], row['fold']))
    summaries = summarize_datasets(rows, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.tsv'
    with results_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RESULT_FIELDS, delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        '# CHCR donor 对照四库冻结审计',
        '',
        '- Manifest: `%s`' % manifest['_path'],
        '- 协议：`%s`' % manifest['protocol'],
        '- Draws / seed: `%d / %d`' % (
            manifest['draws'], manifest['counterfactual_seed']
        ),
        '- 完成 jobs：`%d/%d`' % (len(rows), len(jobs)),
        '',
        '| 数据集 | 完成折 | 支持折 | Degree-control AUPR drop | Positive margin | Pair win rate | Disjoint/overlap 不确定折 | 判定 |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in summaries:
        summary_lines.append(
            '| %s | %d/%d | %d/%d | %s | %s | %s | %d | %s |' % (
                item['dataset'],
                item['completed_folds'], manifest['fold_count'],
                item['supported_folds'], manifest['fold_count'],
                formatted_mean_std(
                    item['AUPR_drop_mean'], item['AUPR_drop_std']
                ),
                formatted_mean_std(
                    item['positive_margin_mean'],
                    item['positive_margin_std'],
                ),
                formatted_mean_std(
                    item['pair_win_rate_mean'], item['pair_win_rate_std']
                ),
                item['overlap_inconclusive_folds'], item['verdict'],
            )
        )
    overall = (
        'PASS' if summaries and all(
            item['verdict'] == 'PASS' for item in summaries
        ) else (
            'PENDING' if any(
                item['verdict'] == 'PENDING' for item in summaries
            ) else 'NO-GO'
        )
    )
    summary_lines.extend([
        '',
        '## 冻结判定',
        '',
        '- 每库支持 fold：`>=%d/%d`' % (
            manifest['minimum_supported_folds_per_dataset'],
            manifest['fold_count'],
        ),
        '- 每库平均 degree-control AUPR drop：`>=%.6f`' % float(
            manifest['minimum_dataset_mean_aupr_drop']
        ),
        '- 每库平均正样本 pair 胜率：`>=%.2f`' % float(
            manifest['minimum_dataset_mean_pair_win_rate']
        ),
        '- 四库总判定：**%s**' % overall,
        '',
        '该批处理只恢复冻结静态 Hctx-P checkpoint 并执行 inner-validation 纯推理；不训练、不读取 outer-test、不修改 donor 或阈值。',
        '',
    ])
    (output_dir / 'summary.md').write_text(
        '\n'.join(summary_lines), encoding='utf-8'
    )
    return rows, summaries, overall


def output_path_for_job(output_dir, job):
    return output_dir / job['slug'] / ('fold_%d' % job['fold'])


def audit_command(args, manifest, job, output_path, dry_run=False):
    command = [
        str(executable_path(args.python_bin)),
        str(AUDIT_SCRIPT),
        '--config', str(job['config']),
        '--checkpoint', str(job['checkpoint']),
        '--fold', str(job['fold']),
        '--draws', str(manifest['draws']),
        '--counterfactual-seed', str(manifest['counterfactual_seed']),
        '--output-dir', str(output_path),
    ]
    if dry_run:
        command.append('--dry-run')
    return command


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def main():
    args = parse_args()
    manifest, all_jobs = load_manifest(args.manifest)
    selected_jobs = select_jobs(all_jobs, args.dataset, args.fold)
    output_dir = repository_path(
        args.output_dir or manifest['output_dir']
    )
    print('CHCR donor-control batch')
    print('  manifest: %s' % manifest['_path'])
    print('  selected jobs: %d/%d' % (len(selected_jobs), len(all_jobs)))
    print('  output: %s' % output_dir)
    for job in selected_jobs:
        print('  - %-18s fold %d %s' % (
            job['dataset'], job['fold'], job['checkpoint']
        ))

    if args.summarize_only:
        _, _, verdict = write_outputs(output_dir, all_jobs, manifest)
        print('Summary rebuilt: %s (%s)' % (
            output_dir / 'summary.md', verdict
        ))
        return 0
    if args.dry_run and not args.protocol_dry_run:
        print('Manifest and checkpoint validation passed.')
        return 0

    failures = 0
    for index, job in enumerate(selected_jobs, start=1):
        job_output = output_path_for_job(output_dir, job)
        report_path = job_output / 'report.json'
        if report_path.is_file() and not args.protocol_dry_run:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report_matches_job(report, job, manifest):
                print('[%d/%d] Reusing %s fold %d' % (
                    index, len(selected_jobs), job['dataset'], job['fold']
                ))
                continue
            if not args.force:
                raise ValueError(
                    'Existing report does not match frozen job: %s. '
                    'Use --force to replace it.' % report_path
                )
        print('[%d/%d] Auditing %s fold %d' % (
            index, len(selected_jobs), job['dataset'], job['fold']
        ))
        command = audit_command(
            args, manifest, job, job_output,
            dry_run=args.protocol_dry_run,
        )
        suffix = 'protocol_dry_run.log' if args.protocol_dry_run else 'audit.log'
        exit_code = run_command(command, job_output / suffix)
        if exit_code != 0:
            failures += 1
            print('Job failed with exit code %d.' % exit_code)
        if not args.protocol_dry_run:
            write_outputs(output_dir, all_jobs, manifest)

    if args.protocol_dry_run:
        if failures:
            print('%d protocol dry-run job(s) failed.' % failures)
            return 1
        print('All selected protocol dry-runs passed.')
        return 0

    _, _, verdict = write_outputs(output_dir, all_jobs, manifest)
    print('Batch summary: %s' % (output_dir / 'summary.md'))
    print('Current verdict: %s' % verdict)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
