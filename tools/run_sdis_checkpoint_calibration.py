#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_TOOL = REPOSITORY_ROOT / 'tools' / 'calibrate_checkpoint_folds.py'
DEFAULT_SOURCE_DIR = (
    REPOSITORY_ROOT / 'results' / 'batch_runs' / 'sdis_full_20260718_212240'
)
METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')


@dataclass(frozen=True)
class CalibrationJob:
    dataset: str
    variant: str
    slug: str
    config: str
    log_name: str


JOBS = (
    CalibrationJob(
        'TCM-Suite', 'HerbOnly', 'tcmsuite_baseline',
        'configs/HDCTI_tcmsuite_cold_start_no_dense_herb_only_full.conf',
        '01_tcmsuite_baseline.log',
    ),
    CalibrationJob(
        'TCM-Suite', 'SDIS', 'tcmsuite_sdis',
        'configs/HDCTI_tcmsuite_cold_start_sdis_full.conf',
        '02_tcmsuite_sdis.log',
    ),
    CalibrationJob(
        'TCMSP', 'HerbOnly', 'tcmsp_baseline',
        'configs/HDCTI_tcmsp_cold_start_no_dense_herb_only_full.conf',
        '03_tcmsp_baseline.log',
    ),
    CalibrationJob(
        'TCMSP', 'SDIS', 'tcmsp_sdis',
        'configs/HDCTI_tcmsp_cold_start_sdis_full.conf',
        '04_tcmsp_sdis.log',
    ),
    CalibrationJob(
        'SymMap2.0', 'HerbOnly', 'symmap_baseline',
        'configs/HDCTI_symmap_cold_start_no_dense_herb_only_full.conf',
        '05_symmap_baseline.log',
    ),
    CalibrationJob(
        'SymMap2.0', 'SDIS', 'symmap_sdis',
        'configs/HDCTI_symmap_cold_start_sdis_full.conf',
        '06_symmap_sdis.log',
    ),
    CalibrationJob(
        'ETCM2.0 mention10', 'HerbOnly', 'etcm_baseline',
        'configs/HDCTI_etcm_mention10_cold_start_no_dense_herb_only_full.conf',
        '07_etcm_baseline.log',
    ),
    CalibrationJob(
        'ETCM2.0 mention10', 'SDIS', 'etcm_sdis',
        'configs/HDCTI_etcm_mention10_cold_start_sdis_full.conf',
        '08_etcm_sdis.log',
    ),
)


CHECKPOINT_PATTERN = re.compile(
    r'^\s*模型权重保存成功:\s*(.+?hdcti_model\.ckpt)\s*$',
    flags=re.MULTILINE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Calibrate all HerbOnly/SDIS five-fold checkpoints without training. '
            'Each threshold is selected on the corresponding inner-validation set.'
        )
    )
    parser.add_argument(
        '--source-dir',
        default=str(DEFAULT_SOURCE_DIR),
        help='Completed run_sdis_full_batch.sh result directory.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate logs, configs, and checkpoint shards without TensorFlow.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-evaluate jobs that already contain report.json.',
    )
    return parser.parse_args()


def extract_checkpoint_prefixes(log_path, repository_root=REPOSITORY_ROOT):
    log_path = Path(log_path)
    matches = CHECKPOINT_PATTERN.findall(
        log_path.read_text(encoding='utf-8', errors='replace')
    )
    if len(matches) != 5:
        raise ValueError(
            'Expected 5 saved checkpoints in %s; found %d.' %
            (log_path, len(matches))
        )
    checkpoints = []
    for value in matches:
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = Path(repository_root) / path
        prefix = path.resolve()
        index_path = Path(str(prefix) + '.index')
        data_files = sorted(prefix.parent.glob(prefix.name + '.data-*'))
        if not index_path.is_file():
            raise FileNotFoundError('Checkpoint index not found: %s' % index_path)
        if not data_files:
            raise FileNotFoundError('Checkpoint data shard not found: %s' % prefix)
        checkpoints.append(prefix)
    return checkpoints


def load_report(report_path):
    payload = json.loads(Path(report_path).read_text(encoding='utf-8'))
    if len(payload.get('fold_results', [])) != 5:
        raise ValueError('Calibration report must contain exactly five folds: %s' % report_path)
    for metric_name in ('AUC', 'AUPR'):
        fixed = payload['fixed_summary'][metric_name]['mean']
        calibrated = payload['calibrated_summary'][metric_name]['mean']
        if not np.isclose(fixed, calibrated, rtol=0.0, atol=1e-12):
            raise ValueError(
                '%s changed after threshold calibration in %s.' %
                (metric_name, report_path)
            )
    return payload


def parse_logged_metric(value):
    return float(str(value).split('(', 1)[0].strip())


def load_source_metrics(results_path):
    expected = {}
    with Path(results_path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row.get('status') != 'OK':
                continue
            expected[(row['dataset'], row['variant'])] = {
                'AUC': parse_logged_metric(row['AUC']),
                'AUPR': parse_logged_metric(row['AUPR']),
            }
    return expected


def validate_report_against_training(job, payload, expected_metrics):
    expected = expected_metrics.get((job.dataset, job.variant))
    if expected is None:
        raise ValueError(
            'Missing successful training summary for %s %s.' %
            (job.dataset, job.variant)
        )
    mismatches = {}
    for metric_name in ('AUC', 'AUPR'):
        restored = payload['fixed_summary'][metric_name]['mean']
        if not np.isclose(restored, expected[metric_name], rtol=0.0, atol=5e-6):
            mismatches[metric_name] = {
                'training_log': expected[metric_name],
                'restored': restored,
            }
    if mismatches:
        raise ValueError(
            'Restored checkpoint metrics do not match the training log for '
            '%s %s: %s' % (job.dataset, job.variant, mismatches)
        )


def report_row(job, report_path):
    payload = load_report(report_path)
    thresholds = np.asarray(
        [row['threshold'] for row in payload['fold_results']], dtype=np.float64
    )
    row = {
        'dataset': job.dataset,
        'variant': job.variant,
        'config': job.config,
        'report': str(report_path),
        'threshold_mean': float(np.mean(thresholds)),
        'threshold_std': float(np.std(thresholds, ddof=1)),
    }
    for result_name in ('fixed', 'calibrated'):
        summary = payload[result_name + '_summary']
        for metric_name in METRICS:
            key = metric_name.lower().replace('-', '_')
            row[result_name + '_' + key] = summary[metric_name]['mean']
            row[result_name + '_' + key + '_std'] = summary[metric_name]['std']
    return row


def metric_text(row, prefix, metric_name):
    key = metric_name.lower().replace('-', '_')
    return '%.6f(+-%.6f)' % (
        row[prefix + '_' + key], row[prefix + '_' + key + '_std']
    )


def paired_deltas(rows):
    indexed = {(row['dataset'], row['variant']): row for row in rows}
    output = []
    for dataset in dict.fromkeys(row['dataset'] for row in rows):
        baseline = indexed.get((dataset, 'HerbOnly'))
        candidate = indexed.get((dataset, 'SDIS'))
        if baseline is None or candidate is None:
            continue
        output.append({
            'dataset': dataset,
            'auc': candidate['fixed_auc'] - baseline['fixed_auc'],
            'aupr': candidate['fixed_aupr'] - baseline['fixed_aupr'],
            'recall': (
                candidate['calibrated_recall'] - baseline['calibrated_recall']
            ),
            'precision': (
                candidate['calibrated_precision'] - baseline['calibrated_precision']
            ),
            'f1_score': (
                candidate['calibrated_f1_score'] - baseline['calibrated_f1_score']
            ),
        })
    return output


def build_markdown(rows, source_dir):
    lines = [
        '# SDIS 五折 checkpoint 纯推理阈值校准',
        '',
        '- 来源批次：`%s`' % source_dir,
        '- 阈值选择：每折仅使用对应 inner-validation，以 F1 最大化选择阈值。',
        '- 外层测试集：只用于一次评价，不参与阈值选择。',
        '- 训练与优化器更新：`0`。',
        '- AUC/AUPR 与阈值无关，并已检查校准前后完全一致。',
        '',
        '## 模型结果',
        '',
        '| 数据集 | 模型 | AUC | AUPR | F1@0.5 | 阈值 | 校准 Recall | 校准 Precision | 校准 F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            '| %s | %s | %s | %s | %s | %.6f(+-%.6f) | %s | %s | %s |' % (
                row['dataset'], row['variant'],
                metric_text(row, 'fixed', 'AUC'),
                metric_text(row, 'fixed', 'AUPR'),
                metric_text(row, 'fixed', 'F1-score'),
                row['threshold_mean'], row['threshold_std'],
                metric_text(row, 'calibrated', 'Recall'),
                metric_text(row, 'calibrated', 'Precision'),
                metric_text(row, 'calibrated', 'F1-score'),
            )
        )

    deltas = paired_deltas(rows)
    lines.extend([
        '',
        '## SDIS 相对 HerbOnly 的配对均值差',
        '',
        '| 数据集 | AUC | AUPR | 校准 Recall | 校准 Precision | 校准 F1 |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in deltas:
        lines.append(
            '| %s | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f |' % (
                row['dataset'], row['auc'], row['aupr'], row['recall'],
                row['precision'], row['f1_score'],
            )
        )
    if deltas:
        lines.append(
            '| **Macro** | **%+.6f** | **%+.6f** | **%+.6f** | '
            '**%+.6f** | **%+.6f** |' % tuple(
                float(np.mean([row[key] for row in deltas]))
                for key in ('auc', 'aupr', 'recall', 'precision', 'f1_score')
            )
        )
    lines.extend([
        '',
        '## 解释边界',
        '',
        '该校准是对分数尺度的描述性审计，不是新的模型选择门槛。SDIS 是否成立仍由预注册的 outer-test AUPR 条件决定；校准结果只用于判断固定 `0.5` 阈值下的 Recall/F1 下降有多少来自分数尺度变化。',
        '',
        '逐折原始结果见各子目录的 `report.md` 与 `report.json`，统一机器可读结果见 `results.tsv`。',
        '',
    ])
    return '\n'.join(lines)


def write_combined_outputs(calibration_dir, source_dir):
    rows = []
    for job in JOBS:
        report_path = Path(calibration_dir) / job.slug / 'report.json'
        if report_path.is_file():
            rows.append(report_row(job, report_path))

    fieldnames = [
        'dataset', 'variant', 'config', 'report', 'threshold_mean', 'threshold_std',
    ]
    for prefix in ('fixed', 'calibrated'):
        for metric_name in METRICS:
            key = metric_name.lower().replace('-', '_')
            fieldnames.extend((prefix + '_' + key, prefix + '_' + key + '_std'))
    results_path = Path(calibration_dir) / 'results.tsv'
    with results_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    summary_path = Path(calibration_dir) / 'summary.md'
    summary_path.write_text(
        build_markdown(rows, source_dir), encoding='utf-8'
    )
    return rows, summary_path


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError('Source batch directory not found: %s' % source_dir)
    calibration_dir = source_dir / 'calibration'
    calibration_dir.mkdir(parents=True, exist_ok=True)
    source_results_path = source_dir / 'results.tsv'
    if not source_results_path.is_file():
        raise FileNotFoundError(
            'Source batch results not found: %s' % source_results_path
        )
    expected_metrics = load_source_metrics(source_results_path)

    prepared = []
    for job in JOBS:
        config_path = REPOSITORY_ROOT / job.config
        log_path = source_dir / job.log_name
        if not config_path.is_file():
            raise FileNotFoundError('Config not found: %s' % config_path)
        if not log_path.is_file():
            raise FileNotFoundError('Training log not found: %s' % log_path)
        checkpoints = extract_checkpoint_prefixes(log_path)
        prepared.append((job, config_path, checkpoints))

    print('SDIS checkpoint-only threshold calibration')
    print('  source: %s' % source_dir)
    print('  jobs: %d; checkpoints: %d' % (
        len(prepared), sum(len(item[2]) for item in prepared)
    ))
    print('  optimizer/training steps: disabled')
    for index, (job, config_path, checkpoints) in enumerate(prepared, start=1):
        print(
            '  [%d/%d] %-18s %-8s %s ... %s' % (
                index, len(prepared), job.dataset, job.variant,
                checkpoints[0].parent.name, checkpoints[-1].parent.name,
            )
        )
        if args.dry_run:
            continue
        output_dir = calibration_dir / job.slug
        report_path = output_dir / 'report.json'
        if report_path.is_file() and not args.force:
            payload = load_report(report_path)
            try:
                validate_report_against_training(job, payload, expected_metrics)
            except ValueError as error:
                print('    stale report rejected: %s' % error)
            else:
                print('    reusing: %s' % report_path)
                continue
        command = [
            sys.executable,
            str(CALIBRATION_TOOL),
            '--config',
            str(config_path),
        ]
        for checkpoint in checkpoints:
            command.extend(('--checkpoint', str(checkpoint)))
        command.extend(('--output-dir', str(output_dir)))
        print('    command: %s' % shlex.join(command))
        completed = subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                'Calibration failed for %s %s with exit code %d.' %
                (job.dataset, job.variant, completed.returncode)
            )
        payload = load_report(report_path)
        validate_report_against_training(job, payload, expected_metrics)

    if args.dry_run:
        print('Dry run passed: all 40 checkpoint prefixes and data shards exist.')
        return 0

    rows, summary_path = write_combined_outputs(calibration_dir, source_dir)
    if len(rows) != len(JOBS):
        raise RuntimeError(
            'Expected %d calibration reports; summarized %d.' %
            (len(JOBS), len(rows))
        )
    print('\nCombined calibration summary: %s' % summary_path)
    print('Completed at: %s' % datetime.now().astimezone().isoformat())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
