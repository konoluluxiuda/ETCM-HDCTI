#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.validate_cold_start_hctx_ablation_configs import (
    DEFAULT_MANIFEST,
    load_rows,
    validate_manifest,
)
from tools.validate_hctx_ablation_configs import repository_path
from tools.summarize_hctx_ablation import parse_fold_metrics


METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')


def metric_mean(value):
    match = re.match(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))', value or '')
    if not match:
        raise ValueError('Cannot parse metric mean from %r.' % value)
    return float(match.group(1))


def successful_row(rows, dataset, variant):
    matching = [
        row for row in rows
        if row.get('dataset') == dataset
        and row.get('variant') == variant
        and row.get('status') == 'OK'
    ]
    if len(matching) != 1:
        raise ValueError(
            'Expected one successful %s row for %s.' % (variant, dataset)
        )
    return matching[0]


def write_tsv(path, rows):
    fields = [
        'dataset',
        'no_context_AUC', 'hctx_AUC', 'sdis_AUC',
        'no_context_AUPR', 'hctx_AUPR', 'sdis_AUPR',
        'hctx_minus_no_context_AUPR', 'sdis_minus_hctx_AUPR',
        'hctx_positive_folds',
        'no_context_Recall', 'hctx_Recall', 'sdis_Recall',
        'no_context_Precision', 'hctx_Precision', 'sdis_Precision',
        'no_context_F1-score', 'hctx_F1-score', 'sdis_F1-score',
    ]
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest_path, no_context_results, output_dir):
    manifest, _ = validate_manifest(manifest_path)
    no_context_results = repository_path(no_context_results)
    reference_results = repository_path(manifest['reference_results'])
    no_rows = load_rows(no_context_results)
    reference_rows = load_rows(reference_results)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in manifest['datasets']:
        no_row = successful_row(no_rows, dataset['name'], 'NoContext')
        hctx_row = successful_row(
            reference_rows, dataset['name'], 'HerbOnly'
        )
        sdis_row = successful_row(reference_rows, dataset['name'], 'SDIS')
        if no_row.get('config_sha256') != dataset['no_context_sha256']:
            raise ValueError(
                'NoContext result hash mismatch for %s.' % dataset['name']
            )
        if repository_path(no_row.get('config', '')) != repository_path(
                dataset['no_context_config']):
            raise ValueError(
                'NoContext result path mismatch for %s.' % dataset['name']
            )
        row = {'dataset': dataset['name']}
        for metric in METRICS:
            row['no_context_' + metric] = metric_mean(no_row[metric])
            row['hctx_' + metric] = metric_mean(hctx_row[metric])
            row['sdis_' + metric] = metric_mean(sdis_row[metric])
        row['hctx_minus_no_context_AUPR'] = (
            row['hctx_AUPR'] - row['no_context_AUPR']
        )
        row['sdis_minus_hctx_AUPR'] = (
            row['sdis_AUPR'] - row['hctx_AUPR']
        )
        no_folds = parse_fold_metrics(repository_path(no_row['log']))
        hctx_folds = parse_fold_metrics(repository_path(hctx_row['log']))
        if sorted(no_folds) != list(range(1, 6)):
            raise ValueError(
                'NoContext log must contain folds 1..5 for %s.'
                % dataset['name']
            )
        if sorted(hctx_folds) != list(range(1, 6)):
            raise ValueError(
                'Hctx-P log must contain folds 1..5 for %s.'
                % dataset['name']
            )
        row['hctx_positive_folds'] = sum(
            hctx_folds[fold]['AUPR'] > no_folds[fold]['AUPR']
            for fold in range(1, 6)
        )
        rows.append(row)

    write_tsv(output_dir / 'results.tsv', rows)
    lines = [
        '# Compound cold-start 统一递进消融',
        '',
        '- NoContext results: `%s`' % no_context_results,
        '- Frozen Hctx-P/SDIS results: `%s`' % reference_results,
        '- Protocol: Strict compound cold-start, five folds, early stopping, '
        'Dot, `attention.max.nodes=0`',
        '',
        '| 数据集 | NoContext AUPR | Hctx-P AUPR | Hctx-P - NoContext | '
        '正向 folds | Hctx-P + SDIS AUPR | SDIS - Hctx-P |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %d/5 | %.6f | %+.6f |' % (
                row['dataset'],
                row['no_context_AUPR'],
                row['hctx_AUPR'],
                row['hctx_minus_no_context_AUPR'],
                row['hctx_positive_folds'],
                row['sdis_AUPR'],
                row['sdis_minus_hctx_AUPR'],
            )
        )
    macro_hctx = sum(
        row['hctx_minus_no_context_AUPR'] for row in rows
    ) / len(rows)
    macro_sdis = sum(
        row['sdis_minus_hctx_AUPR'] for row in rows
    ) / len(rows)
    lines.extend([
        '',
        '- Macro `Hctx-P - NoContext` AUPR: `%+.6f`' % macro_hctx,
        '- Macro `SDIS - Hctx-P` AUPR: `%+.6f`' % macro_sdis,
        '',
        '该结果用于描述同一 cold-start 协议中的机制递进，不设置新的模型选择'
        '门槛，也不据此修改 split、seed、epoch、attention 或模块参数。',
        '',
    ])
    (output_dir / 'summary.md').write_text(
        '\n'.join(lines), encoding='utf-8'
    )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Summarize NoContext, Hctx-P, and SDIS cold-start results.'
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--no-context-results', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    summarize(args.manifest, args.no_context_results, args.output_dir)
    print('Summary: %s' % (
        Path(args.output_dir).resolve() / 'summary.md'
    ))


if __name__ == '__main__':
    main()
