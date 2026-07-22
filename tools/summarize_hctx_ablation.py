#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.validate_hctx_ablation_configs import (
    DEFAULT_MANIFEST,
    repository_path,
    sha256_file,
    validate_manifest,
)


METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')
PAIR_FIELDS = (
    'dataset', 'no_context_AUC', 'hctx_AUC', 'AUC_delta',
    'no_context_AUPR', 'hctx_AUPR', 'AUPR_delta',
    'no_context_Recall', 'hctx_Recall', 'Recall_delta',
    'no_context_Precision', 'hctx_Precision', 'Precision_delta',
    'no_context_F1-score', 'hctx_F1-score', 'F1-score_delta',
    'AUPR_positive_folds', 'folds',
)
FOLD_FIELDS = (
    'dataset', 'fold', 'no_context_AUPR', 'hctx_AUPR', 'AUPR_delta',
)
METRIC_PATTERN = re.compile(
    r'^(AUC|AUPR|Recall|Precision|F1-score):\s*'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
)
FOLD_PATTERN = re.compile(r'^Predicting \[(\d+)\]')


def metric_mean(value):
    match = re.match(
        r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
        value or '',
    )
    if not match:
        raise ValueError('Cannot parse metric mean from %r.' % value)
    return float(match.group(1))


def parse_fold_metrics(log_path):
    folds = {}
    current_fold = None
    in_summary = False
    for raw_line in Path(log_path).read_text(
            encoding='utf-8', errors='replace').splitlines():
        line = raw_line.strip()
        fold_match = FOLD_PATTERN.match(line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            in_summary = False
            folds.setdefault(current_fold, {})
            continue
        if line == 'The result of 5-fold cross validation:':
            current_fold = None
            in_summary = True
            continue
        if current_fold is None or in_summary:
            continue
        metric_match = METRIC_PATTERN.match(line)
        if metric_match:
            folds[current_fold][metric_match.group(1)] = float(
                metric_match.group(2)
            )
    return folds


def read_results(path):
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def result_row(rows, dataset, variant):
    matching = [
        row for row in rows
        if row.get('dataset') == dataset
        and row.get('variant') == variant
        and row.get('status') == 'OK'
    ]
    if len(matching) != 1:
        raise ValueError(
            'Expected one successful %s row for %s, found %d.'
            % (variant, dataset, len(matching))
        )
    return matching[0]


def validate_result_config(row, expected_path, expected_hash):
    row_path = repository_path(row['config'])
    if row_path != repository_path(expected_path):
        raise ValueError('Result config path does not match frozen manifest.')
    if row.get('config_sha256') != expected_hash:
        raise ValueError('Result config hash does not match frozen manifest.')
    if sha256_file(row_path) != expected_hash:
        raise ValueError('Current config hash does not match frozen manifest.')


def decision(pair_rows, manifest):
    gate = manifest['decision_gate']
    deltas = [float(row['AUPR_delta']) for row in pair_rows]
    non_decreasing = sum(value >= 0.0 for value in deltas)
    macro_delta = sum(deltas) / len(deltas)
    minimum_delta = min(deltas)
    fold_supported = sum(
        int(row['AUPR_positive_folds']) >= 3 for row in pair_rows
    )
    criteria = {
        'non_decreasing_datasets': (
            non_decreasing >= gate['minimum_non_decreasing_datasets']
        ),
        'macro_AUPR_delta': (
            macro_delta >= gate['minimum_macro_AUPR_delta']
        ),
        'maximum_single_dataset_drop': (
            minimum_delta >= -gate['maximum_single_dataset_AUPR_drop']
        ),
        'fold_direction_datasets': (
            fold_supported >= gate['minimum_datasets_with_three_positive_folds']
        ),
    }
    return {
        'verdict': 'PASS' if all(criteria.values()) else 'NO-GO',
        'criteria': criteria,
        'non_decreasing_datasets': non_decreasing,
        'macro_AUPR_delta': macro_delta,
        'minimum_dataset_AUPR_delta': minimum_delta,
        'datasets_with_three_positive_folds': fold_supported,
    }


def build_summary(pair_rows, verdict, no_results_path, hctx_results_path):
    lines = [
        '# 统一无稠密注意力 Hctx-P 四库直接消融',
        '',
        '- NoContext results: `%s`' % Path(no_results_path).resolve(),
        '- Frozen Hctx-P results: `%s`' % Path(hctx_results_path).resolve(),
        '- Protocol: Strict pair-stratified, five folds, early stopping, Dot, '
        '`attention.max.nodes=0`',
        '',
        '| 数据集 | NoContext AUPR | Hctx-P AUPR | Delta | Positive folds |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in pair_rows:
        lines.append('| %s | %.6f | %.6f | %+.6f | %d/%d |' % (
            row['dataset'], row['no_context_AUPR'], row['hctx_AUPR'],
            row['AUPR_delta'], row['AUPR_positive_folds'], row['folds'],
        ))
    lines.extend([
        '',
        '## 冻结判定',
        '',
        '- 非下降数据库：`%d/4`' % verdict['non_decreasing_datasets'],
        '- Macro AUPR delta：`%+.6f`' % verdict['macro_AUPR_delta'],
        '- 最小单库 AUPR delta：`%+.6f`' % (
            verdict['minimum_dataset_AUPR_delta']
        ),
        '- 至少 3/5 folds 提高的数据库：`%d/4`' % (
            verdict['datasets_with_three_positive_folds']
        ),
        '- Verdict: **%s**' % verdict['verdict'],
        '',
        '该判定只比较已冻结的匹配配置，不用于重新搜索 Hctx-P 结构、'
        'attention、epoch、seed 或数据集特定参数。',
        '',
    ])
    return '\n'.join(lines)


def summarize(manifest_path, no_context_results, output_dir):
    manifest, _ = validate_manifest(manifest_path)
    no_context_results = repository_path(no_context_results)
    hctx_results = repository_path(manifest['reference_results'])
    output_dir = Path(output_dir).resolve()
    no_rows = read_results(no_context_results)
    hctx_rows = read_results(hctx_results)
    pair_rows = []
    fold_rows = []
    for dataset in manifest['datasets']:
        no_row = result_row(no_rows, dataset['name'], 'NoContext')
        hctx_row = result_row(hctx_rows, dataset['name'], 'Hctx-P')
        validate_result_config(
            no_row, dataset['no_context_config'], dataset['no_context_sha256']
        )
        validate_result_config(
            hctx_row, dataset['hctx_config'], dataset['hctx_sha256']
        )
        no_folds = parse_fold_metrics(repository_path(no_row['log']))
        hctx_folds = parse_fold_metrics(repository_path(hctx_row['log']))
        if sorted(no_folds) != list(range(1, 6)):
            raise ValueError('NoContext log must contain folds 1..5.')
        if sorted(hctx_folds) != list(range(1, 6)):
            raise ValueError('Hctx-P log must contain folds 1..5.')
        positive_folds = 0
        for fold in range(1, 6):
            no_aupr = no_folds[fold].get('AUPR')
            hctx_aupr = hctx_folds[fold].get('AUPR')
            if no_aupr is None or hctx_aupr is None:
                raise ValueError('Missing fold AUPR for %s fold %d.' % (
                    dataset['name'], fold
                ))
            delta = hctx_aupr - no_aupr
            positive_folds += delta > 0.0
            fold_rows.append({
                'dataset': dataset['name'],
                'fold': fold,
                'no_context_AUPR': no_aupr,
                'hctx_AUPR': hctx_aupr,
                'AUPR_delta': delta,
            })
        pair = {
            'dataset': dataset['name'],
            'AUPR_positive_folds': positive_folds,
            'folds': 5,
        }
        for metric in METRICS:
            no_value = metric_mean(no_row[metric])
            hctx_value = metric_mean(hctx_row[metric])
            pair['no_context_' + metric] = no_value
            pair['hctx_' + metric] = hctx_value
            pair[metric + '_delta'] = hctx_value - no_value
        pair_rows.append(pair)
    verdict = decision(pair_rows, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / 'paired_results.tsv', pair_rows, PAIR_FIELDS)
    write_tsv(output_dir / 'paired_folds.tsv', fold_rows, FOLD_FIELDS)
    (output_dir / 'decision.json').write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'summary.md').write_text(
        build_summary(pair_rows, verdict, no_context_results, hctx_results),
        encoding='utf-8',
    )
    return verdict


def main():
    parser = argparse.ArgumentParser(
        description='Pair completed NoContext results with frozen Hctx-P results.'
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--no-context-results', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    verdict = summarize(
        args.manifest, args.no_context_results, args.output_dir
    )
    print('Hctx-P ablation verdict: %s' % verdict['verdict'])
    print('Summary: %s' % (Path(args.output_dir).resolve() / 'summary.md'))


if __name__ == '__main__':
    main()
