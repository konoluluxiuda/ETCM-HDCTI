#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPOSITORY_ROOT
    / 'results'
    / 'chcr_donor_controls'
    / 'four_dataset_static_hctxp'
)
DATASET_ORDER = ('tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10')
DATASET_NAMES = {
    'tcmsuite': 'TCM-Suite',
    'tcmsp': 'TCMSP',
    'symmap': 'SymMap2.0',
    'etcm_mention10': 'ETCM2.0 mention10',
}
SUBGROUP_ORDER = ('H-C degree', 'training C-P degree')
GROUP_ORDER = {
    'H-C degree': ('1', '2-3', '4-10', '>10'),
    'training C-P degree': ('0', '1-2', '3-5', '6-10', '>10'),
}
FOLD_FIELDS = (
    'dataset', 'slug', 'fold', 'subgroup', 'group', 'pairs', 'compounds',
    'mean_margin', 'median_margin', 'margin_std', 'pair_win_rate',
    'compound_win_rate', 'compound_mean_margin', 'analyzable', 'report',
)
SUMMARY_FIELDS = (
    'dataset', 'slug', 'subgroup', 'group', 'folds', 'analyzable_folds',
    'positive_margin_folds', 'positive_fold_fraction', 'pairs_across_folds',
    'fold_margin_mean', 'fold_margin_std', 'pair_weighted_margin',
    'fold_pair_win_mean', 'pair_weighted_pair_win',
    'frozen_direction_consistency', 'pair_win_reference_met',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Summarize primary exact-degree CHCR donor-control strata from '
            'frozen inference reports without retraining.'
        )
    )
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--output-dir')
    parser.add_argument('--dataset', action='append', default=[])
    parser.add_argument('--minimum-positive-fold-fraction', type=float, default=0.75)
    parser.add_argument('--reference-pair-win-rate', type=float, default=0.60)
    return parser.parse_args()


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def sort_key(row):
    slug = row['slug']
    subgroup = row['subgroup']
    group = row['group']
    dataset_rank = (
        DATASET_ORDER.index(slug)
        if slug in DATASET_ORDER else len(DATASET_ORDER)
    )
    subgroup_rank = (
        SUBGROUP_ORDER.index(subgroup)
        if subgroup in SUBGROUP_ORDER else len(SUBGROUP_ORDER)
    )
    groups = GROUP_ORDER.get(subgroup, ())
    group_rank = groups.index(group) if group in groups else len(groups)
    return dataset_rank, subgroup_rank, group_rank, int(row.get('fold', 0))


def report_rows(report_path, slug):
    payload = json.loads(report_path.read_text(encoding='utf-8'))
    metadata = payload['metadata']
    analysis = payload['primary_degree_control']['analysis']
    dataset = metadata.get('dataset_name') or DATASET_NAMES.get(slug, slug)
    fold = int(metadata['fold'])
    rows = []
    for subgroup, groups in analysis['subgroups'].items():
        for group, metrics in groups.items():
            row = {
                'dataset': dataset,
                'slug': slug,
                'fold': fold,
                'subgroup': subgroup,
                'group': group,
                'pairs': int(metrics['pairs']),
                'compounds': int(metrics['compounds']),
                'mean_margin': metrics['mean_margin'],
                'median_margin': metrics['median_margin'],
                'margin_std': metrics['margin_std'],
                'pair_win_rate': metrics['pair_win_rate'],
                'compound_win_rate': metrics['compound_win_rate'],
                'compound_mean_margin': metrics['compound_mean_margin'],
                'analyzable': bool(metrics['analyzable']),
                'report': str(report_path.resolve()),
            }
            rows.append(row)
    return rows


def discover_rows(input_dir, selected_slugs=None):
    input_dir = Path(input_dir).resolve()
    selected = set(selected_slugs or [])
    rows = []
    for dataset_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        slug = dataset_dir.name
        if selected and slug not in selected:
            continue
        reports = sorted(dataset_dir.glob('fold_*/report.json'))
        if not reports:
            continue
        for report_path in reports:
            rows.extend(report_rows(report_path, slug))
    if not rows:
        raise FileNotFoundError('No frozen report.json files found in %s.' % input_dir)
    return sorted(rows, key=sort_key)


def weighted_mean(rows, field):
    weighted = [
        (float(row[field]), int(row['pairs']))
        for row in rows
        if row[field] is not None and int(row['pairs']) > 0
    ]
    total = sum(weight for _, weight in weighted)
    if total == 0:
        return None
    return sum(value * weight for value, weight in weighted) / total


def aggregate_rows(
        rows,
        minimum_positive_fold_fraction=0.75,
        reference_pair_win_rate=0.60):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['slug'], row['subgroup'], row['group'])].append(row)

    summaries = []
    for (dataset, slug, subgroup, group), fold_rows in grouped.items():
        analyzable = [row for row in fold_rows if row['analyzable']]
        margins = [float(row['mean_margin']) for row in analyzable]
        wins = [float(row['pair_win_rate']) for row in analyzable]
        positive_folds = sum(value > 0.0 for value in margins)
        positive_fraction = (
            positive_folds / len(analyzable) if analyzable else None
        )
        weighted_margin = weighted_mean(analyzable, 'mean_margin')
        weighted_win = weighted_mean(analyzable, 'pair_win_rate')
        direction_consistent = bool(
            analyzable
            and positive_fraction >= minimum_positive_fold_fraction
            and weighted_margin > 0.0
        )
        summaries.append({
            'dataset': dataset,
            'slug': slug,
            'subgroup': subgroup,
            'group': group,
            'folds': len(fold_rows),
            'analyzable_folds': len(analyzable),
            'positive_margin_folds': positive_folds,
            'positive_fold_fraction': positive_fraction,
            'pairs_across_folds': sum(int(row['pairs']) for row in analyzable),
            'fold_margin_mean': statistics.fmean(margins) if margins else None,
            'fold_margin_std': statistics.pstdev(margins) if margins else None,
            'pair_weighted_margin': weighted_margin,
            'fold_pair_win_mean': statistics.fmean(wins) if wins else None,
            'pair_weighted_pair_win': weighted_win,
            'frozen_direction_consistency': direction_consistent,
            'pair_win_reference_met': bool(
                weighted_win is not None
                and weighted_win >= reference_pair_win_rate
            ),
        })
    return sorted(summaries, key=sort_key)


def fmt(value, digits=4):
    if value is None:
        return '-'
    return ('%%.%df' % digits) % float(value)


def markdown_table(lines, summaries, subgroup):
    lines.extend([
        '## %s' % subgroup,
        '',
        '| Dataset | Stratum | Analyzable folds | Positive folds | '
        'Fold margin | Weighted margin | Weighted win | Direction consistency |',
        '|---|---|---:|---:|---:|---:|---:|---|',
    ])
    for row in summaries:
        if row['subgroup'] != subgroup:
            continue
        margin = (
            '%s (+-%s)' % (
                fmt(row['fold_margin_mean']), fmt(row['fold_margin_std'])
            )
            if row['fold_margin_mean'] is not None else '-'
        )
        lines.append(
            '| %s | %s | %d/%d | %d/%d | %s | %s | %s | %s |' % (
                row['dataset'], row['group'], row['analyzable_folds'],
                row['folds'], row['positive_margin_folds'],
                row['analyzable_folds'], margin,
                fmt(row['pair_weighted_margin']),
                fmt(row['pair_weighted_pair_win']),
                'PASS' if row['frozen_direction_consistency'] else 'NO-GO',
            )
        )
    lines.append('')


def build_markdown(input_dir, summaries, thresholds):
    lines = [
        '# CHCR Degree-Stratum Failure Analysis',
        '',
        '- Source: `%s`' % Path(input_dir).resolve(),
        '- Input: frozen static Hctx-P primary `exact_degree` inference reports',
        '- Positive-fold threshold: `>= %.2f`' % thresholds['positive_fraction'],
        '- Descriptive pair-win reference: `>= %.2f`' % thresholds['pair_win_rate'],
        '- No retraining, no threshold tuning, no outer-test access.',
        '',
        '折级 margin 用均值和总体标准差描述；weighted 指标按各折正样本 pair 数加权。'
        '不同 inner-validation folds 可能包含重复 records，因此不把 pooled pair 当作独立样本，'
        '也不在此报告 p 值。',
        '',
    ]
    for subgroup in SUBGROUP_ORDER:
        markdown_table(lines, summaries, subgroup)

    symmap = [row for row in summaries if row['slug'] == 'symmap']
    failed = [row for row in symmap if not row['frozen_direction_consistency']]
    passed = [row for row in symmap if row['frozen_direction_consistency']]
    weak_wins = [row for row in symmap if not row['pair_win_reference_met']]
    lines.extend([
        '## SymMap2.0 失败模式',
        '',
    ])
    if not symmap:
        lines.append('未找到 SymMap2.0 冻结结果。')
    else:
        lines.append('通过冻结分层条件的区间：%s。' % (
            '、'.join('%s=%s' % (row['subgroup'], row['group']) for row in passed)
            if passed else '无'
        ))
        lines.append('未通过的区间：%s。' % (
            '、'.join('%s=%s' % (row['subgroup'], row['group']) for row in failed)
            if failed else '无'
        ))
        lines.append('加权 pair 胜率低于 0.60 的区间：%s。' % (
            '、'.join(
                '%s=%s' % (row['subgroup'], row['group'])
                for row in weak_wins
            ) if weak_wins else '无'
        ))
        lines.extend([
            '',
            '因此 SymMap 的总体正 margin 不能解释为所有支持度区间上的稳定药材语义。'
            '论文应将其报告为支持度异质性，而不是通过降低 75% 分层一致性门槛改写为通过。',
            '',
            '该现象并非 SymMap 独有：TCM-Suite 的 training C-P degree 1-2 和 3-5 '
            '区间也未达到方向一致性。区别在于 TCM-Suite 的全部 H-C degree 区间稳定为正，'
            '而 SymMap 同时在 H-C degree=1 与低训练 C-P 支持区间失效。',
        ])
    lines.extend([
        '',
        '## 研究决策',
        '',
        '1. 保留 TCM-Suite、TCMSP 和 ETCM2.0 的 degree-control 机制证据。',
        '2. SymMap 只支持总体趋势，不作为 CHCR 普适机制证据。',
        '3. 不继续搜索 donor、seed 或数据集特定阈值。',
        '4. 后续方法表述改为 support-aware context regularization，并明确适用边界。',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / 'degree_strata_analysis'
    )
    rows = discover_rows(input_dir, args.dataset)
    summaries = aggregate_rows(
        rows,
        minimum_positive_fold_fraction=args.minimum_positive_fold_fraction,
        reference_pair_win_rate=args.reference_pair_win_rate,
    )
    write_tsv(output_dir / 'by_fold.tsv', rows, FOLD_FIELDS)
    write_tsv(output_dir / 'summary.tsv', summaries, SUMMARY_FIELDS)
    report = build_markdown(input_dir, summaries, {
        'positive_fraction': args.minimum_positive_fold_fraction,
        'pair_win_rate': args.reference_pair_win_rate,
    })
    (output_dir / 'report.md').write_text(report, encoding='utf-8')
    print('Degree-stratum analysis written to: %s' % output_dir)


if __name__ == '__main__':
    main()
