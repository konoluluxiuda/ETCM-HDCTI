#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Audit same-compound positive/negative ranking violations on one '
            'Strict inner-validation fold using a frozen checkpoint.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=142026)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def write_tsv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(metadata, analysis):
    summary = analysis['ranking']
    violation_ci = analysis['bootstrap']['macro_violation_rate']
    top1_ci = analysis['bootstrap']['top1_miss_rate']
    lines = [
        '# Compound-Centric 排名目标可行性审计',
        '',
        '- CHCR checkpoint 与编码器完全冻结，优化步数为 0。',
        '- 只使用 Strict inner-validation 中同一 compound 同时出现的正负候选。',
        '- outer-test 未计算；本审计不选择 loss 权重、margin 或采样规则。',
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '| 条件 | 门槛 | 结果 | 通过 |',
        '|---|---:|---:|---|',
        '| 有效 compound | >= %d | %d | %s |' % (
            metadata['thresholds']['minimum_eligible_compounds'],
            summary['eligible_compounds'], analysis['criteria']['eligible_compounds']),
        '| 有效记录覆盖率 | >= %.2f | %.6f | %s |' % (
            metadata['thresholds']['minimum_eligible_record_fraction'],
            summary['eligible_record_fraction'], analysis['criteria']['record_coverage']),
        '| Macro 排序违例率 | >= %.2f | %.6f | %s |' % (
            metadata['thresholds']['minimum_macro_violation_rate'],
            summary['macro_violation_rate'], analysis['criteria']['violation_headroom']),
        '| 违例率 bootstrap 下限 | >= %.2f | %.6f | %s |' % (
            metadata['thresholds']['minimum_violation_ci_lower'],
            violation_ci['lower'], analysis['criteria']['violation_ci']),
        '| Top-1 失误率 | >= %.2f | %.6f | %s |' % (
            metadata['thresholds']['minimum_top1_miss_rate'],
            summary['top1_miss_rate'], analysis['criteria']['top1_headroom']),
        '| Top-1 失误率 bootstrap 下限 | >= %.2f | %.6f | %s |' % (
            metadata['thresholds']['minimum_top1_ci_lower'],
            top1_ci['lower'], analysis['criteria']['top1_ci']),
        '| 广泛存在问题的度数层 | >= %d | %d | %s |' % (
            metadata['thresholds']['minimum_affected_degree_strata'],
            analysis['affected_degree_strata'], analysis['criteria']['broad_across_degree']),
        '',
        '## 冻结排序结果',
        '',
        '| 指标 | 数值 |',
        '|---|---:|',
        '| Validation AUC | %.6f |' % analysis['pair_metrics']['AUC'],
        '| Validation AUPR | %.6f |' % analysis['pair_metrics']['AUPR'],
        '| Macro pairwise accuracy | %.6f |' % summary['macro_pairwise_accuracy'],
        '| Macro violation rate | %.6f |' % summary['macro_violation_rate'],
        '| Micro violation rate | %.6f |' % summary['micro_violation_rate'],
        '| Macro BPR loss | %.6f |' % summary['macro_bpr_loss'],
        '| Macro MRR | %.6f |' % summary['macro_mrr'],
        '| Top-1 hit rate | %.6f |' % summary['top1_hit_rate'],
        '',
        '## 训练 C-P Degree 分层',
        '',
        '| Degree | Compounds | Violation | Top-1 miss | MRR |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in analysis['degree_strata']:
        lines.append('| %s | %d | %.6f | %.6f | %.6f |' % (
            row['degree_bin'], row['compounds'], row['macro_violation_rate'],
            row['top1_miss_rate'], row['macro_mrr']))
    lines.extend([
        '',
        '## 边界',
        '',
        '- 当前候选来自 1:1 随机 inner-validation，不等同于全部 protein 候选排名。',
        '- 正结果只支持运行 validation-only 训练 Pilot；最终仍需固定全候选 Top-K 评价。',
        '- checkpoint 曾使用该 inner-validation 早停，因此本审计只衡量剩余排序违例，不作为独立泛化结果。',
        '- 若未通过，不实现 pairwise/listwise loss，也不搜索 margin 或 loss weight。',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    from tools.analyze_context_subgroups import (
        checkpoint_audit,
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
        restore_snapshot,
        score_snapshot,
    )

    protocol = prepare_protocol(args.config, args.fold)
    if not protocol['validation']:
        raise ValueError('Ranking headroom audit requires inner-validation records.')
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    metadata = {
        'evaluation_type': 'compound_centric_ranking_headroom_frozen_checkpoint_audit',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'optimizer_steps': 0,
        'outer_test_scored': False,
        'outer_test_used_for_selection': False,
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'bootstrap': {
            'draws': args.bootstrap_draws,
            'seed': args.bootstrap_seed,
            'unit': 'compound',
        },
        'thresholds': {
            'minimum_eligible_compounds': 1000,
            'minimum_eligible_record_fraction': 0.40,
            'minimum_macro_violation_rate': 0.05,
            'minimum_violation_ci_lower': 0.04,
            'minimum_top1_miss_rate': 0.08,
            'minimum_top1_ci_lower': 0.06,
            'minimum_degree_stratum_compounds': 100,
            'minimum_stratum_violation_rate': 0.05,
            'minimum_affected_degree_strata': 2,
        },
    }
    print('Compound-centric ranking headroom audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  checkpoint/encoder: frozen; optimizer steps: 0')
    print('  candidate source: Strict inner-validation; outer-test: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI
    from util.compound_ranking import (
        binary_labels_from_records,
        bootstrap_mean_interval,
        compound_group_ranking,
        degree_stratified_ranking,
    )

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('Frozen checkpoint restored.')
    records = protocol['validation']
    _, logits = score_snapshot(snapshot, records, include_context=True)
    labels = binary_labels_from_records(records)
    compound_ids = np.asarray([str(row[0]) for row in records], dtype=object)
    summary, rows = compound_group_ranking(compound_ids, labels, logits)
    training_degrees = Counter(
        str(compound_id)
        for compound_id, _, label in protocol['model_train']
        if float(label) > 0
    )
    degree_strata = degree_stratified_ranking(rows, training_degrees)
    violation_ci = bootstrap_mean_interval(
        [row['violation_rate'] for row in rows],
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    top1_ci = bootstrap_mean_interval(
        [row['top1_miss'] for row in rows],
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed + 1,
    )
    thresholds = metadata['thresholds']
    affected_degree_strata = sum(
        row['compounds'] >= thresholds['minimum_degree_stratum_compounds']
        and row['macro_violation_rate']
        >= thresholds['minimum_stratum_violation_rate']
        for row in degree_strata
    )
    criteria = {
        'eligible_compounds': bool(
            summary['eligible_compounds']
            >= thresholds['minimum_eligible_compounds']
        ),
        'record_coverage': bool(
            summary['eligible_record_fraction']
            >= thresholds['minimum_eligible_record_fraction']
        ),
        'violation_headroom': bool(
            summary['macro_violation_rate']
            >= thresholds['minimum_macro_violation_rate']
        ),
        'violation_ci': bool(
            violation_ci['lower'] >= thresholds['minimum_violation_ci_lower']
        ),
        'top1_headroom': bool(
            summary['top1_miss_rate'] >= thresholds['minimum_top1_miss_rate']
        ),
        'top1_ci': bool(
            top1_ci['lower'] >= thresholds['minimum_top1_ci_lower']
        ),
        'broad_across_degree': bool(
            affected_degree_strata
            >= thresholds['minimum_affected_degree_strata']
        ),
    }
    decision = (
        'supports_compound_centric_ranking_loss_pilot'
        if all(criteria.values()) else 'stop_compound_centric_ranking_loss_route'
    )
    analysis = {
        'decision': decision,
        'records': len(records),
        'pair_metrics': {
            'AUC': float(roc_auc_score(labels, logits)),
            'AUPR': float(average_precision_score(labels, logits)),
        },
        'ranking': summary,
        'bootstrap': {
            'macro_violation_rate': violation_ci,
            'top1_miss_rate': top1_ci,
        },
        'degree_strata': degree_strata,
        'affected_degree_strata': int(affected_degree_strata),
        'criteria': criteria,
    }
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPOSITORY_ROOT / 'results' / 'compound_ranking_headroom' /
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'metadata': metadata, 'analysis': analysis}
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (output_dir / 'report.md').write_text(
        build_markdown(metadata, analysis), encoding='utf-8'
    )
    write_tsv(output_dir / 'per_compound.tsv', rows)
    print('  eligible compounds: %d/%d; record coverage: %.2f%%' % (
        summary['eligible_compounds'], summary['all_compounds'],
        100.0 * summary['eligible_record_fraction']))
    print('  macro violation: %.2f%% (95%% CI %.2f%%-%.2f%%)' % (
        100.0 * summary['macro_violation_rate'],
        100.0 * violation_ci['lower'], 100.0 * violation_ci['upper']))
    print('  top1 miss: %.2f%% (95%% CI %.2f%%-%.2f%%)' % (
        100.0 * summary['top1_miss_rate'],
        100.0 * top1_ci['lower'], 100.0 * top1_ci['upper']))
    print('  affected degree strata: %d' % affected_degree_strata)
    print('  decision: %s' % decision)
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
