#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Frozen-teacher feasibility audit for compound-side counterfactual '
            'context distillation on one Strict inner-validation fold.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--holdout-ratio', type=float, default=0.2)
    parser.add_argument('--student-seed', type=int, default=52026)
    parser.add_argument('--counterfactual-seed', type=int, default=42026)
    parser.add_argument('--draws', type=int, default=5)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--margin-total-weight', type=float, default=0.5)
    parser.add_argument('--minimum-eval-compounds', type=int, default=100)
    parser.add_argument('--minimum-context-coverage', type=float, default=0.95)
    parser.add_argument('--minimum-counterfactual-coverage', type=float, default=0.80)
    parser.add_argument('--minimum-teacher-spearman', type=float, default=0.70)
    parser.add_argument('--minimum-margin-spearman', type=float, default=0.50)
    parser.add_argument('--minimum-margin-sign-agreement', type=float, default=0.70)
    parser.add_argument('--maximum-context-aupr-drop', type=float, default=0.005)
    parser.add_argument('--maximum-ccd-vs-kd-aupr-drop', type=float, default=0.001)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def labels_from_records(records):
    return np.asarray(
        [int(float(row[2]) > 0) for row in records], dtype=np.int32
    )


def pair_indices(snapshot, records):
    compound_indices = np.asarray(
        [snapshot['compound_map'][str(row[0])] for row in records],
        dtype=np.int64,
    )
    protein_indices = np.asarray(
        [snapshot['protein_map'][str(row[1])] for row in records],
        dtype=np.int64,
    )
    return compound_indices, protein_indices


def safe_correlation(function, left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    value = function(left, right)[0]
    return float(value) if np.isfinite(value) else None


def ranking_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int32)
    logits = np.asarray(logits, dtype=np.float64)
    return {
        'AUC': float(roc_auc_score(labels, logits)),
        'AUPR': float(average_precision_score(labels, logits)),
    }


def student_metrics(labels, logits, teacher_logits):
    result = ranking_metrics(labels, logits)
    result.update({
        'teacher_pearson': safe_correlation(pearsonr, logits, teacher_logits),
        'teacher_spearman': safe_correlation(spearmanr, logits, teacher_logits),
        'teacher_logit_mae': float(np.mean(np.abs(logits - teacher_logits))),
    })
    return result


def add_intercept(features, difference=False):
    from util.counterfactual_distillation import explicit_intercept
    return explicit_intercept(features, difference=difference)


def write_tsv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return '%.6f' % value
    return str(value)


def matching_audit(matching):
    pool_sizes = np.asarray(list(matching['pool_sizes'].values()), dtype=np.int64)
    return {
        'draws': matching['draws'],
        'seed': matching['seed'],
        'requested_sources': matching['requested_sources'],
        'eligible_sources': matching['eligible_sources'],
        'source_coverage': float(
            matching['eligible_sources'] / float(matching['requested_sources'])
        ) if matching['requested_sources'] else 0.0,
        'donor_compounds': matching['donor_compounds'],
        'pool_size': {
            'minimum': int(np.min(pool_sizes)) if pool_sizes.size else 0,
            'median': float(np.median(pool_sizes)) if pool_sizes.size else 0.0,
            'mean': float(np.mean(pool_sizes)) if pool_sizes.size else 0.0,
            'maximum': int(np.max(pool_sizes)) if pool_sizes.size else 0,
        },
    }


def build_markdown(metadata, analysis):
    metrics = analysis['evaluation_metrics']
    margin = analysis['counterfactual_margin_evaluation']
    criteria = analysis['criteria']
    lines = [
        '# 反事实上下文蒸馏可行性审计',
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '- Teacher 与超图编码器完全冻结。',
        '- Student 只拟合 Strict inner-validation 的一部分 compound；评价 compound 与拟合 compound 不重叠。',
        '- Student 输入仅包含 H-C 药材上下文、protein 表示及二者的 pair 特征，不含 compound ID embedding。',
        '- outer-test 未读取、未评分，也未用于选择方法。',
        '',
        '## 排名结果',
        '',
        '| 模型 | AUC | AUPR | Teacher Spearman | Teacher logit MAE |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in ('teacher', 'frozen_context_head', 'kd_student', 'ccd_student'):
        row = metrics[name]
        lines.append('| %s | %s | %s | %s | %s |' % (
            name,
            format_value(row['AUC']),
            format_value(row['AUPR']),
            format_value(row.get('teacher_spearman')),
            format_value(row.get('teacher_logit_mae')),
        ))
    lines.extend([
        '',
        '## 反事实 margin',
        '',
        '| 指标 | KD | CCD |',
        '|---|---:|---:|',
        '| Teacher margin Spearman | %s | %s |' % (
            format_value(margin['kd']['teacher_margin_spearman']),
            format_value(margin['ccd']['teacher_margin_spearman']),
        ),
        '| Teacher margin sign agreement | %s | %s |' % (
            format_value(margin['kd']['teacher_margin_sign_agreement']),
            format_value(margin['ccd']['teacher_margin_sign_agreement']),
        ),
        '| Teacher margin MAE | %s | %s |' % (
            format_value(margin['kd']['teacher_margin_mae']),
            format_value(margin['ccd']['teacher_margin_mae']),
        ),
        '',
        '## 预注册条件',
        '',
        '| 条件 | 是否通过 |',
        '|---|---|',
    ])
    for name, passed in criteria.items():
        lines.append('| %s | %s |' % (name, '是' if passed else '否'))
    lines.extend([
        '',
        '## 边界',
        '',
        '- 这是冻结表示上的低成本可蒸馏性审计，不是最终 cold-start 性能。',
        '- Teacher checkpoint 曾使用完整 inner-validation 做早停；本审计只能筛选机制，不能作为独立泛化结果。',
        '- 药材超边表示由冻结 Teacher 学得，仍可能携带训练图统计。真正的归纳结论必须在 compound 实体级隔离训练中复验。',
        '- 合成 donor 仅表示 H-C degree 匹配且药材集合不相交的上下文扰动，不代表已识别的生物学因果反事实。',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    if args.draws <= 0:
        raise ValueError('--draws must be positive.')
    if args.ridge_alpha < 0:
        raise ValueError('--ridge-alpha must be non-negative.')
    if args.margin_total_weight < 0:
        raise ValueError('--margin-total-weight must be non-negative.')

    from rating import resolve_dataset_file
    from tools.analyze_context_subgroups import (
        checkpoint_audit,
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
        restore_snapshot,
        score_snapshot,
        sha256_file,
    )
    from tools.audit_counterfactual_herb_context import (
        read_hc_memberships,
        score_with_counterfactual_contexts,
    )
    from util.counterfactual_distillation import (
        build_cross_pool_counterfactuals,
        context_pair_features,
        grouped_compound_holdout,
        standardize_factual_features,
    )

    protocol = prepare_protocol(args.config, args.fold)
    records = protocol['validation']
    if not records:
        raise ValueError('The audit requires a non-empty Strict inner-validation split.')
    labels = labels_from_records(records)
    compound_ids = np.asarray([str(row[0]) for row in records], dtype=object)
    fit_mask, evaluation_mask = grouped_compound_holdout(
        compound_ids, args.holdout_ratio, args.student_seed
    )
    fit_records = [row for row, keep in zip(records, fit_mask) if keep]
    evaluation_records = [row for row, keep in zip(records, evaluation_mask) if keep]
    if len(np.unique(labels_from_records(fit_records))) != 2:
        raise ValueError('Student fit records must contain both labels.')
    if len(np.unique(labels_from_records(evaluation_records))) != 2:
        raise ValueError('Student evaluation records must contain both labels.')

    fit_compounds = sorted({str(row[0]) for row in fit_records})
    evaluation_compounds = sorted({str(row[0]) for row in evaluation_records})
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    memberships = read_hc_memberships(hc_path)
    fit_matching = build_cross_pool_counterfactuals(
        fit_compounds,
        fit_compounds,
        memberships,
        draws=args.draws,
        seed=args.counterfactual_seed,
    )
    evaluation_matching = build_cross_pool_counterfactuals(
        evaluation_compounds,
        fit_compounds,
        memberships,
        draws=args.draws,
        seed=args.counterfactual_seed + 1,
    )

    metadata = {
        'evaluation_type': 'counterfactual_context_distillation_frozen_teacher_audit',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'selection_split': 'strict_inner_validation_grouped_compound_holdout',
        'teacher_optimizer_steps': 0,
        'outer_test_scored': False,
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'hc_path': str(hc_path),
        'hc_sha256': sha256_file(hc_path),
        'student': {
            'model': 'linear_ridge_pair_student',
            'features': 'H-C context, protein embedding, product, absolute difference',
            'compound_id_embedding': False,
            'holdout_ratio': args.holdout_ratio,
            'seed': args.student_seed,
            'ridge_alpha': args.ridge_alpha,
            'margin_total_weight': args.margin_total_weight,
            'draws': args.draws,
        },
        'fit_matching': matching_audit(fit_matching),
        'evaluation_matching': matching_audit(evaluation_matching),
    }
    print('Counterfactual context distillation audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  checkpoint: %s' % checkpoint)
    print('  student compounds: fit=%d evaluation=%d overlap=0' % (
        len(fit_compounds), len(evaluation_compounds)
    ))
    print('  outer-test: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI
    from util.model_components import context_masked_pair_scores

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('Frozen CHCR teacher restored.')
    expected_terms = {
        'compound_disease': False,
        'herb_protein': True,
        'herb_disease': False,
    }
    if snapshot['context_terms'] != expected_terms:
        raise ValueError('The audit requires static Hctx-P; found %s.' % snapshot['context_terms'])
    if not protocol['conf'].contains('counterfactual.context') or (
            protocol['conf']['counterfactual.context'].strip().lower() != 'true'):
        raise ValueError('The audit requires a frozen CHCR checkpoint/config.')

    fit_compound_indices, fit_protein_indices = pair_indices(snapshot, fit_records)
    eval_compound_indices, eval_protein_indices = pair_indices(
        snapshot, evaluation_records
    )
    fit_features = context_pair_features(
        snapshot['compound_context'][fit_compound_indices],
        snapshot['protein'][fit_protein_indices],
    )
    evaluation_features = context_pair_features(
        snapshot['compound_context'][eval_compound_indices],
        snapshot['protein'][eval_protein_indices],
    )
    _, _, scaled = standardize_factual_features(
        fit_features, evaluation_features
    )
    scaled_fit, scaled_evaluation = scaled
    fit_teacher_logits = score_snapshot(
        snapshot, fit_records, include_context=True
    )[1]
    evaluation_teacher_logits = score_snapshot(
        snapshot, evaluation_records, include_context=True
    )[1]

    kd = Ridge(alpha=args.ridge_alpha, fit_intercept=False)
    kd.fit(add_intercept(scaled_fit), fit_teacher_logits)
    kd_evaluation_logits = kd.predict(add_intercept(scaled_evaluation))

    augmented_features = [add_intercept(scaled_fit)]
    augmented_targets = [fit_teacher_logits]
    augmented_weights = [np.ones(len(fit_records), dtype=np.float64)]
    fit_source_ids = np.asarray([str(row[0]) for row in fit_records], dtype=object)
    fit_labels = labels_from_records(fit_records)
    eligible_fit_mask = np.asarray(
        [
            value in fit_matching['assignments'] and label == 1
            for value, label in zip(fit_source_ids, fit_labels)
        ],
        dtype=bool,
    )
    eligible_fit_records = [
        row for row, keep in zip(fit_records, eligible_fit_mask) if keep
    ]
    eligible_fit_features = fit_features[eligible_fit_mask]
    eligible_fit_teacher = fit_teacher_logits[eligible_fit_mask]
    per_draw_weight = (
        args.margin_total_weight / float(args.draws)
        if args.draws else 0.0
    )
    for draw_index in range(args.draws):
        donor_ids = [
            fit_matching['assignments'][str(row[0])][draw_index]
            for row in eligible_fit_records
        ]
        donor_indices = np.asarray(
            [snapshot['compound_map'][value] for value in donor_ids],
            dtype=np.int64,
        )
        donor_features = context_pair_features(
            snapshot['compound_context'][donor_indices],
            snapshot['protein'][fit_protein_indices[eligible_fit_mask]],
        )
        scaled_difference = (
            eligible_fit_features - donor_features
        ) / np.std(fit_features, axis=0).clip(min=1e-8)
        donor_teacher = score_with_counterfactual_contexts(
            snapshot, eligible_fit_records, donor_indices
        )
        augmented_features.append(add_intercept(scaled_difference, difference=True))
        augmented_targets.append(eligible_fit_teacher - donor_teacher)
        augmented_weights.append(np.full(
            len(eligible_fit_records), per_draw_weight, dtype=np.float64
        ))

    ccd = Ridge(alpha=args.ridge_alpha, fit_intercept=False)
    ccd.fit(
        np.concatenate(augmented_features, axis=0),
        np.concatenate(augmented_targets, axis=0),
        sample_weight=np.concatenate(augmented_weights, axis=0),
    )
    ccd_evaluation_logits = ccd.predict(add_intercept(scaled_evaluation))

    dimension = snapshot['compound'].shape[1]
    zero = np.zeros(dimension, dtype=np.float32)
    weights = snapshot['weights']
    frozen_context_logits = context_masked_pair_scores(
        snapshot['compound'],
        snapshot['protein'],
        snapshot['compound_context'],
        snapshot['protein_context'],
        eval_compound_indices,
        eval_protein_indices,
        weights.get('context_compound_disease', zero),
        weights.get('context_herb_protein', zero),
        weights.get('context_herb_disease', zero),
        mask_compound=True,
        mask_protein=False,
        enabled_terms=snapshot['context_terms'],
        decoder_type=snapshot['pair_decoder']['type'],
        decoder_weights=weights,
    )
    evaluation_labels = labels_from_records(evaluation_records)
    evaluation_metrics = {
        'teacher': ranking_metrics(evaluation_labels, evaluation_teacher_logits),
        'frozen_context_head': student_metrics(
            evaluation_labels, frozen_context_logits, evaluation_teacher_logits
        ),
        'kd_student': student_metrics(
            evaluation_labels, kd_evaluation_logits, evaluation_teacher_logits
        ),
        'ccd_student': student_metrics(
            evaluation_labels, ccd_evaluation_logits, evaluation_teacher_logits
        ),
    }

    evaluation_source_ids = np.asarray(
        [str(row[0]) for row in evaluation_records], dtype=object
    )
    eligible_evaluation_mask = np.asarray(
        [
            value in evaluation_matching['assignments'] and label == 1
            for value, label in zip(evaluation_source_ids, evaluation_labels)
        ],
        dtype=bool,
    )
    eligible_evaluation_records = [
        row for row, keep in zip(evaluation_records, eligible_evaluation_mask)
        if keep
    ]
    factual_kd = kd_evaluation_logits[eligible_evaluation_mask]
    factual_ccd = ccd_evaluation_logits[eligible_evaluation_mask]
    factual_teacher = evaluation_teacher_logits[eligible_evaluation_mask]
    factual_features = evaluation_features[eligible_evaluation_mask]
    protein_indices = eval_protein_indices[eligible_evaluation_mask]
    teacher_margins = []
    kd_margins = []
    ccd_margins = []
    pair_rows = []
    fit_scale = np.std(fit_features, axis=0).clip(min=1e-8)
    fit_mean = np.mean(fit_features, axis=0)
    for draw_index in range(args.draws):
        donor_ids = [
            evaluation_matching['assignments'][str(row[0])][draw_index]
            for row in eligible_evaluation_records
        ]
        donor_indices = np.asarray(
            [snapshot['compound_map'][value] for value in donor_ids],
            dtype=np.int64,
        )
        donor_features = context_pair_features(
            snapshot['compound_context'][donor_indices],
            snapshot['protein'][protein_indices],
        )
        scaled_donors = (donor_features - fit_mean) / fit_scale
        donor_kd = kd.predict(add_intercept(scaled_donors))
        donor_ccd = ccd.predict(add_intercept(scaled_donors))
        donor_teacher = score_with_counterfactual_contexts(
            snapshot, eligible_evaluation_records, donor_indices
        )
        teacher_margins.append(factual_teacher - donor_teacher)
        kd_margins.append(factual_kd - donor_kd)
        ccd_margins.append(factual_ccd - donor_ccd)
    teacher_margins = np.concatenate(teacher_margins)
    kd_margins = np.concatenate(kd_margins)
    ccd_margins = np.concatenate(ccd_margins)

    def margin_metrics(values):
        nonzero = np.abs(teacher_margins) > 1e-12
        return {
            'teacher_margin_pearson': safe_correlation(
                pearsonr, values, teacher_margins
            ),
            'teacher_margin_spearman': safe_correlation(
                spearmanr, values, teacher_margins
            ),
            'teacher_margin_mae': float(np.mean(np.abs(values - teacher_margins))),
            'teacher_margin_sign_agreement': float(np.mean(
                np.sign(values[nonzero]) == np.sign(teacher_margins[nonzero])
            )),
        }

    margin_evaluation = {
        'records': len(eligible_evaluation_records),
        'draws': args.draws,
        'comparisons': int(len(teacher_margins)),
        'positive_record_coverage': float(
            len(eligible_evaluation_records) /
            float(max(1, np.sum(evaluation_labels == 1)))
        ),
        'kd': margin_metrics(kd_margins),
        'ccd': margin_metrics(ccd_margins),
    }
    context_coverage = float(np.mean(
        np.linalg.norm(
            snapshot['compound_context'][eval_compound_indices], axis=1
        ) > 0
    ))
    ccd_metrics = evaluation_metrics['ccd_student']
    kd_metrics = evaluation_metrics['kd_student']
    frozen_metrics = evaluation_metrics['frozen_context_head']
    ccd_margin = margin_evaluation['ccd']
    criteria = {
        'evaluation_compounds': len(evaluation_compounds) >= args.minimum_eval_compounds,
        'context_coverage': context_coverage >= args.minimum_context_coverage,
        'counterfactual_record_coverage': (
            margin_evaluation['positive_record_coverage']
            >= args.minimum_counterfactual_coverage
        ),
        'ccd_context_head_noninferiority': (
            ccd_metrics['AUPR'] >= frozen_metrics['AUPR'] - args.maximum_context_aupr_drop
        ),
        'ccd_kd_noninferiority': (
            ccd_metrics['AUPR'] >= kd_metrics['AUPR'] - args.maximum_ccd_vs_kd_aupr_drop
        ),
        'teacher_rank_transfer': (
            ccd_metrics['teacher_spearman'] is not None
            and ccd_metrics['teacher_spearman'] >= args.minimum_teacher_spearman
        ),
        'counterfactual_margin_transfer': (
            ccd_margin['teacher_margin_spearman'] is not None
            and ccd_margin['teacher_margin_spearman'] >= args.minimum_margin_spearman
        ),
        'counterfactual_direction_transfer': (
            ccd_margin['teacher_margin_sign_agreement']
            >= args.minimum_margin_sign_agreement
        ),
    }
    decision = (
        'supports_ccd_compound_cold_start_pilot'
        if all(criteria.values()) else
        'stop_ccd_route_before_joint_training'
    )
    analysis = {
        'decision': decision,
        'split': {
            'fit_records': len(fit_records),
            'evaluation_records': len(evaluation_records),
            'fit_compounds': len(fit_compounds),
            'evaluation_compounds': len(evaluation_compounds),
            'compound_overlap': len(set(fit_compounds) & set(evaluation_compounds)),
            'evaluation_context_coverage': context_coverage,
        },
        'evaluation_metrics': evaluation_metrics,
        'counterfactual_margin_evaluation': margin_evaluation,
        'criteria': criteria,
    }

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'counterfactual_context_distillation' /
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'report.json').write_text(
        json.dumps({'metadata': metadata, 'analysis': analysis}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'report.md').write_text(
        build_markdown(metadata, analysis), encoding='utf-8'
    )
    for index, row in enumerate(evaluation_records):
        pair_rows.append({
            'compound_id': str(row[0]),
            'protein_id': str(row[1]),
            'label': int(float(row[2]) > 0),
            'teacher_logit': float(evaluation_teacher_logits[index]),
            'frozen_context_logit': float(frozen_context_logits[index]),
            'kd_student_logit': float(kd_evaluation_logits[index]),
            'ccd_student_logit': float(ccd_evaluation_logits[index]),
            'counterfactual_eligible': int(eligible_evaluation_mask[index]),
        })
    write_tsv(output_dir / 'evaluation_pairs.tsv', pair_rows)

    print('\nDecision: %s' % decision)
    for name, row in evaluation_metrics.items():
        print('  %s: AUPR=%.6f AUC=%.6f teacher_spearman=%s' % (
            name,
            row['AUPR'],
            row['AUC'],
            format_value(row.get('teacher_spearman')),
        ))
    print('  CCD margin: spearman=%s sign_agreement=%.6f coverage=%.2f%%' % (
        format_value(ccd_margin['teacher_margin_spearman']),
        ccd_margin['teacher_margin_sign_agreement'],
        100.0 * margin_evaluation['positive_record_coverage'],
    ))
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
