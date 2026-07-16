#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Frozen-checkpoint audit of specificity-weighted H-C/P-D hyperedge '
            'contexts on one Strict inner-validation fold.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--audit-split-seed', type=int, default=122026)
    parser.add_argument('--permutation-draws', type=int, default=200)
    parser.add_argument('--permutation-seed', type=int, default=132026)
    parser.add_argument('--minimum-context-change', type=float, default=0.01)
    parser.add_argument('--minimum-broad-reduction', type=float, default=0.10)
    parser.add_argument('--minimum-aupr-delta', type=float, default=0.001)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def metrics(labels, scores):
    return {
        'AUC': float(roc_auc_score(labels, scores)),
        'AUPR': float(average_precision_score(labels, scores)),
    }


def degree_bin(value):
    value = int(value)
    if value <= 0:
        return '0'
    power = int(np.floor(np.log2(value)))
    return '%d-%d' % (2 ** power, 2 ** (power + 1) - 1)


def mean_cosine(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    values = np.sum(left * right, axis=1)
    covered = (np.linalg.norm(left, axis=1) > 0) & (np.linalg.norm(right, axis=1) > 0)
    return float(np.mean(values[covered])) if np.any(covered) else 0.0


def build_markdown(metadata, analysis):
    herb = analysis['contexts']['herb']
    disease = analysis['contexts']['disease']
    residual = analysis.get('residual')
    lines = [
        '# 特异性约束超边重加权冻结审计',
        '',
        '- checkpoint、编码器与超边表示完全冻结，优化步数为 0。',
        '- 只使用 H-C/P-D incidence 与冻结超边表示；H-D 未使用。',
        '- 超边权重固定为 `log(1 + 节点数 / 超边度数)`，不搜索公式或强度。',
        '- outer-test 未计算，selection/audit 来自 Strict inner-validation 的确定性二分。',
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '## 上下文结构审计',
        '',
        '| 侧 | Uniform 快照一致性 | 覆盖率 | 权重 CV | 平均余弦变化 | 宽泛超边质量降低 | 结构可用 |',
        '|---|---:|---:|---:|---:|---:|---|',
        '| Herb / H-C | %.6f | %.6f | %.6f | %.6f | %.6f | %s |' % (
            herb['uniform_snapshot_cosine'], herb['coverage'], herb['weight_cv'],
            herb['mean_cosine_distance'], herb['broad_mass_relative_reduction'],
            analysis['eligible_sides']['herb']),
        '| Disease / P-D | %.6f | %.6f | %.6f | %.6f | %.6f | %s |' % (
            disease['uniform_snapshot_cosine'], disease['coverage'], disease['weight_cv'],
            disease['mean_cosine_distance'], disease['broad_mass_relative_reduction'],
            analysis['eligible_sides']['disease']),
        '',
        '## 冻结特征',
        '',
        '| 特征 | Audit AUC | Audit AUPR | 与 baseline Spearman |',
        '|---|---:|---:|---:|',
    ]
    for name, row in sorted(analysis['standalone_features'].items()):
        lines.append('| %s | %.6f | %.6f | %.6f |' % (
            name, row['AUC'], row['AUPR'], row['baseline_spearman']))
    if residual is not None:
        permutation = analysis['degree_stratified_permutation']
        lines.extend([
            '',
            '## 独立残差审计',
            '',
            '| 条件 | 门槛 | 结果 | 通过 |',
            '|---|---:|---:|---|',
            '| 选择系数 alpha | > 0 | %.6f | %s |' % (
                residual['selected']['alpha'], analysis['criteria']['positive_alpha']),
            '| Audit AUPR 增量 | >= %.6f | %+.6f | %s |' % (
                metadata['thresholds']['minimum_AUPR_delta'],
                residual['audit_delta']['AUPR'], analysis['criteria']['AUPR_delta']),
            '| 度数分层置换检验 | p <= 0.05 | %.6f | %s |' % (
                permutation['one_sided_p'], analysis['criteria']['permutation']),
            '| 正负类特异性特征分离 | > 0 | %.6f | %s |' % (
                residual['audit_feature_separation']['positive_minus_negative'],
                analysis['criteria']['separation']),
            '',
            '选择特征：`%s`；audit baseline/fused AUPR：`%.6f/%.6f`。' % (
                residual['selected']['feature'], residual['audit_baseline']['AUPR'],
                residual['audit_fused']['AUPR']),
        ])
    lines.extend([
        '',
        '## 边界',
        '',
        '- 这是固定权重的冻结表示筛选，不是最终模型或训练结果。',
        '- 若未通过，不搜索 IDF 变体、指数、温度或残差系数网格。',
        '- 若通过，下一步才把相同固定权重接入超图传播，并执行 validation-only Pilot。',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
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

    protocol = prepare_protocol(args.config, args.fold)
    if not protocol['validation']:
        raise ValueError('The specificity audit requires a non-empty inner-validation split.')
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    pd_path = Path(resolve_dataset_file(str(dataset_dir), 'P_D')).resolve()
    metadata = {
        'evaluation_type': 'hyperedge_specificity_frozen_checkpoint_audit',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'optimizer_steps': 0,
        'outer_test_scored': False,
        'outer_test_used_for_selection': False,
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'side_information': {
            'hc_path': str(hc_path),
            'hc_sha256': sha256_file(hc_path),
            'pd_path': str(pd_path),
            'pd_sha256': sha256_file(pd_path),
            'hd_used': False,
            'cp_used_for_specificity': False,
        },
        'specificity': {
            'formula': 'log1p(node_count / hyperedge_degree)',
            'formula_search': False,
        },
        'selection': {
            'audit_split_seed': args.audit_split_seed,
            'residual_alphas': [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            'permutation_draws': args.permutation_draws,
            'permutation_seed': args.permutation_seed,
        },
        'thresholds': {
            'minimum_uniform_snapshot_cosine': 0.999,
            'minimum_coverage': 0.95,
            'minimum_weight_cv': 0.10,
            'minimum_context_change': args.minimum_context_change,
            'minimum_broad_mass_reduction': args.minimum_broad_reduction,
            'minimum_AUPR_delta': args.minimum_aupr_delta,
            'maximum_permutation_p': 0.05,
        },
    }
    print('Hyperedge specificity audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  checkpoint/encoder/hyperedge embeddings: frozen')
    print('  weight=log1p(node_count/hyperedge_degree); no formula search')
    print('  optimizer steps: 0; outer-test: disabled; H-D: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI
    from util.hyperedge_specificity import (
        aggregate_hyperedge_contexts,
        context_change_statistics,
        hyperedge_specificity_weights,
        specificity_pair_features,
    )
    from util.latent_context_bridge import (
        degree_stratified_permutation,
        select_positive_residual,
        stratified_half_split,
    )
    from util.sparse_global_diffusion import build_incidence_matrix

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('Frozen checkpoint restored.')
    hc_incidence, hc_incidence_stats = build_incidence_matrix(
        hc_path,
        snapshot['compound_map'],
        snapshot['herb_map'],
        node_column=1,
        context_column=0,
    )
    pd_incidence, pd_incidence_stats = build_incidence_matrix(
        pd_path,
        snapshot['protein_map'],
        snapshot['disease_map'],
        node_column=0,
        context_column=1,
    )
    herb_weights, _ = hyperedge_specificity_weights(hc_incidence)
    disease_weights, _ = hyperedge_specificity_weights(pd_incidence)
    uniform_herb_context = aggregate_hyperedge_contexts(
        hc_incidence, snapshot['herb_edge']
    )
    uniform_disease_context = aggregate_hyperedge_contexts(
        pd_incidence, snapshot['disease_edge']
    )
    specific_herb_context = aggregate_hyperedge_contexts(
        hc_incidence, snapshot['herb_edge'], herb_weights
    )
    specific_disease_context = aggregate_hyperedge_contexts(
        pd_incidence, snapshot['disease_edge'], disease_weights
    )
    herb_stats = context_change_statistics(
        hc_incidence,
        herb_weights,
        uniform_herb_context,
        specific_herb_context,
    )
    disease_stats = context_change_statistics(
        pd_incidence,
        disease_weights,
        uniform_disease_context,
        specific_disease_context,
    )
    herb_stats['uniform_snapshot_cosine'] = mean_cosine(
        uniform_herb_context, snapshot['compound_context']
    )
    disease_stats['uniform_snapshot_cosine'] = mean_cosine(
        uniform_disease_context, snapshot['protein_context']
    )
    herb_stats['incidence'] = hc_incidence_stats
    disease_stats['incidence'] = pd_incidence_stats
    for name, row in (('herb', herb_stats), ('disease', disease_stats)):
        if row['uniform_snapshot_cosine'] < 0.999:
            raise ValueError(
                'Uniform %s context does not reproduce the frozen model snapshot: %.6f' %
                (name, row['uniform_snapshot_cosine'])
            )

    thresholds = metadata['thresholds']
    eligible_sides = {}
    for name, row in (('herb', herb_stats), ('disease', disease_stats)):
        eligible_sides[name] = bool(
            row['coverage'] >= thresholds['minimum_coverage']
            and row['weight_cv'] >= thresholds['minimum_weight_cv']
            and row['mean_cosine_distance'] >= thresholds['minimum_context_change']
            and row['broad_mass_relative_reduction']
            >= thresholds['minimum_broad_mass_reduction']
        )

    records = protocol['validation']
    compound_indices = np.asarray([
        snapshot['compound_map'][str(row[0])] for row in records
    ], dtype=np.int64)
    protein_indices = np.asarray([
        snapshot['protein_map'][str(row[1])] for row in records
    ], dtype=np.int64)
    labels = np.asarray([int(float(row[2]) > 0) for row in records], dtype=np.int32)
    _, baseline_logits = score_snapshot(snapshot, records, include_context=True)
    zero = np.zeros(snapshot['compound'].shape[1], dtype=np.float32)
    feature_scores = specificity_pair_features(
        snapshot['compound'],
        snapshot['protein'],
        snapshot['compound_context'],
        specific_herb_context,
        specific_disease_context,
        compound_indices,
        protein_indices,
        snapshot['weights'].get('context_herb_protein', zero),
    )
    eligible_features = {}
    if eligible_sides['herb']:
        eligible_features['herb_specificity_replacement_delta'] = (
            feature_scores['herb_specificity_replacement_delta']
        )
    if eligible_sides['disease']:
        eligible_features['compound_specific_disease_cosine'] = (
            feature_scores['compound_specific_disease_cosine']
        )
    if all(eligible_sides.values()):
        eligible_features['specific_context_cosine'] = (
            feature_scores['specific_context_cosine']
        )

    selection, audit = stratified_half_split(labels, args.audit_split_seed)
    standalone = {}
    for name, values in feature_scores.items():
        row = metrics(labels[audit], values[audit])
        row['baseline_spearman'] = float(
            spearmanr(baseline_logits[audit], values[audit]).statistic
        )
        row['structurally_eligible'] = bool(name in eligible_features)
        standalone[name] = row

    residual = None
    permutation = None
    criteria = {
        'eligible_view': bool(eligible_features),
        'positive_alpha': False,
        'AUPR_delta': False,
        'permutation': False,
        'separation': False,
    }
    if eligible_features:
        residual = select_positive_residual(
            labels,
            baseline_logits,
            eligible_features,
            selection,
            audit,
        )
        compound_degree = np.asarray(hc_incidence.sum(axis=1)).reshape(-1)
        protein_degree = np.asarray(pd_incidence.sum(axis=1)).reshape(-1)
        strata = np.asarray([
            '%s|%s' % (
                degree_bin(compound_degree[compound_index]),
                degree_bin(protein_degree[protein_index]),
            )
            for compound_index, protein_index in zip(
                compound_indices[audit], protein_indices[audit]
            )
        ], dtype=object)
        permutation = degree_stratified_permutation(
            labels[audit],
            baseline_logits[audit],
            residual['audit_feature'],
            residual['selected']['alpha'],
            strata,
            draws=args.permutation_draws,
            seed=args.permutation_seed,
        )
        criteria.update({
            'positive_alpha': bool(residual['selected']['alpha'] > 0),
            'AUPR_delta': bool(
                residual['audit_delta']['AUPR'] >= args.minimum_aupr_delta
            ),
            'permutation': bool(permutation['one_sided_p'] <= 0.05),
            'separation': bool(
                residual['audit_feature_separation']['positive_minus_negative'] > 0
            ),
        })
    decision = (
        'supports_specificity_weighted_hypergraph_training_pilot'
        if all(criteria.values()) else 'stop_hyperedge_specificity_route'
    )
    analysis = {
        'decision': decision,
        'records': len(records),
        'selection_records': int(len(selection)),
        'audit_records': int(len(audit)),
        'contexts': {
            'herb': herb_stats,
            'disease': disease_stats,
        },
        'eligible_sides': eligible_sides,
        'eligible_features': sorted(eligible_features),
        'standalone_features': standalone,
        'residual': residual,
        'degree_stratified_permutation': permutation,
        'criteria': criteria,
    }
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPOSITORY_ROOT / 'results' / 'hyperedge_specificity' /
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
    print('  herb: context_change=%.6f broad_reduction=%.2f%% eligible=%s' % (
        herb_stats['mean_cosine_distance'],
        100.0 * herb_stats['broad_mass_relative_reduction'],
        eligible_sides['herb']))
    print('  disease: context_change=%.6f broad_reduction=%.2f%% eligible=%s' % (
        disease_stats['mean_cosine_distance'],
        100.0 * disease_stats['broad_mass_relative_reduction'],
        eligible_sides['disease']))
    if residual is not None:
        print('  selected: %s alpha=%.3f audit_AUPR_delta=%+.6f' % (
            residual['selected']['feature'], residual['selected']['alpha'],
            residual['audit_delta']['AUPR']))
        print('  degree-stratified permutation p=%.6f' % permutation['one_sided_p'])
    print('  decision: %s' % decision)
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
