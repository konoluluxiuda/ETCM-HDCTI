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
            'Audit a frozen Top-K sparse multi-hop H-C/P-D diffusion view on '
            'one Strict inner-validation fold without optimizer steps or outer-test scoring.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--minimum-hop', type=int, default=2)
    parser.add_argument('--maximum-hop', type=int, default=4)
    parser.add_argument('--restart', type=float, default=0.15)
    parser.add_argument('--candidate-multiplier', type=int, default=4)
    parser.add_argument('--audit-split-seed', type=int, default=102026)
    parser.add_argument('--permutation-draws', type=int, default=200)
    parser.add_argument('--permutation-seed', type=int, default=112026)
    parser.add_argument('--minimum-novel-fraction', type=float, default=0.25)
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


def build_markdown(metadata, analysis):
    compound = analysis['diffusion']['compound']
    protein = analysis['diffusion']['protein']
    residual = analysis.get('residual')
    lines = [
        '# Top-K 稀疏多跳全局扩散冻结审计',
        '',
        '- 编码器、CHCR checkpoint 和所有模型参数完全冻结，优化步数为 0。',
        '- 只使用 H-C 与 P-D 固定侧信息；不使用 H-D 或额外 C-P 标签。',
        '- 只累积投影图第 %d-%d 阶路径，一阶局部投影不进入全局视图。' % (
            metadata['diffusion']['minimum_hop'], metadata['diffusion']['maximum_hop']),
        '- 每个节点只保留 Top-%d 非自身邻居；outer-test 未计算。' % (
            metadata['diffusion']['top_k']),
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '## 结构审计',
        '',
        '| 侧 | 覆盖率 | 平均邻居数 | 非一阶邻居比例 | 结构可用 |',
        '|---|---:|---:|---:|---|',
        '| Compound / H-C | %.6f | %.2f | %.6f | %s |' % (
            compound['coverage'], compound['mean_neighbors'],
            compound['novel_edge_fraction'], analysis['eligible_sides']['compound']),
        '| Protein / P-D | %.6f | %.2f | %.6f | %s |' % (
            protein['coverage'], protein['mean_neighbors'],
            protein['novel_edge_fraction'], analysis['eligible_sides']['protein']),
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
            '| 正负类扩散特征分离 | > 0 | %.6f | %s |' % (
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
        '- 这是冻结 checkpoint 的方法筛选，不是训练结果或最终模型指标。',
        '- checkpoint 曾使用完整 inner-validation 早停；selection/audit 二分只用于减少本次系数选择的直接过拟合。',
        '- 若未通过，不搜索 K、扩散阶数、restart 或残差系数网格。',
        '- 若通过，下一步才实现轻量可训练扩散编码器，并使用新的 validation-only Pilot 验证。',
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
        raise ValueError('The diffusion audit requires a non-empty inner-validation split.')
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    pd_path = Path(resolve_dataset_file(str(dataset_dir), 'P_D')).resolve()
    metadata = {
        'evaluation_type': 'sparse_multihop_global_diffusion_frozen_checkpoint_audit',
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
            'cp_used_for_diffusion': False,
        },
        'diffusion': {
            'top_k': args.top_k,
            'minimum_hop': args.minimum_hop,
            'maximum_hop': args.maximum_hop,
            'restart': args.restart,
            'candidate_multiplier': args.candidate_multiplier,
        },
        'selection': {
            'audit_split_seed': args.audit_split_seed,
            'residual_alphas': [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            'permutation_draws': args.permutation_draws,
            'permutation_seed': args.permutation_seed,
        },
        'thresholds': {
            'minimum_coverage': 0.95,
            'minimum_novel_fraction': args.minimum_novel_fraction,
            'minimum_AUPR_delta': args.minimum_aupr_delta,
            'maximum_permutation_p': 0.05,
        },
    }
    print('Sparse global diffusion audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  checkpoint/encoder: frozen')
    print('  K=%d projected_hops=%d-%d restart=%.2f' % (
        args.top_k, args.minimum_hop, args.maximum_hop, args.restart))
    print('  optimizer steps: 0; outer-test: disabled; H-D: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI
    from util.latent_context_bridge import (
        degree_stratified_permutation,
        select_positive_residual,
        stratified_half_split,
    )
    from util.sparse_global_diffusion import (
        build_incidence_matrix,
        diffuse_embeddings,
        global_pair_features,
        sparse_multihop_diffusion,
    )

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('Frozen checkpoint restored.')
    hc_incidence, hc_stats = build_incidence_matrix(
        hc_path,
        snapshot['compound_map'],
        snapshot['herb_map'],
        node_column=1,
        context_column=0,
    )
    pd_incidence, pd_stats = build_incidence_matrix(
        pd_path,
        snapshot['protein_map'],
        snapshot['disease_map'],
        node_column=0,
        context_column=1,
    )
    compound_diffusion, _, compound_stats = sparse_multihop_diffusion(
        hc_incidence,
        top_k=args.top_k,
        minimum_hop=args.minimum_hop,
        maximum_hop=args.maximum_hop,
        restart=args.restart,
        candidate_multiplier=args.candidate_multiplier,
    )
    protein_diffusion, _, protein_stats = sparse_multihop_diffusion(
        pd_incidence,
        top_k=args.top_k,
        minimum_hop=args.minimum_hop,
        maximum_hop=args.maximum_hop,
        restart=args.restart,
        candidate_multiplier=args.candidate_multiplier,
    )
    compound_stats['incidence'] = hc_stats
    protein_stats['incidence'] = pd_stats
    global_compounds = diffuse_embeddings(compound_diffusion, snapshot['compound'])
    global_proteins = diffuse_embeddings(protein_diffusion, snapshot['protein'])

    records = protocol['validation']
    compound_indices = np.asarray([
        snapshot['compound_map'][str(row[0])] for row in records
    ], dtype=np.int64)
    protein_indices = np.asarray([
        snapshot['protein_map'][str(row[1])] for row in records
    ], dtype=np.int64)
    labels = np.asarray([int(float(row[2]) > 0) for row in records], dtype=np.int32)
    _, baseline_logits = score_snapshot(snapshot, records, include_context=True)
    feature_scores = global_pair_features(
        snapshot['compound'],
        snapshot['protein'],
        global_compounds,
        global_proteins,
        compound_indices,
        protein_indices,
    )
    selection, audit = stratified_half_split(labels, args.audit_split_seed)
    eligible_sides = {
        'compound': bool(
            compound_stats['coverage'] >= metadata['thresholds']['minimum_coverage']
            and compound_stats['novel_edge_fraction'] >= args.minimum_novel_fraction
        ),
        'protein': bool(
            protein_stats['coverage'] >= metadata['thresholds']['minimum_coverage']
            and protein_stats['novel_edge_fraction'] >= args.minimum_novel_fraction
        ),
    }
    eligible_features = {}
    if eligible_sides['compound']:
        eligible_features['compound_global_cosine'] = feature_scores['compound_global_cosine']
    if eligible_sides['protein']:
        eligible_features['protein_global_cosine'] = feature_scores['protein_global_cosine']
    if all(eligible_sides.values()):
        eligible_features['dual_global_cosine'] = feature_scores['dual_global_cosine']

    standalone = {}
    for name, values in feature_scores.items():
        correlation = spearmanr(baseline_logits[audit], values[audit]).statistic
        row = metrics(labels[audit], values[audit])
        row['baseline_spearman'] = float(correlation)
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
    passed = all(criteria.values())
    decision = (
        'supports_sparse_global_diffusion_training_pilot'
        if passed else 'stop_sparse_global_diffusion_route'
    )
    analysis = {
        'decision': decision,
        'records': len(records),
        'selection_records': int(len(selection)),
        'audit_records': int(len(audit)),
        'class_balance': {
            'positives': int(np.sum(labels == 1)),
            'negatives': int(np.sum(labels == 0)),
        },
        'diffusion': {
            'compound': compound_stats,
            'protein': protein_stats,
        },
        'eligible_sides': eligible_sides,
        'eligible_features': sorted(eligible_features),
        'standalone_features': standalone,
        'residual': residual,
        'degree_stratified_permutation': permutation,
        'criteria': criteria,
    }
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPOSITORY_ROOT / 'results' / 'sparse_global_diffusion' /
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
    print('  compound: coverage=%.2f%% novelty=%.2f%% mean_neighbors=%.2f' % (
        100.0 * compound_stats['coverage'],
        100.0 * compound_stats['novel_edge_fraction'],
        compound_stats['mean_neighbors']))
    print('  protein: coverage=%.2f%% novelty=%.2f%% mean_neighbors=%.2f' % (
        100.0 * protein_stats['coverage'],
        100.0 * protein_stats['novel_edge_fraction'],
        protein_stats['mean_neighbors']))
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
