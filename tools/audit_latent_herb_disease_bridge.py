#!/usr/bin/env python3
import argparse
import csv
import json
import sys
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
            'Frozen-checkpoint feasibility audit for a sparse latent herb-disease '
            'set bridge on one Strict inner-validation fold.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--audit-split-seed', type=int, default=62026)
    parser.add_argument('--permutation-draws', type=int, default=200)
    parser.add_argument('--permutation-seed', type=int, default=72026)
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate protocol, relation files, and checkpoint without TensorFlow.',
    )
    return parser.parse_args()


def degree_bin(value):
    value = int(value)
    if value <= 0:
        return '0'
    lower_power = int(np.floor(np.log2(value)))
    lower = 2 ** lower_power
    upper = 2 ** (lower_power + 1) - 1
    return '%d-%d' % (lower, upper)


def write_tsv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(labels, scores):
    return {
        'AUC': float(roc_auc_score(labels, scores)),
        'AUPR': float(average_precision_score(labels, scores)),
    }


def build_markdown(metadata, analysis):
    selected = analysis['residual_selection']['selected']
    delta = analysis['residual_selection']['audit_delta']
    permutation = analysis['degree_stratified_permutation']
    separation = analysis['residual_selection']['audit_feature_separation']
    lines = [
        '# 潜在药材—疾病集合桥接可行性审计',
        '',
        '- 评价范围：Strict inner-validation 再二分；未计算 outer-test 指标，也未用于方法选择。',
        '- Checkpoint：`%s`' % metadata['checkpoint']['prefix'],
        '- Fold：`%d/%d`' % (metadata['fold'], metadata['fold_count']),
        '- 优化器更新：`0`；模型参数：冻结。',
        '- 关系输入：独立 H-C 与 P-D；未读取 H-D。',
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '| 条件 | 预注册阈值 | 结果 | 通过 |',
        '|---|---:|---:|---|',
        '| 上下文覆盖率 | >= 0.95 | %.6f | %s |' % (
            analysis['coverage']['fraction'], analysis['criteria']['coverage']),
        '| 选择的正残差系数 | > 0 | %.6f | %s |' % (
            selected['alpha'], analysis['criteria']['positive_alpha']),
        '| Audit AUPR 增量 | >= 0.001 | %.6f | %s |' % (
            delta['AUPR'], analysis['criteria']['AUPR_delta']),
        '| 度数分层置换检验 | p <= 0.05 | %.6f | %s |' % (
            permutation['one_sided_p'], analysis['criteria']['permutation']),
        '| 正负类对齐分离 | > 0 | %.6f | %s |' % (
            separation['positive_minus_negative'], analysis['criteria']['separation']),
        '',
        '## 主要结果',
        '',
        '| 项目 | 数值 |',
        '|---|---:|',
        '| 选择的聚合特征 | `%s` |' % selected['feature'],
        '| 选择的残差系数 | %.6f |' % selected['alpha'],
        '| Selection baseline AUPR | %.6f |' % (
            analysis['residual_selection']['selection_baseline']['AUPR']),
        '| Selection fused AUPR | %.6f |' % selected['selection_AUPR'],
        '| Audit baseline AUPR | %.6f |' % (
            analysis['residual_selection']['audit_baseline']['AUPR']),
        '| Audit fused AUPR | %.6f |' % (
            analysis['residual_selection']['audit_fused']['AUPR']),
        '| Audit AUPR 增量 | %.6f |' % delta['AUPR'],
        '| Audit AUC 增量 | %.6f |' % delta['AUC'],
        '| 置换零分布 AUPR 增量均值 | %.6f |' % (
            permutation['null_mean_AUPR_delta']),
        '| 置换零分布 AUPR 增量 P95 | %.6f |' % (
            permutation['null_p95_AUPR_delta']),
        '',
        '## 解释边界',
        '',
        '- 原始药材与疾病超边空间没有为彼此对齐而训练；本结果只评估其现成余弦几何是否含有增量信号。',
        '- 未通过不能否定带低秩投影的可训练桥接，只表示不能直接使用冻结空间的无参数 Top-K 对齐。',
        '- Selection 半集只用于选择 Top-K 与非负残差系数；Audit 半集只评价一次。',
        '- 度数分层置换保留药材/疾病上下文规模分布，用于排除明显的上下文度数替代解释。',
        '- 本审计用于方法筛选，不是最终泛化实验，也不支持因果或生物学关联断言。',
        '- checkpoint 本身曾使用完整 inner-validation 进行 early stopping；本次再二分只隔离桥接特征和残差系数的选择，不能视为完全独立重复。',
        '',
    ]
    return '\n'.join(lines)


def main():
    args = parse_args()
    if args.permutation_draws <= 0:
        raise ValueError('--permutation-draws must be positive.')

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
        raise ValueError('The bridge audit requires a non-empty inner-validation split.')
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    pd_path = Path(resolve_dataset_file(str(dataset_dir), 'P_D')).resolve()
    metadata = {
        'evaluation_type': 'latent_herb_disease_bridge_frozen_checkpoint_audit',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'selection_split': 'strict_inner_validation_nested_half',
        'optimizer_steps': 0,
        'outer_test_scored': False,
        'outer_test_used_for_selection': False,
        'transductive_entity_universe': True,
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'side_information': {
            'hc_path': str(hc_path),
            'hc_sha256': sha256_file(hc_path),
            'pd_path': str(pd_path),
            'pd_sha256': sha256_file(pd_path),
            'hd_used': False,
        },
        'audit': {
            'top_k': sorted(set(args.top_k)),
            'split_seed': args.audit_split_seed,
            'permutation_draws': args.permutation_draws,
            'permutation_seed': args.permutation_seed,
        },
    }
    print('Latent herb-disease bridge audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  split: Strict inner-validation nested half')
    print('  checkpoint: %s' % checkpoint)
    print('  optimizer/training steps: disabled')
    print('  H-D relation: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI
    from util.latent_context_bridge import (
        build_incidence_index,
        degree_stratified_permutation,
        pair_alignment_scores,
        select_positive_residual,
        stratified_half_split,
    )

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('Frozen checkpoint restored.')
    compound_herbs = build_incidence_index(
        hc_path, snapshot['compound_map'], snapshot['herb_map'], entity_column=1
    )
    protein_diseases = build_incidence_index(
        pd_path, snapshot['protein_map'], snapshot['disease_map'], entity_column=0
    )
    records = protocol['validation']
    compound_indices = np.asarray(
        [snapshot['compound_map'][str(row[0])] for row in records], dtype=np.int64
    )
    protein_indices = np.asarray(
        [snapshot['protein_map'][str(row[1])] for row in records], dtype=np.int64
    )
    labels = np.asarray([int(float(row[2]) > 0) for row in records], dtype=np.int32)
    _, baseline_logits = score_snapshot(snapshot, records, include_context=True)
    alignment = pair_alignment_scores(
        snapshot['herb_edge'],
        snapshot['disease_edge'],
        compound_herbs,
        protein_diseases,
        compound_indices,
        protein_indices,
        top_ks=args.top_k,
    )
    covered = alignment['covered']
    if np.sum(covered) < 4 or len(np.unique(labels[covered])) < 2:
        raise ValueError('Insufficient covered records for the nested bridge audit.')
    covered_positions = np.flatnonzero(covered)
    covered_labels = labels[covered]
    covered_baseline = baseline_logits[covered]
    covered_features = {
        key: values[covered] for key, values in alignment['scores'].items()
    }
    selection, audit = stratified_half_split(
        covered_labels, seed=args.audit_split_seed
    )
    residual = select_positive_residual(
        covered_labels,
        covered_baseline,
        covered_features,
        selection,
        audit,
    )
    audit_herb_degrees = alignment['herb_degrees'][covered][audit]
    audit_disease_degrees = alignment['disease_degrees'][covered][audit]
    strata = np.asarray([
        '%s|%s' % (degree_bin(herb), degree_bin(disease))
        for herb, disease in zip(audit_herb_degrees, audit_disease_degrees)
    ], dtype=object)
    permutation = degree_stratified_permutation(
        covered_labels[audit],
        covered_baseline[audit],
        residual['audit_feature'],
        residual['selected']['alpha'],
        strata,
        draws=args.permutation_draws,
        seed=args.permutation_seed,
    )
    standalone = {
        name: metric_summary(covered_labels[audit], values[audit])
        for name, values in covered_features.items()
    }
    criteria = {
        'coverage': bool(np.mean(covered) >= 0.95),
        'positive_alpha': bool(residual['selected']['alpha'] > 0),
        'AUPR_delta': bool(residual['audit_delta']['AUPR'] >= 0.001),
        'permutation': bool(permutation['one_sided_p'] <= 0.05),
        'separation': bool(
            residual['audit_feature_separation']['positive_minus_negative'] > 0
        ),
    }
    decision = (
        'supports_trainable_sparse_bridge_pilot'
        if all(criteria.values())
        else 'raw_alignment_inconclusive_consider_low_rank_probe'
    )
    analysis = {
        'decision': decision,
        'pre_registered_thresholds': {
            'coverage': 0.95,
            'AUPR_delta': 0.001,
            'permutation_p': 0.05,
            'positive_alignment_separation': 0.0,
        },
        'criteria': criteria,
        'coverage': {
            'records': int(np.sum(covered)),
            'total_records': len(records),
            'fraction': float(np.mean(covered)),
        },
        'nested_split': {
            'seed': args.audit_split_seed,
            'selection_records': len(selection),
            'audit_records': len(audit),
        },
        'standalone_audit_metrics': standalone,
        'residual_selection': {
            key: value for key, value in residual.items()
            if key != 'audit_feature'
        },
        'degree_stratified_permutation': permutation,
    }

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPOSITORY_ROOT / 'results' / 'latent_hd_bridge' /
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    split_by_position = {int(value): 'selection' for value in selection}
    split_by_position.update({int(value): 'audit' for value in audit})
    for covered_index, original_index in enumerate(covered_positions):
        row = {
            'compound_id': str(records[original_index][0]),
            'protein_id': str(records[original_index][1]),
            'label': int(labels[original_index]),
            'nested_split': split_by_position[covered_index],
            'baseline_logit': float(baseline_logits[original_index]),
            'herb_degree': int(alignment['herb_degrees'][original_index]),
            'disease_degree': int(alignment['disease_degrees'][original_index]),
        }
        for feature_name, values in alignment['scores'].items():
            row[feature_name] = float(values[original_index])
        rows.append(row)
    write_tsv(output_dir / 'pair_scores.tsv', rows)
    payload = {'metadata': metadata, 'analysis': analysis}
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (output_dir / 'report.md').write_text(
        build_markdown(metadata, analysis), encoding='utf-8'
    )
    print('  coverage: %d/%d (%.2f%%)' % (
        analysis['coverage']['records'],
        analysis['coverage']['total_records'],
        100.0 * analysis['coverage']['fraction'],
    ))
    print('  selected: %s alpha=%.3f' % (
        residual['selected']['feature'], residual['selected']['alpha']))
    print('  audit AUPR: %.6f -> %.6f (delta %+.6f)' % (
        residual['audit_baseline']['AUPR'],
        residual['audit_fused']['AUPR'],
        residual['audit_delta']['AUPR'],
    ))
    print('  degree-stratified permutation p: %.6f' % permutation['one_sided_p'])
    print('  decision: %s' % decision)
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
