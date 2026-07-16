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
            'Frozen-encoder low-rank bilinear probe for latent herb-disease '
            'context alignment on one Strict inner-validation fold.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--probe-seeds', type=int, nargs='+', default=[82026, 82027, 82028])
    parser.add_argument('--audit-split-seed', type=int, default=62026)
    parser.add_argument('--permutation-draws', type=int, default=200)
    parser.add_argument('--permutation-seed', type=int, default=92026)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def metrics(labels, logits):
    return {
        'AUC': float(roc_auc_score(labels, logits)),
        'AUPR': float(average_precision_score(labels, logits)),
    }


def degree_bin(value):
    value = int(value)
    if value <= 0:
        return '0'
    power = int(np.floor(np.log2(value)))
    return '%d-%d' % (2 ** power, 2 ** (power + 1) - 1)


def write_tsv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(metadata, analysis):
    ensemble = analysis['ensemble']
    permutation = analysis['degree_stratified_permutation']
    lines = [
        '# 潜在药材—疾病低秩投影 Probe',
        '',
        '- 编码器和原模型参数完全冻结；只训练 rank-%d probe。' % metadata['probe']['rank'],
        '- Probe train / audit 均来自 Strict inner-validation 的确定性二分。',
        '- Outer-test 不计算指标、不参与模型或方法选择。',
        '- H-D 未使用；输入仅为模型已有的 H-C/P-D 聚合上下文。',
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '| 条件 | 阈值 | 结果 | 通过 |',
        '|---|---:|---:|---|',
        '| 非零上下文覆盖率 | >= 0.95 | %.6f | %s |' % (
            analysis['coverage']['fraction'], analysis['criteria']['coverage']),
        '| 三个 probe seed AUPR 均为正增益 | 3/3 | %d/3 | %s |' % (
            analysis['positive_seed_count'], analysis['criteria']['all_seeds_positive']),
        '| Ensemble audit AUPR 增量 | >= 0.001 | %.6f | %s |' % (
            ensemble['delta']['AUPR'], analysis['criteria']['AUPR_delta']),
        '| 度数分层置换检验 | p <= 0.05 | %.6f | %s |' % (
            permutation['one_sided_p'], analysis['criteria']['permutation']),
        '| 正负类 residual 分离 | > 0 | %.6f | %s |' % (
            ensemble['positive_minus_negative_residual'], analysis['criteria']['separation']),
        '',
        '## 结果',
        '',
        '| Probe seed | 初始 loss | 最终 loss | Audit AUC 增量 | Audit AUPR 增量 |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in analysis['probe_runs']:
        lines.append('| %d | %.6f | %.6f | %+.6f | %+.6f |' % (
            row['seed'], row['initial_loss'], row['final_loss'],
            row['audit_delta']['AUC'], row['audit_delta']['AUPR']))
    lines.extend([
        '',
        '| Ensemble 指标 | Baseline | Probe | 增量 |',
        '|---|---:|---:|---:|',
        '| AUC | %.6f | %.6f | %+.6f |' % (
            ensemble['baseline']['AUC'], ensemble['fused']['AUC'],
            ensemble['delta']['AUC']),
        '| AUPR | %.6f | %.6f | %+.6f |' % (
            ensemble['baseline']['AUPR'], ensemble['fused']['AUPR'],
            ensemble['delta']['AUPR']),
        '',
        '## 边界',
        '',
        '- 这是冻结表示上的方法筛选 probe，不是最终模型结果。',
        '- Probe 使用聚合上下文，只用于判断跨空间可学习性；通过后仍需实现并验证集合级稀疏桥接。',
        '- checkpoint 曾用完整 inner-validation early stopping，因此这不是完全独立重复；任何正结果都必须在新的训练协议中复验。',
        '- 未通过时终止当前潜在桥接路线，不继续搜索 rank、学习率或正则强度。',
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
        raise ValueError('The low-rank probe requires a non-empty inner-validation split.')
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    pd_path = Path(resolve_dataset_file(str(dataset_dir), 'P_D')).resolve()
    metadata = {
        'evaluation_type': 'latent_herb_disease_low_rank_frozen_encoder_probe',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'optimizer_steps_on_encoder': 0,
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
        'probe': {
            'rank': args.rank,
            'steps': args.steps,
            'learning_rate': args.learning_rate,
            'l2': args.l2,
            'seeds': args.probe_seeds,
            'audit_split_seed': args.audit_split_seed,
            'permutation_draws': args.permutation_draws,
            'permutation_seed': args.permutation_seed,
        },
    }
    print('Latent herb-disease low-rank probe')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  encoder/checkpoint: frozen')
    print('  rank=%d steps=%d seeds=%s' % (
        args.rank, args.steps, ','.join(map(str, args.probe_seeds))))
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
        low_rank_bilinear_residual,
        stratified_half_split,
        train_low_rank_bilinear_probe,
    )

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('Frozen checkpoint restored.')
    records = protocol['validation']
    compound_indices = np.asarray([
        snapshot['compound_map'][str(row[0])] for row in records
    ], dtype=np.int64)
    protein_indices = np.asarray([
        snapshot['protein_map'][str(row[1])] for row in records
    ], dtype=np.int64)
    labels = np.asarray([int(float(row[2]) > 0) for row in records], dtype=np.int32)
    herb_contexts = snapshot['compound_context'][compound_indices]
    disease_contexts = snapshot['protein_context'][protein_indices]
    _, baseline_logits = score_snapshot(snapshot, records, include_context=True)
    covered = (
        (np.linalg.norm(herb_contexts, axis=1) > 0)
        & (np.linalg.norm(disease_contexts, axis=1) > 0)
    )
    positions = np.flatnonzero(covered)
    labels = labels[covered]
    baseline_logits = baseline_logits[covered]
    herb_contexts = herb_contexts[covered]
    disease_contexts = disease_contexts[covered]
    compound_indices = compound_indices[covered]
    protein_indices = protein_indices[covered]
    selection, audit = stratified_half_split(labels, args.audit_split_seed)
    baseline_audit = metrics(labels[audit], baseline_logits[audit])
    probe_runs = []
    residuals = []
    for seed in args.probe_seeds:
        probe = train_low_rank_bilinear_probe(
            herb_contexts[selection],
            disease_contexts[selection],
            labels[selection],
            baseline_logits[selection],
            rank=args.rank,
            steps=args.steps,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=seed,
        )
        residual = low_rank_bilinear_residual(
            herb_contexts, disease_contexts, probe['left'], probe['right']
        )
        residuals.append(residual)
        fused_audit = metrics(labels[audit], baseline_logits[audit] + residual[audit])
        probe_runs.append({
            'seed': int(seed),
            'initial_loss': probe['initial_loss'],
            'final_loss': probe['final_loss'],
            'selection': metrics(
                labels[selection], baseline_logits[selection] + residual[selection]
            ),
            'audit': fused_audit,
            'audit_delta': {
                key: float(fused_audit[key] - baseline_audit[key])
                for key in ('AUC', 'AUPR')
            },
        })
    ensemble_residual = np.mean(np.asarray(residuals), axis=0)
    ensemble_fused = metrics(
        labels[audit], baseline_logits[audit] + ensemble_residual[audit]
    )
    ensemble_delta = {
        key: float(ensemble_fused[key] - baseline_audit[key])
        for key in ('AUC', 'AUPR')
    }

    compound_herbs = build_incidence_index(
        hc_path, snapshot['compound_map'], snapshot['herb_map'], entity_column=1
    )
    protein_diseases = build_incidence_index(
        pd_path, snapshot['protein_map'], snapshot['disease_map'], entity_column=0
    )
    herb_degrees = np.asarray([
        len(compound_herbs.get(int(index), ())) for index in compound_indices
    ], dtype=np.int64)
    disease_degrees = np.asarray([
        len(protein_diseases.get(int(index), ())) for index in protein_indices
    ], dtype=np.int64)
    strata = np.asarray([
        '%s|%s' % (degree_bin(herb), degree_bin(disease))
        for herb, disease in zip(herb_degrees[audit], disease_degrees[audit])
    ], dtype=object)
    permutation = degree_stratified_permutation(
        labels[audit],
        baseline_logits[audit],
        ensemble_residual[audit],
        1.0,
        strata,
        draws=args.permutation_draws,
        seed=args.permutation_seed,
    )
    positive_residual = ensemble_residual[audit][labels[audit] == 1]
    negative_residual = ensemble_residual[audit][labels[audit] == 0]
    separation = float(np.mean(positive_residual) - np.mean(negative_residual))
    positive_seed_count = sum(
        row['audit_delta']['AUPR'] > 0 for row in probe_runs
    )
    criteria = {
        'coverage': bool(np.mean(covered) >= 0.95),
        'all_seeds_positive': bool(positive_seed_count == len(args.probe_seeds)),
        'AUPR_delta': bool(ensemble_delta['AUPR'] >= 0.001),
        'permutation': bool(permutation['one_sided_p'] <= 0.05),
        'separation': bool(separation > 0),
    }
    decision = (
        'supports_trainable_sparse_set_bridge_pilot'
        if all(criteria.values())
        else 'stop_latent_bridge_route_use_context_masked_inductive_training'
    )
    analysis = {
        'decision': decision,
        'pre_registered_thresholds': {
            'coverage': 0.95,
            'positive_probe_seeds': len(args.probe_seeds),
            'ensemble_AUPR_delta': 0.001,
            'permutation_p': 0.05,
            'positive_residual_separation': 0.0,
        },
        'criteria': criteria,
        'coverage': {
            'records': int(np.sum(covered)),
            'total_records': len(records),
            'fraction': float(np.mean(covered)),
        },
        'nested_split': {
            'selection_records': len(selection),
            'audit_records': len(audit),
            'seed': args.audit_split_seed,
        },
        'positive_seed_count': int(positive_seed_count),
        'probe_runs': probe_runs,
        'ensemble': {
            'baseline': baseline_audit,
            'fused': ensemble_fused,
            'delta': ensemble_delta,
            'standalone_residual': metrics(labels[audit], ensemble_residual[audit]),
            'positive_mean_residual': float(np.mean(positive_residual)),
            'negative_mean_residual': float(np.mean(negative_residual)),
            'positive_minus_negative_residual': separation,
        },
        'degree_stratified_permutation': permutation,
    }

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPOSITORY_ROOT / 'results' / 'latent_hd_bridge_probe' /
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_by_position = {int(value): 'selection' for value in selection}
    split_by_position.update({int(value): 'audit' for value in audit})
    rows = []
    for local_index, original_position in enumerate(positions):
        rows.append({
            'compound_id': str(records[original_position][0]),
            'protein_id': str(records[original_position][1]),
            'label': int(labels[local_index]),
            'nested_split': split_by_position[local_index],
            'baseline_logit': float(baseline_logits[local_index]),
            'ensemble_residual': float(ensemble_residual[local_index]),
            'fused_logit': float(
                baseline_logits[local_index] + ensemble_residual[local_index]
            ),
            'herb_degree': int(herb_degrees[local_index]),
            'disease_degree': int(disease_degrees[local_index]),
        })
    write_tsv(output_dir / 'pair_scores.tsv', rows)
    payload = {'metadata': metadata, 'analysis': analysis}
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (output_dir / 'report.md').write_text(
        build_markdown(metadata, analysis), encoding='utf-8'
    )
    print('  coverage: %d/%d (%.2f%%)' % (
        analysis['coverage']['records'], analysis['coverage']['total_records'],
        100.0 * analysis['coverage']['fraction']))
    for row in probe_runs:
        print('  seed %d: loss %.6f -> %.6f audit AUPR delta %+.6f' % (
            row['seed'], row['initial_loss'], row['final_loss'],
            row['audit_delta']['AUPR']))
    print('  ensemble audit AUPR: %.6f -> %.6f (delta %+.6f)' % (
        baseline_audit['AUPR'], ensemble_fused['AUPR'], ensemble_delta['AUPR']))
    print('  degree-stratified permutation p: %.6f' % permutation['one_sided_p'])
    print('  decision: %s' % decision)
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
