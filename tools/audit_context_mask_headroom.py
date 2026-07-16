#!/usr/bin/env python3
import argparse
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
            'Frozen-checkpoint headroom audit for compound/protein context masking '
            'on one Strict inner-validation fold.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--minimum-aupr', type=float, default=0.8)
    parser.add_argument('--minimum-retention', type=float, default=0.85)
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def metrics(labels, logits):
    return {
        'AUC': float(roc_auc_score(labels, logits)),
        'AUPR': float(average_precision_score(labels, logits)),
    }


def build_markdown(metadata, analysis):
    lines = [
        '# 上下文掩码归纳训练可行性审计',
        '',
        '- 模型与 checkpoint 完全冻结，没有优化器更新。',
        '- 只计算 Strict inner-validation；outer-test 不计算指标、不参与方法选择。',
        '- 成分掩码：用候选成分的 H-C 药材上下文替代成分 ID 表示。',
        '- 蛋白掩码：用候选蛋白的 P-D 疾病上下文替代蛋白 ID 表示。',
        '- 不读取 H-D。',
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '| 模式 | AUC | AUPR | AUPR 保留率 | 达到 Pilot 门槛 |',
        '|---|---:|---:|---:|---|',
    ]
    for mode in ('full', 'compound_masked', 'protein_masked', 'both_masked'):
        row = analysis['modes'][mode]
        lines.append('| %s | %.6f | %.6f | %.6f | %s |' % (
            mode, row['AUC'], row['AUPR'], row['AUPR_retention'],
            row.get('eligible_for_pilot', '-')))
    lines.extend([
        '',
        '## 预注册门槛',
        '',
        '- 单侧 masked AUPR >= %.2f。' % metadata['thresholds']['minimum_AUPR'],
        '- 单侧 masked/full AUPR 保留率 >= %.2f。' % (
            metadata['thresholds']['minimum_AUPR_retention']),
        '- 上下文非零覆盖率 >= 0.95。',
        '- 双侧均通过时进入 dual-side CMIT Pilot；仅一侧通过时只训练该侧。',
        '',
        '## 边界',
        '',
        '- 本审计只说明冻结模型的侧信息是否具备基本可预测性，不是 CMIT 性能结果。',
        '- 当前是 transductive Strict fold；真正的 cold-start 结论必须使用实体级隔离划分。',
        '- 掩码上下文仍由训练图上的共享编码器产生，不能表述为完全不依赖训练实体。',
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
    )

    protocol = prepare_protocol(args.config, args.fold)
    if not protocol['validation']:
        raise ValueError('Context-mask audit requires a non-empty inner-validation split.')
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    metadata = {
        'evaluation_type': 'context_mask_headroom_frozen_checkpoint_audit',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'optimizer_steps': 0,
        'outer_test_scored': False,
        'outer_test_used_for_selection': False,
        'transductive_entity_universe': True,
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'thresholds': {
            'minimum_AUPR': args.minimum_aupr,
            'minimum_AUPR_retention': args.minimum_retention,
            'minimum_context_coverage': 0.95,
        },
    }
    print('Context-mask headroom audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  checkpoint: %s' % checkpoint)
    print('  optimizer/training steps: disabled')
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
    print('Frozen checkpoint restored.')
    records = protocol['validation']
    compound_indices = np.asarray([
        snapshot['compound_map'][str(row[0])] for row in records
    ], dtype=np.int64)
    protein_indices = np.asarray([
        snapshot['protein_map'][str(row[1])] for row in records
    ], dtype=np.int64)
    labels = np.asarray([int(float(row[2]) > 0) for row in records], dtype=np.int32)
    dimension = snapshot['compound'].shape[1]
    zero = np.zeros(dimension, dtype=np.float32)
    weights = snapshot['weights']

    mode_flags = {
        'full': (False, False),
        'compound_masked': (True, False),
        'protein_masked': (False, True),
        'both_masked': (True, True),
    }
    mode_metrics = {}
    for mode, (mask_compound, mask_protein) in mode_flags.items():
        logits = context_masked_pair_scores(
            snapshot['compound'],
            snapshot['protein'],
            snapshot['compound_context'],
            snapshot['protein_context'],
            compound_indices,
            protein_indices,
            weights.get('context_compound_disease', zero),
            weights.get('context_herb_protein', zero),
            weights.get('context_herb_disease', zero),
            mask_compound=mask_compound,
            mask_protein=mask_protein,
            enabled_terms=snapshot['context_terms'],
            decoder_type=snapshot['pair_decoder']['type'],
            decoder_weights=weights,
        )
        mode_metrics[mode] = metrics(labels, logits)

    full_aupr = mode_metrics['full']['AUPR']
    herb_norms = np.linalg.norm(
        snapshot['compound_context'][compound_indices], axis=1
    )
    disease_norms = np.linalg.norm(
        snapshot['protein_context'][protein_indices], axis=1
    )
    coverage = {
        'compound_context': float(np.mean(herb_norms > 0)),
        'protein_context': float(np.mean(disease_norms > 0)),
        'both_contexts': float(np.mean((herb_norms > 0) & (disease_norms > 0))),
    }
    eligible = []
    for mode, row in mode_metrics.items():
        row['AUPR_retention'] = float(row['AUPR'] / full_aupr)
        if mode in ('compound_masked', 'protein_masked'):
            context_key = (
                'compound_context' if mode == 'compound_masked'
                else 'protein_context'
            )
            row['eligible_for_pilot'] = bool(
                row['AUPR'] >= args.minimum_aupr
                and row['AUPR_retention'] >= args.minimum_retention
                and coverage[context_key] >= 0.95
            )
            if row['eligible_for_pilot']:
                eligible.append(mode)
    if len(eligible) == 2:
        decision = 'supports_dual_side_cmit_pilot'
    elif eligible == ['compound_masked']:
        decision = 'supports_compound_side_cmit_pilot'
    elif eligible == ['protein_masked']:
        decision = 'supports_protein_side_cmit_pilot'
    else:
        decision = 'stop_context_masked_training_route'
    analysis = {
        'decision': decision,
        'eligible_modes': eligible,
        'records': len(records),
        'class_balance': {
            'positives': int(np.sum(labels == 1)),
            'negatives': int(np.sum(labels == 0)),
        },
        'context_coverage': coverage,
        'modes': mode_metrics,
    }

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        REPOSITORY_ROOT / 'results' / 'context_mask_headroom' /
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
    print('  context coverage: compound %.2f%% protein %.2f%%' % (
        100.0 * coverage['compound_context'], 100.0 * coverage['protein_context']))
    for mode in mode_flags:
        row = mode_metrics[mode]
        print('  %s: AUC %.6f AUPR %.6f retention %.2f%%' % (
            mode, row['AUC'], row['AUPR'], 100.0 * row['AUPR_retention']))
    print('  decision: %s' % decision)
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
