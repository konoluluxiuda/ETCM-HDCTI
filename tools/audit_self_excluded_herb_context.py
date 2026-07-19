#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_CASES = (
    (
        'TCM-Suite',
        'configs/HDCTI_tcmsuite_cold_start_herb_only_pilot.conf',
        'saved_model/2026-07-16 22-25-13/hdcti_model.ckpt',
    ),
    (
        'TCMSP',
        'configs/HDCTI_tcmsp_cold_start_herb_only_pilot.conf',
        'saved_model/2026-07-16 22-51-03/hdcti_model.ckpt',
    ),
    (
        'SymMap2.0',
        'configs/HDCTI_symmap_cold_start_herb_only_pilot.conf',
        'saved_model/2026-07-17 00-13-31/hdcti_model.ckpt',
    ),
    (
        'ETCM2.0',
        'configs/HDCTI_etcm_mention10_cold_start_herb_only_pilot.conf',
        'saved_model/2026-07-17 10-51-44/hdcti_model.ckpt',
    ),
)

MINIMUM_NONDECREASING_DATASETS = 3
MINIMUM_MACRO_AUPR_GAIN = 0.005
MAXIMUM_DATASET_AUPR_DROP = 0.005
MINIMUM_RECORD_COVERAGE = 0.90
MAXIMUM_RECONSTRUCTION_ERROR = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Frozen four-dataset feasibility audit for a direct self-excluded '
            'herb-context cold-start encoder. Outer-test is never evaluated.'
        )
    )
    parser.add_argument(
        '--case',
        nargs=3,
        action='append',
        metavar=('NAME', 'CONFIG', 'CHECKPOINT'),
        help='Override defaults with one or more named audit cases.',
    )
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument(
        '--device', choices=('cpu', 'config'), default='cpu',
        help='Use CPU by default because the audit performs one frozen forward pass.',
    )
    parser.add_argument(
        '--output-dir',
        default='results/self_excluded_herb_context/frozen_four_dataset_seed2026',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate protocols and checkpoints without importing TensorFlow.',
    )
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def slugify(value):
    return ''.join(
        character.lower() if character.isalnum() else '_'
        for character in value
    ).strip('_')


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'p90': None,
            'maximum': None,
        }
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'maximum': float(np.max(values)),
    }


def metric_delta(current, reference):
    return {
        key: float(current[key] - reference[key])
        for key in ('AUC', 'AUPR')
        if current[key] is not None and reference[key] is not None
    }


def write_tsv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol(name, protocol):
    conf = protocol['conf']
    if not conf.contains('split.strategy'):
        raise ValueError('%s does not declare split.strategy.' % name)
    if conf['split.strategy'].strip().lower() != 'compound_cold_start':
        raise ValueError('%s is not a compound cold-start protocol.' % name)
    if not protocol['validation']:
        raise ValueError('%s has no Strict inner-validation records.' % name)
    if not conf.contains('attention.max.nodes'):
        raise ValueError('%s does not freeze attention.max.nodes.' % name)
    if int(conf['attention.max.nodes']) != 2000:
        raise ValueError(
            '%s must use the existing unified exploratory threshold 2000.' % name
        )
    train_compounds = {str(row[0]) for row in protocol['model_train']}
    validation_compounds = {str(row[0]) for row in protocol['validation']}
    overlap = train_compounds & validation_compounds
    if overlap:
        raise ValueError(
            '%s inner split is not compound-disjoint: %d overlaps.' %
            (name, len(overlap))
        )
    return {
        'training_compounds': len(train_compounds),
        'validation_compounds': len(validation_compounds),
        'compound_overlap': 0,
    }


def prepare_cases(raw_cases, fold):
    from tools.analyze_context_subgroups import (
        checkpoint_audit,
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
    )

    cases = []
    for name, config_value, checkpoint_value in raw_cases:
        config_path = resolve_path(config_value)
        checkpoint, checkpoint_files = normalize_checkpoint(
            resolve_path(checkpoint_value)
        )
        protocol = prepare_protocol(config_path, fold)
        split_audit = validate_protocol(name, protocol)
        cases.append({
            'name': name,
            'slug': slugify(name),
            'config': config_path,
            'checkpoint': checkpoint,
            'checkpoint_files': checkpoint_files,
            'protocol': protocol,
            'fold': fold,
            'metadata': {
                'name': name,
                'protocol': protocol_audit(protocol),
                'split': split_audit,
                'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
            },
        })
    return cases


def audit_case(case, tf, HDCTI, set_global_seed):
    from rating import resolve_dataset_file
    from tools.analyze_context_subgroups import restore_snapshot, score_snapshot
    from util.context_subgroups import binary_metrics
    from util.model_components import resolve_hyperedge_attention
    from util.self_excluded_context import (
        build_direct_self_excluded_contexts,
        cosine_distance_rows,
        herb_protein_context_logits,
        indexed_hc_memberships,
    )

    protocol = case['protocol']
    if resolve_hyperedge_attention(protocol['conf'])['enabled']:
        raise ValueError(
            '%s enables hyperedge attention; the direct mean-edge identity '
            'required by this audit would not hold.' % case['name']
        )
    snapshot = restore_snapshot(
        tf,
        HDCTI,
        set_global_seed,
        protocol,
        case['checkpoint'],
        case['fold'],
        include_context_audit=True,
    )
    expected_terms = {
        'compound_disease': False,
        'herb_protein': True,
        'herb_disease': False,
    }
    if snapshot['context_terms'] != expected_terms:
        raise ValueError(
            '%s is not a frozen static Hctx-P checkpoint: %s.' %
            (case['name'], snapshot['context_terms'])
        )
    if snapshot['pair_decoder']['type'] != 'dot':
        raise ValueError('%s does not use the frozen Dot decoder.' % case['name'])
    if 'context_herb_protein' not in snapshot['weights']:
        raise ValueError('%s checkpoint lacks Hctx-P weights.' % case['name'])

    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    memberships = indexed_hc_memberships(
        hc_path, snapshot['herb_map'], snapshot['compound_map']
    )
    contexts = build_direct_self_excluded_contexts(
        snapshot['herb_edge'],
        snapshot['hc_edge_inputs'],
        memberships['herb_members'],
        memberships['compound_herbs'],
        snapshot['num_compounds'],
    )
    reconstruction_error = contexts['edge_reconstruction_relative_error']
    if (
            reconstruction_error['maximum'] is None
            or reconstruction_error['maximum'] > MAXIMUM_RECONSTRUCTION_ERROR):
        raise ValueError(
            '%s H-C edge reconstruction failed: %s.' %
            (case['name'], reconstruction_error)
        )

    records = protocol['validation']
    compound_indices = np.fromiter(
        (snapshot['compound_map'][str(row[0])] for row in records),
        dtype=np.int64,
        count=len(records),
    )
    protein_indices = np.fromiter(
        (snapshot['protein_map'][str(row[1])] for row in records),
        dtype=np.int64,
        count=len(records),
    )
    labels = np.asarray(
        [int(float(row[2]) > 0) for row in records], dtype=np.int32
    )
    base_logits, current_total_logits = score_snapshot(
        snapshot, records, include_context=True
    )
    current_context_logits = current_total_logits - base_logits
    context_weight = snapshot['weights']['context_herb_protein']
    inclusive_context_logits = herb_protein_context_logits(
        contexts['inclusive_contexts'],
        snapshot['protein'],
        compound_indices,
        protein_indices,
        context_weight,
    )
    excluded_context_logits = herb_protein_context_logits(
        contexts['self_excluded_contexts'],
        snapshot['protein'],
        compound_indices,
        protein_indices,
        context_weight,
    )
    variants = {
        'current_total': current_total_logits,
        'current_context_only': current_context_logits,
        'rebuilt_inclusive_total': base_logits + inclusive_context_logits,
        'self_excluded_total': base_logits + excluded_context_logits,
        'self_excluded_context_only': excluded_context_logits,
    }
    metrics = {
        name: binary_metrics(labels, logits)
        for name, logits in variants.items()
    }
    metrics['base_only'] = binary_metrics(labels, base_logits)

    unique_validation_indices = np.unique(compound_indices)
    eligible_compounds = contexts['eligible'][unique_validation_indices]
    eligible_records = contexts['eligible'][compound_indices]
    inclusive_distances = cosine_distance_rows(
        snapshot['compound_context'],
        contexts['inclusive_contexts'],
        mask=contexts['incident_herb_counts'] > 0,
    )
    exclusion_distances = cosine_distance_rows(
        contexts['inclusive_contexts'],
        contexts['self_excluded_contexts'],
        mask=contexts['eligible'],
    )
    result = {
        'name': case['name'],
        'metadata': case['metadata'],
        'hc_relation': {
            'path': str(hc_path),
            'raw_rows': memberships['raw_rows'],
            'mapped_rows': memberships['mapped_rows'],
            'unique_mapped_rows': memberships['unique_mapped_rows'],
        },
        'reconstruction': {
            'edge_relative_error': reconstruction_error,
            'current_vs_rebuilt_context_cosine_distance': summarize_values(
                inclusive_distances
            ),
            'current_vs_rebuilt_context_logit_max_abs': float(np.max(np.abs(
                current_context_logits - inclusive_context_logits
            ))),
        },
        'coverage': {
            'validation_records': int(len(records)),
            'eligible_records': int(np.sum(eligible_records)),
            'record_fraction': float(np.mean(eligible_records)),
            'validation_compounds': int(unique_validation_indices.size),
            'eligible_compounds': int(np.sum(eligible_compounds)),
            'compound_fraction': float(np.mean(eligible_compounds)),
        },
        'context_change': {
            'inclusive_vs_self_excluded_cosine_distance': summarize_values(
                exclusion_distances
            ),
            'fraction_above_0_01': float(np.mean(exclusion_distances > 0.01))
            if exclusion_distances.size else None,
        },
        'metrics': metrics,
        'deltas': {
            'base_suppression': metric_delta(
                metrics['current_context_only'], metrics['current_total']
            ),
            'direct_self_exclusion': metric_delta(
                metrics['self_excluded_context_only'],
                metrics['current_context_only'],
            ),
            'candidate_vs_current': metric_delta(
                metrics['self_excluded_context_only'], metrics['current_total']
            ),
            'self_exclusion_with_base': metric_delta(
                metrics['self_excluded_total'], metrics['current_total']
            ),
        },
    }
    return result, {
        'compound_id': [str(row[0]) for row in records],
        'protein_id': [str(row[1]) for row in records],
        'label': labels,
        'eligible': eligible_records.astype(np.int32),
        'base_logit': base_logits,
        'current_context_logit': current_context_logits,
        'current_total_logit': current_total_logits,
        'self_excluded_context_logit': excluded_context_logits,
        'self_excluded_total_logit': base_logits + excluded_context_logits,
    }


def pair_rows(columns):
    names = list(columns)
    count = len(columns[names[0]])
    return [
        {
            name: (
                value.item() if isinstance(value, np.generic) else value
            )
            for name, value in (
                (name, columns[name][index]) for name in names
            )
        }
        for index in range(count)
    ]


def aggregate_results(results):
    deltas = np.asarray([
        result['deltas']['candidate_vs_current']['AUPR']
        for result in results
    ], dtype=np.float64)
    base_suppression_deltas = np.asarray([
        result['deltas']['base_suppression']['AUPR']
        for result in results
    ], dtype=np.float64)
    direct_exclusion_deltas = np.asarray([
        result['deltas']['direct_self_exclusion']['AUPR']
        for result in results
    ], dtype=np.float64)
    coverages = np.asarray([
        result['coverage']['record_fraction'] for result in results
    ], dtype=np.float64)
    criteria = {
        'nondecreasing_datasets': int(np.sum(deltas >= 0.0))
        >= MINIMUM_NONDECREASING_DATASETS,
        'macro_AUPR_gain': float(np.mean(deltas)) >= MINIMUM_MACRO_AUPR_GAIN,
        'maximum_dataset_drop': float(np.min(deltas))
        >= -MAXIMUM_DATASET_AUPR_DROP,
        'record_coverage': float(np.min(coverages)) >= MINIMUM_RECORD_COVERAGE,
    }
    return {
        'decision': (
            'go_to_single_fold_training_pilot'
            if all(criteria.values())
            else 'stop_self_excluded_context_route'
        ),
        'candidate': 'self_excluded_context_only',
        'reference': 'current_total',
        'macro_AUPR_gain': float(np.mean(deltas)),
        'macro_base_suppression_AUPR_gain': float(np.mean(base_suppression_deltas)),
        'macro_direct_self_exclusion_AUPR_gain': float(np.mean(direct_exclusion_deltas)),
        'minimum_AUPR_gain': float(np.min(deltas)),
        'maximum_AUPR_gain': float(np.max(deltas)),
        'nondecreasing_datasets': int(np.sum(deltas >= 0.0)),
        'minimum_record_coverage': float(np.min(coverages)),
        'thresholds': {
            'minimum_nondecreasing_datasets': MINIMUM_NONDECREASING_DATASETS,
            'minimum_macro_AUPR_gain': MINIMUM_MACRO_AUPR_GAIN,
            'maximum_dataset_AUPR_drop': MAXIMUM_DATASET_AUPR_DROP,
            'minimum_record_coverage': MINIMUM_RECORD_COVERAGE,
        },
        'criteria': criteria,
    }


def build_markdown(payload):
    aggregate = payload['aggregate']
    lines = [
        '# 自排除药材上下文冻结审计',
        '',
        '- 范围：四库 compound-cold-start fold 1 Strict inner-validation。',
        '- 优化器更新：`0`；outer-test：未读取。',
        '- 历史统一口径：`attention.max.nodes=2000`。该口径只用于低成本可行性审计。',
        '- 候选分数：只保留直接 self-excluded Hctx-P，屏蔽未训练 compound ID 的基础点积。',
        '',
        '## 判定',
        '',
        '**%s**' % aggregate['decision'],
        '',
        '| 数据集 | 当前 AUPR | Self-excluded context-only AUPR | Delta | 覆盖率 |',
        '|---|---:|---:|---:|---:|',
    ]
    for result in payload['datasets']:
        lines.append('| %s | %.6f | %.6f | %+.6f | %.2f%% |' % (
            result['name'],
            result['metrics']['current_total']['AUPR'],
            result['metrics']['self_excluded_context_only']['AUPR'],
            result['deltas']['candidate_vs_current']['AUPR'],
            100.0 * result['coverage']['record_fraction'],
        ))
    lines.extend([
        '',
        '## 增益分解',
        '',
        '| 数据集 | 当前 total | 当前 context-only | Self-excluded context-only | 屏蔽基础点积增量 | 直接自排除增量 |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for result in payload['datasets']:
        lines.append('| %s | %.6f | %.6f | %.6f | %+.6f | %+.6f |' % (
            result['name'],
            result['metrics']['current_total']['AUPR'],
            result['metrics']['current_context_only']['AUPR'],
            result['metrics']['self_excluded_context_only']['AUPR'],
            result['deltas']['base_suppression']['AUPR'],
            result['deltas']['direct_self_exclusion']['AUPR'],
        ))
    lines.extend([
        '',
        '- Macro 基础点积屏蔽增量：`%+.6f`。' % (
            aggregate['macro_base_suppression_AUPR_gain']
        ),
        '- Macro 直接自排除增量：`%+.6f`。' % (
            aggregate['macro_direct_self_exclusion_AUPR_gain']
        ),
        '- 因此本轮正信号主要来自冷启动时屏蔽未训练 compound ID 基础点积；直接自排除只提供接近零的额外 macro 增益，尚不能单独作为已确认创新。',
        '',
        '| 汇总条件 | 阈值 | 结果 | 通过 |',
        '|---|---:|---:|---|',
        '| 非下降数据库数 | >= %d | %d | %s |' % (
            MINIMUM_NONDECREASING_DATASETS,
            aggregate['nondecreasing_datasets'],
            aggregate['criteria']['nondecreasing_datasets'],
        ),
        '| Macro AUPR 增量 | >= %.3f | %+.6f | %s |' % (
            MINIMUM_MACRO_AUPR_GAIN,
            aggregate['macro_AUPR_gain'],
            aggregate['criteria']['macro_AUPR_gain'],
        ),
        '| 最大单库下降 | <= %.3f | %+.6f | %s |' % (
            MAXIMUM_DATASET_AUPR_DROP,
            aggregate['minimum_AUPR_gain'],
            aggregate['criteria']['maximum_dataset_drop'],
        ),
        '| 最低记录覆盖率 | >= %.2f | %.6f | %s |' % (
            MINIMUM_RECORD_COVERAGE,
            aggregate['minimum_record_coverage'],
            aggregate['criteria']['record_coverage'],
        ),
        '',
        '## 解释边界',
        '',
        '- 审计精确删除候选 compound 对每层药材超边均值的直接贡献，但不重新运行整图，因此仍保留早期传播形成的间接影响。',
        '- 现有 Hctx-P 权重未针对 self-excluded 表示重新训练，阴性结果不能证明所有归纳编码器均无效。',
        '- 只有全部预注册条件通过，才允许实现一版固定的训练模块；否则停止该路线，不搜索混合系数或数据集特定门控。',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    raw_cases = args.case if args.case else DEFAULT_CASES
    cases = prepare_cases(raw_cases, args.fold)
    output_dir = resolve_path(args.output_dir)
    base_payload = {
        'created_at': datetime.now().astimezone().isoformat(),
        'evaluation_type': 'frozen_direct_self_excluded_herb_context',
        'fold': args.fold,
        'optimizer_steps': 0,
        'outer_test_evaluated': False,
        'device': args.device,
        'cases': [case['metadata'] for case in cases],
    }
    if args.dry_run:
        print(json.dumps(base_payload, ensure_ascii=False, indent=2))
        return 0

    if args.device == 'cpu':
        os.environ['HDCTI_FORCE_CPU'] = '1'
    from util.gpu import configure_cuda_environment
    configure_cuda_environment(cases[0]['protocol']['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print('\nAuditing %s...' % case['name'])
        result, columns = audit_case(case, tf, HDCTI, set_global_seed)
        results.append(result)
        case_dir = output_dir / case['slug']
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'report.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        write_tsv(case_dir / 'pair_scores.tsv', pair_rows(columns))
        print('  current AUPR: %.6f' % result['metrics']['current_total']['AUPR'])
        print('  candidate AUPR: %.6f (%+.6f)' % (
            result['metrics']['self_excluded_context_only']['AUPR'],
            result['deltas']['candidate_vs_current']['AUPR'],
        ))
        print('  record coverage: %.2f%%' % (
            100.0 * result['coverage']['record_fraction']
        ))

    payload = dict(base_payload)
    payload['datasets'] = results
    payload['aggregate'] = aggregate_results(results)
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'report.md').write_text(
        build_markdown(payload), encoding='utf-8'
    )
    summary_rows = []
    for result in results:
        summary_rows.append({
            'dataset': result['name'],
            'current_AUC': result['metrics']['current_total']['AUC'],
            'current_AUPR': result['metrics']['current_total']['AUPR'],
            'candidate_AUC': result['metrics']['self_excluded_context_only']['AUC'],
            'candidate_AUPR': result['metrics']['self_excluded_context_only']['AUPR'],
            'delta_AUC': result['deltas']['candidate_vs_current']['AUC'],
            'delta_AUPR': result['deltas']['candidate_vs_current']['AUPR'],
            'record_coverage': result['coverage']['record_fraction'],
            'compound_coverage': result['coverage']['compound_fraction'],
        })
    write_tsv(output_dir / 'metrics.tsv', summary_rows)
    print('\nDecision: %s' % payload['aggregate']['decision'])
    print('  macro AUPR gain: %+.6f' % payload['aggregate']['macro_AUPR_gain'])
    print('  nondecreasing datasets: %d/%d' % (
        payload['aggregate']['nondecreasing_datasets'], len(results)
    ))
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
