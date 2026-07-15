#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Pure-inference feasibility audit for degree-matched counterfactual '
            'herb contexts on one Strict inner-validation fold.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--draws', type=int, default=20)
    parser.add_argument('--counterfactual-seed', type=int, default=42026)
    parser.add_argument('--minimum-group-pairs', type=int, default=30)
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate inputs and matching coverage without loading TensorFlow.',
    )
    return parser.parse_args()


def read_hc_memberships(path, allowed_compounds=None):
    allowed = (
        {str(value) for value in allowed_compounds}
        if allowed_compounds is not None else None
    )
    memberships = defaultdict(set)
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            herb_id, compound_id = str(parts[0]), str(parts[1])
            if allowed is None or compound_id in allowed:
                memberships[compound_id].add(herb_id)
    return dict(memberships)


def score_with_counterfactual_contexts(snapshot, records, donor_indices):
    from util.model_components import context_interaction_pair_scores

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
    donor_contexts = snapshot['compound_context'][np.asarray(donor_indices)]
    dimension = snapshot['compound'].shape[1]
    zero_weight = np.zeros(dimension, dtype=np.float32)
    weights = snapshot['weights']
    return np.asarray(context_interaction_pair_scores(
        snapshot['compound'],
        snapshot['protein'],
        snapshot['compound_context'],
        snapshot['protein_context'],
        compound_indices,
        protein_indices,
        weights.get('context_compound_disease', zero_weight),
        weights.get('context_herb_protein', zero_weight),
        weights.get('context_herb_disease', zero_weight),
        enabled_terms=snapshot['context_terms'],
        decoder_type=snapshot['pair_decoder']['type'],
        decoder_weights=weights,
        pair_compound_contexts=donor_contexts,
    ), dtype=np.float64)


def flatten_subgroups(subgroups):
    rows = []
    for subgroup, groups in subgroups.items():
        for group, summary in groups.items():
            row = {'subgroup': subgroup, 'group': group}
            row.update(summary)
            ci = row.pop('pair_win_rate_ci95', [None, None])
            row['pair_win_rate_ci95_lower'] = ci[0]
            row['pair_win_rate_ci95_upper'] = ci[1]
            rows.append(row)
    return rows


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


def markdown_value(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return '%.6f' % value
    return str(value)


def build_markdown(metadata, analysis):
    positive = analysis['positive_pairs']
    counterfactual = analysis['counterfactual_AUPR']
    lines = [
        '# 反事实药材上下文纯推理审计',
        '',
        '- Fold: `%d/%d`' % (metadata['fold'], metadata['fold_count']),
        '- 评价范围：Strict inner-validation；不读取 outer-test 指标。',
        '- Checkpoint: `%s`' % metadata['checkpoint']['prefix'],
        '- 反事实规则：相同 H-C degree、药材集合完全不相交、按 compound 固定替换。',
        '- Draws / seed: `%d / %d`' % (
            metadata['matching']['draws'], metadata['matching']['seed']
        ),
        '',
        '## 判定',
        '',
        '**%s**' % analysis['decision'],
        '',
        '| 条件 | 阈值 | 结果 | 通过 |',
        '|---|---:|---:|---|',
        '| 可审计记录覆盖率 | >= %.2f | %s | %s |' % (
            analysis['pre_registered_thresholds']['coverage'],
            markdown_value(analysis['coverage']['fraction']),
            analysis['criteria']['coverage'],
        ),
        '| 正样本 pair 胜率 | >= %.2f | %s | %s |' % (
            analysis['pre_registered_thresholds']['positive_pair_win_rate'],
            markdown_value(positive['pair_win_rate']),
            analysis['criteria']['positive_pair_win_rate'],
        ),
        '| 正样本平均 logit margin | > 0 | %s | %s |' % (
            markdown_value(positive['mean_margin']),
            analysis['criteria']['positive_mean_margin'],
        ),
        '| Factual - counterfactual AUPR | >= %.3f | %s | %s |' % (
            analysis['pre_registered_thresholds']['AUPR_drop'],
            markdown_value(counterfactual['mean_factual_minus_counterfactual']),
            analysis['criteria']['AUPR_drop'],
        ),
        '| 正方向 degree strata | >= %.2f | %s | %s |' % (
            analysis['pre_registered_thresholds']['degree_strata_consistency'],
            markdown_value(analysis['degree_strata_positive_fraction']),
            analysis['criteria']['degree_strata_consistency'],
        ),
        '',
        '## 主要结果',
        '',
        '| 指标 | 数值 |',
        '|---|---:|',
        '| Factual AUPR | %s |' % markdown_value(analysis['factual_AUPR']),
        '| Counterfactual AUPR mean | %s |' % markdown_value(counterfactual['mean']),
        '| Counterfactual AUPR std | %s |' % markdown_value(counterfactual['std']),
        '| Factual - counterfactual AUPR | %s |' % markdown_value(
            counterfactual['mean_factual_minus_counterfactual']
        ),
        '| Positive pair win rate | %s |' % markdown_value(positive['pair_win_rate']),
        '| Positive compound win rate | %s |' % markdown_value(
            positive['compound_win_rate']
        ),
        '| Positive mean margin | %s |' % markdown_value(positive['mean_margin']),
        '| Positive standardized margin | %s |' % markdown_value(
            positive['standardized_mean_margin']
        ),
        '',
        '## 解释边界',
        '',
        '- 本审计只判断冻结模型是否对真实 H-C 上下文表现出可测依赖，不能证明因果机制。',
        '- 未记录的替换上下文是合成反事实，不代表生物学上的确认错误上下文。',
        '- 多个 C-P pair 可能共享同一 compound；因此同时报告 pair 与 compound 聚合结果。',
        '- 只有判定为 `supports_CHCR_training_pilot` 时，才进入单折 CHCR 训练 Pilot。',
        '',
    ]
    return '\n'.join(lines)


def main():
    args = parse_args()
    if args.draws <= 0:
        raise ValueError('--draws must be positive.')

    from rating import resolve_dataset_file
    from tools.analyze_context_subgroups import (
        checkpoint_audit,
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
        restore_snapshot,
        score_snapshot,
        sha256_file,
        weight_audit,
    )
    from util.context_subgroups import herb_degree_bin, training_cp_degree_bin
    from util.counterfactual_context import (
        build_exact_degree_counterfactuals,
        summarize_counterfactual_audit,
    )

    protocol = prepare_protocol(args.config, args.fold)
    if not protocol['validation']:
        raise ValueError(
            'The counterfactual audit requires a non-empty inner-validation split.'
        )
    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    validation_compounds = sorted({str(row[0]) for row in protocol['validation']})
    memberships = read_hc_memberships(hc_path)
    matching = build_exact_degree_counterfactuals(
        memberships.keys(),
        memberships,
        draws=args.draws,
        seed=args.counterfactual_seed,
    )
    assignments = matching['assignments']
    covered_validation_compounds = {
        compound_id for compound_id in validation_compounds
        if compound_id in assignments
    }
    covered_validation_records = sum(
        str(row[0]) in covered_validation_compounds
        for row in protocol['validation']
    )

    metadata = {
        'evaluation_type': 'counterfactual_herb_context_pure_inference',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'selection_split': 'strict_inner_validation',
        'optimizer_steps': 0,
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'side_information': {
            'hc_path': str(hc_path),
            'hc_sha256': sha256_file(hc_path),
        },
        'matching': {
            'rule': 'exact_hc_degree_and_disjoint_herb_set',
            'draws': matching['draws'],
            'seed': matching['seed'],
            'dataset_compounds': matching['requested_compounds'],
            'dataset_eligible_compounds': matching['eligible_compounds'],
            'validation_compounds': len(validation_compounds),
            'covered_validation_compounds': len(covered_validation_compounds),
            'validation_records': len(protocol['validation']),
            'covered_validation_records': int(covered_validation_records),
        },
    }
    print('Counterfactual herb-context audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  split: Strict inner-validation only')
    print('  checkpoint: %s' % checkpoint)
    print('  optimizer/training steps: disabled')
    print('  exact-match record coverage: %d/%d' % (
        covered_validation_records, len(protocol['validation'])
    ))
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    print('HerbOnly checkpoint restored.')
    expected_terms = {
        'compound_disease': False,
        'herb_protein': True,
        'herb_disease': False,
    }
    if snapshot['context_terms'] != expected_terms:
        raise ValueError(
            'The audit requires frozen static HerbOnly terms; found %s.' %
            snapshot['context_terms']
        )
    if protocol['conf'].contains('context.herb_protein.mode'):
        mode = protocol['conf']['context.herb_protein.mode'].strip().lower()
        if mode != 'static':
            raise ValueError('The audit requires context.herb_protein.mode=static.')
    if 'context_herb_protein' not in snapshot['weights']:
        raise ValueError('Checkpoint does not contain context_herb_protein.')
    metadata['herb_context_weight'] = weight_audit(
        snapshot['weights']['context_herb_protein']
    )

    records = [
        row for row in protocol['validation']
        if str(row[0]) in assignments
        and str(row[0]) in snapshot['compound_map']
    ]
    if not records:
        raise ValueError('No validation records have an exact counterfactual match.')
    base_logits, factual_logits = score_snapshot(
        snapshot, records, include_context=True
    )
    compound_ids = [str(row[0]) for row in records]
    protein_ids = [str(row[1]) for row in records]
    labels = np.asarray(
        [int(float(row[2]) > 0) for row in records], dtype=np.int32
    )
    counterfactual_logits = []
    donor_ids_by_draw = []
    for draw_index in range(args.draws):
        donor_ids = [assignments[compound_id][draw_index] for compound_id in compound_ids]
        donor_indices = [snapshot['compound_map'][donor_id] for donor_id in donor_ids]
        counterfactual_logits.append(score_with_counterfactual_contexts(
            snapshot, records, donor_indices
        ))
        donor_ids_by_draw.append(donor_ids)
    counterfactual_logits = np.asarray(counterfactual_logits, dtype=np.float64)

    hc_degrees = [len(memberships[compound_id]) for compound_id in compound_ids]
    train_cp_degrees = Counter(
        str(row[0]) for row in protocol['model_train'] if float(row[2]) > 0
    )
    raw_train_cp_degrees = [
        train_cp_degrees.get(compound_id, 0) for compound_id in compound_ids
    ]
    subgroup_values = {
        'H-C degree': [herb_degree_bin(value) for value in hc_degrees],
        'training C-P degree': [
            training_cp_degree_bin(value) for value in raw_train_cp_degrees
        ],
    }
    analysis = summarize_counterfactual_audit(
        labels,
        compound_ids,
        factual_logits,
        counterfactual_logits,
        subgroup_values=subgroup_values,
        requested_records=len(protocol['validation']),
        minimum_group_pairs=args.minimum_group_pairs,
    )
    mean_counterfactual_logits = analysis.pop('mean_counterfactual_logits')
    mean_margins = analysis.pop('mean_margins')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'counterfactual_context' / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'metadata': metadata, 'analysis': analysis}
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (output_dir / 'report.md').write_text(
        build_markdown(metadata, analysis), encoding='utf-8'
    )
    write_tsv(output_dir / 'draw_metrics.tsv', analysis['draws'])
    write_tsv(
        output_dir / 'subgroup_metrics.tsv',
        flatten_subgroups(analysis['subgroups']),
    )

    pair_rows = []
    factual_context_logits = factual_logits - base_logits
    mean_counterfactual_context_logits = mean_counterfactual_logits - base_logits
    for index, (compound_id, protein_id, label) in enumerate(
            zip(compound_ids, protein_ids, labels)):
        donors = [donor_ids_by_draw[draw][index] for draw in range(args.draws)]
        pair_rows.append({
            'compound_id': compound_id,
            'protein_id': protein_id,
            'label': int(label),
            'hc_degree': hc_degrees[index],
            'hc_degree_group': subgroup_values['H-C degree'][index],
            'training_cp_degree': raw_train_cp_degrees[index],
            'training_cp_degree_group': subgroup_values['training C-P degree'][index],
            'counterfactual_pool_size': matching['pool_sizes'][compound_id],
            'unique_donors_used': len(set(donors)),
            'donor_compound_ids': ','.join(donors),
            'base_logit': float(base_logits[index]),
            'factual_context_logit': float(factual_context_logits[index]),
            'factual_total_logit': float(factual_logits[index]),
            'mean_counterfactual_context_logit': float(
                mean_counterfactual_context_logits[index]
            ),
            'mean_counterfactual_total_logit': float(
                mean_counterfactual_logits[index]
            ),
            'factual_minus_counterfactual_logit': float(mean_margins[index]),
            'factual_draw_win_rate': float(np.mean(
                factual_logits[index] > counterfactual_logits[:, index]
            )),
        })
    write_tsv(output_dir / 'pair_margins.tsv', pair_rows)

    print('\nDecision: %s' % analysis['decision'])
    print('  factual AUPR: %.6f' % analysis['factual_AUPR'])
    print('  counterfactual AUPR: %.6f (delta %.6f)' % (
        analysis['counterfactual_AUPR']['mean'],
        analysis['counterfactual_AUPR']['mean_factual_minus_counterfactual'],
    ))
    print('  positive pair win rate: %.6f' % (
        analysis['positive_pairs']['pair_win_rate']
    ))
    print('  positive mean margin: %.6f' % (
        analysis['positive_pairs']['mean_margin']
    ))
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
