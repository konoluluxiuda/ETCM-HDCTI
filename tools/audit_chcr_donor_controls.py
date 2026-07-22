#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


STRATEGIES = (
    ('random', 'Random'),
    ('exact_degree', 'Degree-matched, overlap allowed'),
    ('exact_degree_disjoint', 'Degree-matched, disjoint'),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Compare random, degree-matched, and degree-matched disjoint herb '
            'context donors on one frozen Hctx-P checkpoint.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--draws', type=int, default=20)
    parser.add_argument('--counterfactual-seed', type=int, default=42026)
    parser.add_argument('--minimum-group-pairs', type=int, default=30)
    parser.add_argument('--minimum-common-coverage', type=float, default=0.90)
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate protocol, checkpoint files, and common donor coverage.',
    )
    return parser.parse_args()


def markdown_value(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return '%.6f' % value
    return str(value)


def comparison_decision(
        analyses,
        primary_degree_analysis=None,
        degree_control_overlap_fraction=None,
        minimum_overlap_fraction=0.10,
        minimum_common_coverage=0.90):
    disjoint = analyses['exact_degree_disjoint']
    degree = analyses['exact_degree']
    primary_degree = primary_degree_analysis or degree
    disjoint_margin = disjoint['positive_pairs']['mean_margin']
    degree_margin = degree['positive_pairs']['mean_margin']
    criteria = {
        'degree_control_coverage': (
            primary_degree['coverage']['fraction'] >= minimum_common_coverage
        ),
        'degree_control_positive_pair_win_rate': (
            primary_degree['positive_pairs']['pair_win_rate'] is not None
            and primary_degree['positive_pairs']['pair_win_rate'] >= 0.60
        ),
        'degree_control_positive_mean_margin': (
            primary_degree['positive_pairs']['mean_margin'] is not None
            and primary_degree['positive_pairs']['mean_margin'] > 0.0
        ),
        'degree_control_AUPR_drop': (
            primary_degree['counterfactual_AUPR'][
                'mean_factual_minus_counterfactual'
            ] >= 0.001
        ),
        'degree_control_strata_consistency': (
            primary_degree['degree_strata_positive_fraction'] is not None
            and primary_degree['degree_strata_positive_fraction'] >= 0.75
        ),
        'disjoint_control_coverage': (
            disjoint['coverage']['fraction'] >= minimum_common_coverage
        ),
        'disjoint_positive_pair_win_rate': (
            disjoint['positive_pairs']['pair_win_rate'] is not None
            and disjoint['positive_pairs']['pair_win_rate'] >= 0.60
        ),
        'disjoint_positive_mean_margin': (
            disjoint['positive_pairs']['mean_margin'] is not None
            and disjoint['positive_pairs']['mean_margin'] > 0.0
        ),
        'disjoint_AUPR_drop': (
            disjoint['counterfactual_AUPR'][
                'mean_factual_minus_counterfactual'
            ] >= 0.001
        ),
        'disjoint_degree_strata_consistency': (
            disjoint['degree_strata_positive_fraction'] is not None
            and disjoint['degree_strata_positive_fraction'] >= 0.75
        ),
        'degree_control_overlap_coverage': (
            degree_control_overlap_fraction is not None
            and degree_control_overlap_fraction >= minimum_overlap_fraction
        ),
        'disjoint_margin_exceeds_degree_control': (
            disjoint_margin is not None
            and degree_margin is not None
            and disjoint_margin > degree_margin
        ),
        'disjoint_AUPR_drop_exceeds_degree_control': (
            disjoint['counterfactual_AUPR'][
                'mean_factual_minus_counterfactual'
            ]
            > degree['counterfactual_AUPR'][
                'mean_factual_minus_counterfactual'
            ]
        ),
    }
    beyond_degree_keys = (
        'degree_control_coverage',
        'degree_control_positive_pair_win_rate',
        'degree_control_positive_mean_margin',
        'degree_control_AUPR_drop',
        'degree_control_strata_consistency',
    )
    disjoint_keys = (
        'disjoint_control_coverage',
        'disjoint_positive_pair_win_rate',
        'disjoint_positive_mean_margin',
        'disjoint_AUPR_drop',
        'disjoint_degree_strata_consistency',
    )
    overlap_keys = (
        'disjoint_margin_exceeds_degree_control',
        'disjoint_AUPR_drop_exceeds_degree_control',
    )
    if not criteria['degree_control_coverage']:
        decision = 'inconclusive_degree_control_coverage'
    elif not all(criteria[key] for key in beyond_degree_keys):
        decision = 'does_not_support_context_specificity_beyond_degree'
    elif not criteria['disjoint_control_coverage']:
        decision = (
            'supports_context_specificity_beyond_degree_'
            'disjoint_coverage_inconclusive'
        )
    elif not all(criteria[key] for key in disjoint_keys):
        decision = (
            'supports_context_specificity_beyond_degree_'
            'disjoint_not_confirmed'
        )
    elif not criteria['degree_control_overlap_coverage']:
        decision = (
            'supports_context_specificity_beyond_degree_'
            'disjoint_confirmed_overlap_inconclusive'
        )
    elif all(criteria[key] for key in overlap_keys):
        decision = 'supports_context_specificity_beyond_degree_and_overlap'
    else:
        decision = (
            'supports_context_specificity_beyond_degree_'
            'disjoint_confirmed'
        )
    return {
        'decision': decision,
        'criteria': criteria,
        'thresholds': {
            'degree_control_coverage': minimum_common_coverage,
            'disjoint_control_coverage': minimum_common_coverage,
            'disjoint_positive_pair_win_rate': 0.60,
            'disjoint_AUPR_drop': 0.001,
            'disjoint_degree_strata_consistency': 0.75,
            'degree_control_overlap_fraction': minimum_overlap_fraction,
            'overlap_effect': 'strictly_positive descriptive comparison',
        },
    }


def strategy_row(strategy, label, matching, assignment_audit, analysis):
    return {
        'strategy': strategy,
        'label': label,
        'eligible_compounds': matching['eligible_compounds'],
        'assignment_overlap_fraction': assignment_audit['overlap_fraction'],
        'assignment_degree_matched_fraction': assignment_audit[
            'degree_matched_fraction'
        ],
        'factual_AUPR': analysis['factual_AUPR'],
        'counterfactual_AUPR': analysis['counterfactual_AUPR']['mean'],
        'AUPR_drop': analysis['counterfactual_AUPR'][
            'mean_factual_minus_counterfactual'
        ],
        'positive_pair_win_rate': analysis['positive_pairs']['pair_win_rate'],
        'positive_pair_win_ci95_lower': analysis['positive_pairs'][
            'pair_win_rate_ci95'
        ][0],
        'positive_pair_win_ci95_upper': analysis['positive_pairs'][
            'pair_win_rate_ci95'
        ][1],
        'positive_compound_win_rate': analysis['positive_pairs'][
            'compound_win_rate'
        ],
        'positive_mean_margin': analysis['positive_pairs']['mean_margin'],
        'negative_mean_margin': analysis['negative_pairs']['mean_margin'],
        'degree_strata_positive_fraction': analysis[
            'degree_strata_positive_fraction'
        ],
    }


def build_markdown(metadata, strategy_rows, decision, primary_degree_row):
    lines = [
        '# CHCR donor 对照纯推理审计',
        '',
        '- Fold: `%d/%d`' % (metadata['fold'], metadata['fold_count']),
        '- 评价范围：Strict inner-validation；outer-test 未读取。',
        '- Checkpoint: `%s`' % metadata['checkpoint']['prefix'],
        '- 模型：冻结静态 Hctx-P；CHCR、SDIS 和其他候选模块关闭。',
        '- Draws / seed: `%d / %d`' % (
            metadata['draws'], metadata['counterfactual_seed']
        ),
        '- 三种策略使用共同可审计 records：`%d/%d`（%.2f%%）。' % (
            metadata['common_coverage']['records'],
            metadata['common_coverage']['requested_records'],
            100.0 * metadata['common_coverage']['fraction'],
        ),
        '',
        '## 主度数控制',
        '',
        '主判定使用全部可获得同 H-C degree donor 的 records，不要求 donor 与事实药材集合不相交。',
        '',
        '| Coverage | Factual AUPR | CF AUPR | Factual-CF AUPR | Positive margin | Pair win rate |',
        '|---:|---:|---:|---:|---:|---:|',
        '| %s | %s | %s | %s | %s | %s |' % (
            markdown_value(metadata['degree_control_coverage']['fraction']),
            markdown_value(primary_degree_row['factual_AUPR']),
            markdown_value(primary_degree_row['counterfactual_AUPR']),
            markdown_value(primary_degree_row['AUPR_drop']),
            markdown_value(primary_degree_row['positive_mean_margin']),
            markdown_value(primary_degree_row['positive_pair_win_rate']),
        ),
        '',
        '## 共同子集加强对照',
        '',
        '| Donor | Overlap | Degree match | CF AUPR | Factual-CF AUPR | Positive margin | Pair win rate | 95% CI |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in strategy_rows:
        lines.append(
            '| %s | %s | %s | %s | %s | %s | %s | %s--%s |' % (
                row['label'],
                markdown_value(row['assignment_overlap_fraction']),
                markdown_value(row['assignment_degree_matched_fraction']),
                markdown_value(row['counterfactual_AUPR']),
                markdown_value(row['AUPR_drop']),
                markdown_value(row['positive_mean_margin']),
                markdown_value(row['positive_pair_win_rate']),
                markdown_value(row['positive_pair_win_ci95_lower']),
                markdown_value(row['positive_pair_win_ci95_upper']),
            )
        )
    lines.extend([
        '',
        '## 冻结判定',
        '',
        '**%s**' % decision['decision'],
        '',
        '| 条件 | 通过 |',
        '|---|---|',
    ])
    for name, passed in decision['criteria'].items():
        lines.append('| `%s` | %s |' % (name, passed))
    lines.extend([
        '',
        '## 解释边界',
        '',
        '- Random donor 可同时改变 H-C degree 与药材集合，只作为宽松对照。',
        '- Degree-matched donor 去除度数差异，但允许与事实药材集合重叠。',
        '- Degree-matched disjoint donor 同时控制度数并删除事实药材重叠，是判断上下文特异性的主要对照。',
        '- `disjoint_coverage_inconclusive` 表示主度数控制通过，但同度数且不相交 donor 的覆盖不足 90%。',
        '- `overlap_inconclusive` 表示允许重叠的 donor 实际重叠不足 10%，因此不能把两个同度数策略的差异解释为 overlap 效应。',
        '- 这是冻结模型上的合成干预，不构成药材语义的生物学因果证明。',
        '',
    ])
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
    from tools.audit_counterfactual_herb_context import (
        flatten_subgroups,
        read_hc_memberships,
        score_with_counterfactual_contexts,
        write_tsv,
    )
    from util.context_subgroups import herb_degree_bin, training_cp_degree_bin
    from util.counterfactual_context import (
        build_counterfactual_donors,
        summarize_counterfactual_audit,
        summarize_donor_assignments,
    )

    protocol = prepare_protocol(args.config, args.fold)
    if not protocol['validation']:
        raise ValueError('The audit requires a non-empty inner-validation split.')
    conf = protocol['conf']
    if conf.contains('counterfactual.context') and str(
            conf['counterfactual.context']).strip().lower() == 'true':
        raise ValueError(
            'Use a frozen Hctx-P checkpoint without CHCR training for this audit.'
        )
    split_strategy = (
        str(conf['split.strategy']).strip().lower()
        if conf.contains('split.strategy') else 'pair_stratified'
    )
    if split_strategy != 'pair_stratified':
        raise ValueError(
            'CHCR donor controls require split.strategy=pair_stratified.'
        )

    checkpoint, checkpoint_files = normalize_checkpoint(args.checkpoint)
    dataset_dir = Path(conf['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    memberships = read_hc_memberships(hc_path)
    validation_compounds = sorted({str(row[0]) for row in protocol['validation']})
    matching_sources = (
        validation_compounds if args.dry_run else memberships.keys()
    )
    matchings = {}
    assignment_audits = {}
    for strategy, _ in STRATEGIES:
        matching = build_counterfactual_donors(
            matching_sources,
            memberships,
            draws=args.draws,
            seed=args.counterfactual_seed,
            strategy=strategy,
            donor_compound_ids=memberships.keys(),
            assignment_compound_ids=(
                None if args.dry_run else validation_compounds
            ),
        )
        matchings[strategy] = matching
        assignment_audits[strategy] = summarize_donor_assignments(
            matching['assignments'], memberships
        )
    common_compounds = set(validation_compounds)
    for strategy, _ in STRATEGIES:
        common_compounds.intersection_update(
            matchings[strategy]['assignments'].keys()
        )
    common_records = [
        row for row in protocol['validation'] if str(row[0]) in common_compounds
    ]
    degree_control_compounds = set(validation_compounds).intersection(
        matchings['exact_degree']['assignments']
    )
    degree_control_records = [
        row for row in protocol['validation']
        if str(row[0]) in degree_control_compounds
    ]
    degree_control_fraction = float(
        len(degree_control_records) / float(len(protocol['validation']))
    )
    degree_control_coverage = {
        'compounds': len(degree_control_compounds),
        'requested_compounds': len(validation_compounds),
        'records': len(degree_control_records),
        'requested_records': len(protocol['validation']),
        'fraction': degree_control_fraction,
        'minimum_required': args.minimum_common_coverage,
        'passed': degree_control_fraction >= args.minimum_common_coverage,
    }
    context_supported_compounds = set(validation_compounds).intersection(
        memberships
    )
    exact_degree_compounds = set(validation_compounds).intersection(
        matchings['exact_degree']['assignments']
    )
    disjoint_compounds = set(validation_compounds).intersection(
        matchings['exact_degree_disjoint']['assignments']
    )
    context_supported_records = [
        row for row in protocol['validation']
        if str(row[0]) in context_supported_compounds
    ]

    def lost_records(compounds):
        return sum(
            str(row[0]) in compounds for row in protocol['validation']
        )

    no_context_compounds = set(validation_compounds) - context_supported_compounds
    no_exact_degree_compounds = (
        context_supported_compounds - exact_degree_compounds
    )
    no_disjoint_compounds = exact_degree_compounds - disjoint_compounds
    common_coverage = {
        'compounds': len(common_compounds),
        'requested_compounds': len(validation_compounds),
        'records': len(common_records),
        'requested_records': len(protocol['validation']),
        'fraction': float(len(common_records) / float(len(protocol['validation']))),
        'minimum_required': args.minimum_common_coverage,
        'passed': (
            float(len(common_records) / float(len(protocol['validation'])))
            >= args.minimum_common_coverage
        ),
        'context_supported_records': len(context_supported_records),
        'context_supported_fraction': float(
            len(context_supported_records) / float(len(protocol['validation']))
        ),
        'donor_coverage_given_context': float(
            len(common_records) / float(len(context_supported_records))
        ) if context_supported_records else 0.0,
        'eligibility_loss': {
            'no_hc_context_compounds': len(no_context_compounds),
            'no_hc_context_records': lost_records(no_context_compounds),
            'no_exact_degree_donor_compounds': len(
                no_exact_degree_compounds
            ),
            'no_exact_degree_donor_records': lost_records(
                no_exact_degree_compounds
            ),
            'no_disjoint_exact_degree_donor_compounds': len(
                no_disjoint_compounds
            ),
            'no_disjoint_exact_degree_donor_records': lost_records(
                no_disjoint_compounds
            ),
        },
    }
    metadata = {
        'evaluation_type': 'chcr_donor_control_pure_inference',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': protocol['fold_count'],
        'selection_split': 'strict_inner_validation',
        'optimizer_steps': 0,
        'draws': args.draws,
        'counterfactual_seed': args.counterfactual_seed,
        'matching_scope': (
            'validation_sources' if args.dry_run else 'all_hc_sources'
        ),
        'protocol': protocol_audit(protocol),
        'checkpoint': checkpoint_audit(checkpoint, checkpoint_files),
        'hc_path': str(hc_path),
        'hc_sha256': sha256_file(hc_path),
        'degree_control_coverage': degree_control_coverage,
        'common_coverage': common_coverage,
        'matching': {
            strategy: {
                'eligible_compounds': matchings[strategy]['eligible_compounds'],
                'requested_compounds': matchings[strategy]['requested_compounds'],
                'assignment_audit': assignment_audits[strategy],
            }
            for strategy, _ in STRATEGIES
        },
    }
    print('CHCR donor-control audit')
    print('  fold: %d/%d' % (args.fold, protocol['fold_count']))
    print('  split: Strict inner-validation only')
    print('  checkpoint: %s' % checkpoint)
    print('  common record coverage: %d/%d' % (
        common_coverage['records'], common_coverage['requested_records']
    ))
    print('  primary degree-control coverage: %d/%d' % (
        degree_control_coverage['records'],
        degree_control_coverage['requested_records'],
    ))
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        if not degree_control_coverage['passed']:
            print(
                'Primary degree-control coverage failed: %.2f%% < %.2f%%.' % (
                    100.0 * degree_control_coverage['fraction'],
                    100.0 * args.minimum_common_coverage,
                )
            )
            return 3
        return 0
    if not common_records:
        raise ValueError('No validation records share all three donor strategies.')

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(conf)
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI

    snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, protocol, checkpoint, args.fold
    )
    expected_terms = {
        'compound_disease': False,
        'herb_protein': True,
        'herb_disease': False,
    }
    if snapshot['context_terms'] != expected_terms:
        raise ValueError('The audit requires frozen static Hctx-P terms.')
    mode = (
        str(conf['context.herb_protein.mode']).strip().lower()
        if conf.contains('context.herb_protein.mode') else 'static'
    )
    if mode != 'static':
        raise ValueError('The audit requires context.herb_protein.mode=static.')
    if 'context_herb_protein' not in snapshot['weights']:
        raise ValueError('Checkpoint does not contain context_herb_protein.')
    metadata['herb_context_weight'] = weight_audit(
        snapshot['weights']['context_herb_protein']
    )

    records = [
        row for row in common_records
        if str(row[0]) in snapshot['compound_map']
    ]
    if len(records) != len(common_records):
        raise ValueError('Some common donor records are absent from the checkpoint map.')
    base_logits, factual_logits = score_snapshot(
        snapshot, records, include_context=True
    )
    compound_ids = [str(row[0]) for row in records]
    protein_ids = [str(row[1]) for row in records]
    labels = np.asarray(
        [int(float(row[2]) > 0) for row in records], dtype=np.int32
    )
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

    primary_degree_records = [
        row for row in degree_control_records
        if str(row[0]) in snapshot['compound_map']
    ]
    if len(primary_degree_records) != len(degree_control_records):
        raise ValueError(
            'Some primary degree-control records are absent from the checkpoint map.'
        )
    _, primary_factual_logits = score_snapshot(
        snapshot, primary_degree_records, include_context=True
    )
    primary_compound_ids = [str(row[0]) for row in primary_degree_records]
    primary_labels = np.asarray([
        int(float(row[2]) > 0) for row in primary_degree_records
    ], dtype=np.int32)
    primary_hc_degrees = [
        len(memberships[compound_id]) for compound_id in primary_compound_ids
    ]
    primary_train_cp_degrees = [
        train_cp_degrees.get(compound_id, 0)
        for compound_id in primary_compound_ids
    ]
    primary_subgroups = {
        'H-C degree': [
            herb_degree_bin(value) for value in primary_hc_degrees
        ],
        'training C-P degree': [
            training_cp_degree_bin(value)
            for value in primary_train_cp_degrees
        ],
    }
    primary_degree_logits = []
    degree_assignments = matchings['exact_degree']['assignments']
    for draw_index in range(args.draws):
        donor_ids = [
            degree_assignments[compound_id][draw_index]
            for compound_id in primary_compound_ids
        ]
        missing = [
            donor_id for donor_id in donor_ids
            if donor_id not in snapshot['compound_map']
        ]
        if missing:
            raise ValueError(
                'Primary degree-control donors are absent from checkpoint map: %s' %
                ', '.join(sorted(set(missing))[:5])
            )
        donor_indices = [
            snapshot['compound_map'][donor_id] for donor_id in donor_ids
        ]
        primary_degree_logits.append(score_with_counterfactual_contexts(
            snapshot, primary_degree_records, donor_indices
        ))
    primary_degree_analysis = summarize_counterfactual_audit(
        primary_labels,
        primary_compound_ids,
        primary_factual_logits,
        np.asarray(primary_degree_logits, dtype=np.float64),
        subgroup_values=primary_subgroups,
        requested_records=len(protocol['validation']),
        minimum_group_pairs=args.minimum_group_pairs,
    )

    analyses = {}
    donor_ids_by_strategy = {}
    for strategy, _ in STRATEGIES:
        assignments = matchings[strategy]['assignments']
        strategy_logits = []
        strategy_donors = []
        for draw_index in range(args.draws):
            donor_ids = [
                assignments[compound_id][draw_index]
                for compound_id in compound_ids
            ]
            missing = [
                donor_id for donor_id in donor_ids
                if donor_id not in snapshot['compound_map']
            ]
            if missing:
                raise ValueError(
                    'Donor compounds are absent from checkpoint map: %s' %
                    ', '.join(sorted(set(missing))[:5])
                )
            donor_indices = [
                snapshot['compound_map'][donor_id] for donor_id in donor_ids
            ]
            strategy_logits.append(score_with_counterfactual_contexts(
                snapshot, records, donor_indices
            ))
            strategy_donors.append(donor_ids)
        analysis = summarize_counterfactual_audit(
            labels,
            compound_ids,
            factual_logits,
            np.asarray(strategy_logits, dtype=np.float64),
            subgroup_values=subgroup_values,
            requested_records=len(protocol['validation']),
            minimum_group_pairs=args.minimum_group_pairs,
        )
        analyses[strategy] = analysis
        donor_ids_by_strategy[strategy] = strategy_donors

    strategy_rows = []
    common_assignment_audits = {}
    for strategy, label in STRATEGIES:
        common_assignments = {
            compound_id: matchings[strategy]['assignments'][compound_id]
            for compound_id in common_compounds
        }
        common_assignment_audit = summarize_donor_assignments(
            common_assignments, memberships
        )
        common_assignment_audits[strategy] = common_assignment_audit
        strategy_rows.append(strategy_row(
            strategy,
            label,
            matchings[strategy],
            common_assignment_audit,
            analyses[strategy],
        ))
    primary_degree_assignments = {
        compound_id: matchings['exact_degree']['assignments'][compound_id]
        for compound_id in degree_control_compounds
    }
    primary_degree_assignment_audit = summarize_donor_assignments(
        primary_degree_assignments, memberships
    )
    primary_degree_row = strategy_row(
        'exact_degree_primary',
        'Degree-matched primary',
        matchings['exact_degree'],
        primary_degree_assignment_audit,
        primary_degree_analysis,
    )
    decision = comparison_decision(
        analyses,
        primary_degree_analysis=primary_degree_analysis,
        degree_control_overlap_fraction=common_assignment_audits[
            'exact_degree'
        ]['overlap_fraction'],
        minimum_common_coverage=args.minimum_common_coverage,
    )

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'chcr_donor_controls' / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_analyses = {}
    draw_rows = []
    subgroup_rows = []
    mean_margins = {}
    mean_counterfactual_logits = {}
    for strategy, label in STRATEGIES:
        analysis = dict(analyses[strategy])
        mean_margins[strategy] = analysis.pop('mean_margins')
        mean_counterfactual_logits[strategy] = analysis.pop(
            'mean_counterfactual_logits'
        )
        serializable_analyses[strategy] = analysis
        for row in analysis['draws']:
            draw_rows.append({'strategy': strategy, 'label': label, **row})
        for row in flatten_subgroups(analysis['subgroups']):
            subgroup_rows.append({'strategy': strategy, 'label': label, **row})
    payload = {
        'metadata': metadata,
        'primary_degree_control': {
            'metrics': primary_degree_row,
            'analysis': None,
        },
        'strategy_metrics': strategy_rows,
        'analyses': serializable_analyses,
        'comparison': decision,
    }
    serializable_primary_degree = dict(primary_degree_analysis)
    serializable_primary_degree.pop('mean_margins')
    serializable_primary_degree.pop('mean_counterfactual_logits')
    payload['primary_degree_control']['analysis'] = (
        serializable_primary_degree
    )
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'report.md').write_text(
        build_markdown(
            metadata, strategy_rows, decision, primary_degree_row
        ),
        encoding='utf-8',
    )
    write_tsv(output_dir / 'primary_degree_metrics.tsv', [primary_degree_row])
    write_tsv(output_dir / 'strategy_metrics.tsv', strategy_rows)
    write_tsv(output_dir / 'draw_metrics.tsv', draw_rows)
    write_tsv(output_dir / 'subgroup_metrics.tsv', subgroup_rows)

    factual_context_logits = factual_logits - base_logits
    pair_rows = []
    for index, (compound_id, protein_id, label) in enumerate(
            zip(compound_ids, protein_ids, labels)):
        row = {
            'compound_id': compound_id,
            'protein_id': protein_id,
            'label': int(label),
            'hc_degree': hc_degrees[index],
            'training_cp_degree': raw_train_cp_degrees[index],
            'base_logit': float(base_logits[index]),
            'factual_context_logit': float(factual_context_logits[index]),
            'factual_total_logit': float(factual_logits[index]),
        }
        for strategy, _ in STRATEGIES:
            donors = [
                donor_ids_by_strategy[strategy][draw][index]
                for draw in range(args.draws)
            ]
            mean_cf = mean_counterfactual_logits[strategy][index]
            row['%s_pool_size' % strategy] = matchings[strategy][
                'pool_sizes'
            ][compound_id]
            row['%s_donors' % strategy] = ','.join(donors)
            row['%s_mean_total_logit' % strategy] = float(mean_cf)
            row['%s_mean_context_logit' % strategy] = float(
                mean_cf - base_logits[index]
            )
            row['%s_margin' % strategy] = float(
                mean_margins[strategy][index]
            )
        pair_rows.append(row)
    write_tsv(output_dir / 'pair_margins.tsv', pair_rows)

    print('\nDecision: %s' % decision['decision'])
    for row in strategy_rows:
        print(
            '  %s: AUPR drop=%.6f positive_margin=%.6f win=%.6f' % (
                row['label'],
                row['AUPR_drop'],
                row['positive_mean_margin'],
                row['positive_pair_win_rate'],
            )
        )
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
