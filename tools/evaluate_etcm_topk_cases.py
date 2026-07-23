#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Restore frozen ETCM checkpoints and generate evidence-annotated '
            'Top-K case predictions without training.'
        )
    )
    parser.add_argument(
        '--selection-manifest',
        default='configs/etcm_topk_case_selection.json',
    )
    parser.add_argument(
        '--output-dir',
        default='results/etcm_topk_cases/fold1',
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def joined(values):
    return ';'.join(sorted({str(value) for value in values if value}))


def evidence_indexes(validation_root):
    from util.etcm_topk_cases import group_rows, read_tsv

    confirmed_pair_rows = read_tsv(
        validation_root / 'relations' / 'confirmed_target_pairs.tsv'
    )
    potential_pair_rows = read_tsv(
        validation_root / 'relations' / 'potential_target_pairs.tsv'
    )
    confirmed_evidence = read_tsv(
        validation_root / 'relations' / 'confirmed_target_evidence.tsv'
    )
    potential_evidence = read_tsv(
        validation_root / 'relations' / 'potential_target_evidence.tsv'
    )
    return {
        'confirmed_pairs': {
            (str(row['compound_id_processed']), str(row['protein_id_processed'])): row
            for row in confirmed_pair_rows
            if row.get('compound_id_processed') and row.get('protein_id_processed')
        },
        'potential_pairs': {
            (str(row['compound_id_processed']), str(row['protein_id_processed'])): row
            for row in potential_pair_rows
            if row.get('compound_id_processed') and row.get('protein_id_processed')
        },
        'confirmed_evidence': group_rows(
            [
                row for row in confirmed_evidence
                if row.get('compound_id_processed') and row.get('protein_id_processed')
            ],
            ('compound_id_processed', 'protein_id_processed'),
        ),
        'potential_evidence': group_rows(
            [
                row for row in potential_evidence
                if row.get('compound_id_processed') and row.get('protein_id_processed')
            ],
            ('compound_id_processed', 'protein_id_processed'),
        ),
    }


def annotate_prediction(row, indexes, protein_mapping):
    from util.etcm_topk_cases import evidence_classification

    pair = (str(row['compound_id']), str(row['protein_id']))
    level, status = evidence_classification(
        pair, indexes['confirmed_pairs'], indexes['potential_pairs']
    )
    confirmed_pair = indexes['confirmed_pairs'].get(pair, {})
    potential_pair = indexes['potential_pairs'].get(pair, {})
    confirmed_evidence = indexes['confirmed_evidence'].get(pair, [])
    potential_evidence = indexes['potential_evidence'].get(pair, [])
    mapping = protein_mapping.get(str(row['protein_id']), {})
    return dict(row, **{
        'gene_symbol': (
            confirmed_pair.get('gene_symbol')
            or potential_pair.get('gene_symbol')
            or mapping.get('gene_symbols', '')
        ),
        'target_name': mapping.get('target_names', ''),
        'uniprot_accessions': mapping.get('uniprot_accessions', ''),
        'evidence_level': level,
        'evidence_status': status,
        'confirmed_evidence_count': confirmed_pair.get('evidence_count', ''),
        'confirmed_activities': joined(
            evidence.get('activity') for evidence in confirmed_evidence
        ),
        'confirmed_references': joined(
            url
            for evidence in confirmed_evidence
            for url in evidence.get('reference_urls', '').split('|')
        ),
        'potential_similar_score': potential_pair.get('similar_score_mean', ''),
        'potential_activities': joined(
            evidence.get('activity') for evidence in potential_evidence
        ),
        'potential_references': joined(
            url
            for evidence in potential_evidence
            for url in evidence.get('reference_urls', '').split('|')
        ),
    })


def build_herb_context_rows(selected_cases, dataset_path, validation_root):
    from util.etcm_topk_cases import (
        normalize_name,
        read_csv_index,
        read_tsv,
        relation_pairs,
    )

    herb_mapping = read_csv_index(
        dataset_path / 'mappings' / 'herb_id_map.csv', 'herb_id'
    )
    hc_pairs = relation_pairs(dataset_path / 'H_C.txt')
    model_herbs = defaultdict(set)
    for herb_id, compound_id in hc_pairs:
        model_herbs[compound_id].add(herb_id)

    external_herbs = defaultdict(list)
    for row in read_tsv(validation_root / 'relations' / 'ingredient_herb.tsv'):
        external_herbs[str(row['tcmip_id'])].append(row)

    output = []
    for case in selected_cases:
        compound_id = str(case['compound_id'])
        tcmip_ids = str(case['all_tcmip_ids']).split(';')
        ingredient_rows = [
            row
            for tcmip_id in tcmip_ids
            for row in external_herbs.get(tcmip_id, [])
        ]
        model_rows = [
            herb_mapping[herb_id]
            for herb_id in sorted(model_herbs.get(compound_id, set()))
            if herb_id in herb_mapping
        ]
        ingredient_pinyin = {
            normalize_name(row.get('herb_name_pinyin', ''))
            for row in ingredient_rows
            if normalize_name(row.get('herb_name_pinyin', ''))
        }
        model_pinyin = {
            normalize_name(row.get('herb_name_pinyin', ''))
            for row in model_rows
            if normalize_name(row.get('herb_name_pinyin', ''))
        }
        overlap = ingredient_pinyin & model_pinyin
        output.append({
            'compound_id': compound_id,
            'tcmip_id': case['tcmip_id'],
            'compound_name': case['compound_name'],
            'support_stratum': case['support_stratum'],
            'model_train_cp_degree': case['model_train_cp_degree'],
            'ingredient_herb_count': len({
                (
                    row.get('herb_name_pinyin', ''),
                    row.get('herb_name_latin', ''),
                )
                for row in ingredient_rows
            }),
            'model_hc_herb_count': len(model_rows),
            'pinyin_overlap_count': len(overlap),
            'ingredient_herbs': joined(
                '%s|%s' % (
                    row.get('herb_name_pinyin', ''),
                    row.get('herb_name_latin', ''),
                )
                for row in ingredient_rows
            ),
            'model_hc_herbs': joined(
                '%s|%s' % (
                    row.get('herb_name_pinyin', ''),
                    row.get('herb_name_latin', ''),
                )
                for row in model_rows
            ),
            'overlap_pinyin_normalized': joined(overlap),
        })
    return output


def case_metrics(selected_cases, method_rankings, confirmed_targets, top_k):
    rows = []
    for method, compound_rankings in method_rankings.items():
        for case in selected_cases:
            compound_id = str(case['compound_id'])
            targets = sorted(confirmed_targets.get(compound_id, set()))
            rank_map = {
                row['protein_id']: row for row in compound_rankings[compound_id]
            }
            ranks = [rank_map[target]['rank'] for target in targets]
            rows.append({
                'method': method,
                'compound_id': compound_id,
                'compound_name': case['compound_name'],
                'support_stratum': case['support_stratum'],
                'confirmed_targets': len(targets),
                'first_confirmed_rank': min(ranks),
                'MRR': 1.0 / min(ranks),
                'Recall@10': sum(rank <= 10 for rank in ranks) / float(len(ranks)),
                'Recall@%d' % top_k: (
                    sum(rank <= top_k for rank in ranks) / float(len(ranks))
                ),
            })
    return rows


def aggregate_case_metrics(metrics, top_k):
    grouped = defaultdict(list)
    for row in metrics:
        grouped[row['method']].append(row)
    output = []
    for method, rows in grouped.items():
        output.append({
            'method': method,
            'cases': len(rows),
            'confirmed_targets': sum(
                int(row['confirmed_targets']) for row in rows
            ),
            'macro_MRR': float(np.mean([row['MRR'] for row in rows])),
            'macro_Recall@10': float(np.mean([
                row['Recall@10'] for row in rows
            ])),
            'macro_Recall@%d' % top_k: float(np.mean([
                row['Recall@%d' % top_k] for row in rows
            ])),
            'confirmed_hits@10': sum(
                round(row['Recall@10'] * row['confirmed_targets'])
                for row in rows
            ),
            'confirmed_hits@%d' % top_k: sum(
                round(
                    row['Recall@%d' % top_k] * row['confirmed_targets']
                )
                for row in rows
            ),
        })
    return output


def build_markdown(selection, metrics, aggregate_metrics, top_k):
    lines = [
        '# ETCM2.0 Top-K 案例解释',
        '',
        '- 数据集：`%s`' % selection['dataset'],
        '- Strict fold：`%d`' % selection['fold'],
        '- 案例数：`%d`' % selection['case_count'],
        '- 候选范围：mention10 全部 protein，过滤完整 `C_P.txt` 已知关系。',
        '- 运行方式：冻结 checkpoint 纯推理，optimizer steps = 0。',
        '- ingredient Confirmed/Potential 证据不参与训练或模型选择。',
        '',
        '## 冻结案例',
        '',
        '| 成分 | 支持层 | 训练 C-P degree | 未见确认靶点 |',
        '|---|---|---:|---:|',
    ]
    for case in selection['selected_cases']:
        lines.append('| %s | %s | %s | %s |' % (
            case['compound_name'],
            case['support_stratum'],
            case['model_train_cp_degree'],
            case['unseen_confirmed_targets'],
        ))
    lines.extend([
        '',
        '## 外部确认关系排名',
        '',
        '| 模型 | 成分 | First rank | MRR | Recall@10 | Recall@%d |' % top_k,
        '|---|---|---:|---:|---:|---:|',
    ])
    for row in metrics:
        lines.append('| %s | %s | %d | %.6f | %.6f | %.6f |' % (
            row['method'],
            row['compound_name'],
            row['first_confirmed_rank'],
            row['MRR'],
            row['Recall@10'],
            row['Recall@%d' % top_k],
        ))
    lines.extend([
        '',
        '## 五个案例汇总',
        '',
        '| 模型 | Confirmed hits@10 | Confirmed hits@%d | Macro MRR | Macro Recall@10 | Macro Recall@%d |' % (
            top_k, top_k),
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in aggregate_metrics:
        lines.append('| %s | %d/%d | %d/%d | %.6f | %.6f | %.6f |' % (
            row['method'],
            row['confirmed_hits@10'],
            row['confirmed_targets'],
            row['confirmed_hits@%d' % top_k],
            row['confirmed_targets'],
            row['macro_MRR'],
            row['macro_Recall@10'],
            row['macro_Recall@%d' % top_k],
        ))
    lines.extend([
        '',
        '该汇总只描述 5 个在查看模型分数前冻结的证据丰富案例，不是全量外部验证，',
        '也不用于重新选择模型或超参数。',
        '',
        '## 解释边界',
        '',
        '- A 级为训练数据中未见且带直接 Activity/Reference 的 ETCM Confirmed Target。',
        '- C 级 Potential Target 只表示与 ETCM 推断网络一致，不能作为独立真值。',
        '- E 级表示当前证据库未覆盖，不等同于确认负例。',
        '- Rank gain 为 `Strict rank - 当前模型 rank`，正值表示相对 Strict 前移。',
        '- 本结果只用于冻结案例解释，不反向选择 checkpoint 或修改模型。',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    from tools.analyze_context_subgroups import (
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
        restore_snapshot,
        score_snapshot,
    )
    from util.etcm_topk_cases import (
        rank_candidates,
        read_csv_index,
        read_tsv,
        relation_pairs,
        sha256_file,
        write_json,
        write_tsv,
    )

    selection_path = resolve_path(args.selection_manifest)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('selection_status') != 'frozen_before_checkpoint_scoring':
        raise ValueError('Case selection was not frozen before checkpoint scoring.')
    if selection.get('model_scores_read'):
        raise ValueError('Selection manifest reports that model scores were read.')

    fold = int(selection['fold'])
    top_k = int(json.loads(
        Path(selection['inputs']['checkpoint_manifest']).read_text(
            encoding='utf-8'
        )
    )['top_k'])
    selected_cases = selection['selected_cases']
    reference_audit = selection['protocols'][0]['protocol']
    dataset_path = Path(reference_audit['datapath']).parent
    evidence_manifest_path = Path(
        selection['inputs']['external_evidence_manifest']
    )
    if sha256_file(evidence_manifest_path) != (
            selection['inputs']['external_evidence_manifest_sha256']):
        raise ValueError('External evidence manifest changed after case selection.')
    validation_root = evidence_manifest_path.parent

    protocols = []
    for checkpoint_spec in selection['checkpoints']:
        config_path = Path(checkpoint_spec['config'])
        if sha256_file(config_path) != checkpoint_spec['config_sha256']:
            raise ValueError('Config changed after case selection: %s' % config_path)
        protocol = prepare_protocol(config_path, fold)
        audit = protocol_audit(protocol)
        for field in (
                'datapath', 'strict_assignments_sha256', 'outer_train_sha256',
                'outer_test_sha256', 'model_train_sha256', 'validation_sha256'):
            if audit[field] != reference_audit[field]:
                raise ValueError(
                    '%s protocol mismatch for %s.' %
                    (field, checkpoint_spec['method'])
                )
        checkpoint, data_files = normalize_checkpoint(
            checkpoint_spec['checkpoint']
        )
        if sha256_file(str(checkpoint) + '.index') != (
                checkpoint_spec['checkpoint_index_sha256']):
            raise ValueError('Checkpoint index changed: %s' % checkpoint)
        if len(data_files) != 1 or sha256_file(data_files[0]) != (
                checkpoint_spec['checkpoint_data_sha256']):
            raise ValueError('Checkpoint data changed: %s' % checkpoint)
        protocols.append((checkpoint_spec, protocol, checkpoint))

    known_pairs = relation_pairs(dataset_path / 'C_P.txt')
    protein_mapping = read_csv_index(
        dataset_path / 'mappings' / 'protein_id_map.csv', 'protein_id'
    )
    evidence = evidence_indexes(validation_root)
    confirmed_targets = defaultdict(set)
    for row in read_tsv(
            validation_root / 'validation' / selection['dataset'] /
            'unseen_confirmed.tsv'):
        confirmed_targets[str(row['compound_id'])].add(str(row['protein_id']))

    dry_run_report = {
        'selection_manifest': str(selection_path),
        'selection_manifest_sha256': sha256_file(selection_path),
        'selected_compounds': len(selected_cases),
        'known_pairs_filtered': len(known_pairs),
        'protein_candidates': len(protein_mapping),
        'models': [
            {
                'method': spec['method'],
                'config': str(protocol['config_path']),
                'checkpoint': str(checkpoint),
            }
            for spec, protocol, checkpoint in protocols
        ],
        'optimizer_steps': 0,
        'checkpoint_updates': 0,
    }
    print('ETCM Top-K case evaluation')
    print(json.dumps(dry_run_report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(protocols[0][1]['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI

    method_rankings = {}
    checkpoint_audits = []
    for checkpoint_spec, protocol, checkpoint in protocols:
        snapshot = restore_snapshot(
            tf, HDCTI, set_global_seed, protocol, checkpoint, fold
        )
        method = checkpoint_spec['variant']
        protein_ids = [
            protein_id for protein_id, _ in sorted(
                snapshot['protein_map'].items(), key=lambda item: item[1]
            )
        ]
        compound_rankings = {}
        for case in selected_cases:
            compound_id = str(case['compound_id'])
            candidates = [
                protein_id for protein_id in protein_ids
                if (compound_id, protein_id) not in known_pairs
            ]
            records = [
                [compound_id, protein_id, 0.0] for protein_id in candidates
            ]
            _, logits = score_snapshot(snapshot, records, include_context=True)
            compound_rankings[compound_id] = rank_candidates(candidates, logits)
        method_rankings[method] = compound_rankings
        checkpoint_audits.append({
            'method': checkpoint_spec['method'],
            'variant': method,
            'checkpoint': str(checkpoint),
            'checkpoint_index_sha256': checkpoint_spec[
                'checkpoint_index_sha256'
            ],
            'optimizer_steps': 0,
        })
        print('  %s checkpoint restored and scored.' % method)

    strict_method = 'NoContext'
    strict_ranks = {
        compound_id: {
            row['protein_id']: row['rank'] for row in rows
        }
        for compound_id, rows in method_rankings[strict_method].items()
    }
    case_index = {
        str(case['compound_id']): case for case in selected_cases
    }
    prediction_rows = []
    annotated_rows = []
    for method, compounds in method_rankings.items():
        for compound_id, ranked in compounds.items():
            for row in ranked[:top_k]:
                output = {
                    'method': method,
                    'compound_id': compound_id,
                    'tcmip_id': case_index[compound_id]['tcmip_id'],
                    'compound_name': case_index[compound_id]['compound_name'],
                    'support_stratum': case_index[compound_id]['support_stratum'],
                    'model_train_cp_degree': case_index[compound_id][
                        'model_train_cp_degree'
                    ],
                    'rank': row['rank'],
                    'protein_id': row['protein_id'],
                    'logit': '%.10f' % row['logit'],
                    'score': '%.10f' % row['score'],
                    'strict_rank': strict_ranks[compound_id][row['protein_id']],
                    'rank_gain_vs_strict': (
                        strict_ranks[compound_id][row['protein_id']] - row['rank']
                    ),
                }
                prediction_rows.append(output)
                annotated_rows.append(
                    annotate_prediction(output, evidence, protein_mapping)
                )

    confirmed_rank_rows = []
    for method, compounds in method_rankings.items():
        for case in selected_cases:
            compound_id = str(case['compound_id'])
            rank_map = {
                row['protein_id']: row for row in compounds[compound_id]
            }
            for protein_id in sorted(confirmed_targets[compound_id]):
                ranked = rank_map[protein_id]
                pair = (compound_id, protein_id)
                pair_row = evidence['confirmed_pairs'].get(pair, {})
                confirmed_rank_rows.append({
                    'method': method,
                    'compound_id': compound_id,
                    'tcmip_id': case['tcmip_id'],
                    'compound_name': case['compound_name'],
                    'support_stratum': case['support_stratum'],
                    'protein_id': protein_id,
                    'gene_symbol': pair_row.get(
                        'gene_symbol',
                        protein_mapping.get(protein_id, {}).get(
                            'gene_symbols', ''
                        ),
                    ),
                    'rank': ranked['rank'],
                    'score': '%.10f' % ranked['score'],
                    'hit@10': int(ranked['rank'] <= 10),
                    'hit@%d' % top_k: int(ranked['rank'] <= top_k),
                })

    metrics = case_metrics(
        selected_cases, method_rankings, confirmed_targets, top_k
    )
    aggregate_metrics = aggregate_case_metrics(metrics, top_k)
    herb_rows = build_herb_context_rows(
        selected_cases, dataset_path, validation_root
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = [
        'method', 'compound_id', 'tcmip_id', 'compound_name',
        'support_stratum', 'model_train_cp_degree', 'rank', 'protein_id',
        'logit', 'score', 'strict_rank', 'rank_gain_vs_strict',
    ]
    annotation_fields = prediction_fields + [
        'gene_symbol', 'target_name', 'uniprot_accessions',
        'evidence_level', 'evidence_status', 'confirmed_evidence_count',
        'confirmed_activities', 'confirmed_references',
        'potential_similar_score', 'potential_activities',
        'potential_references',
    ]
    write_tsv(
        output_dir / 'topk_predictions.tsv',
        prediction_rows,
        prediction_fields,
    )
    write_tsv(
        output_dir / 'evidence_annotated_topk.tsv',
        annotated_rows,
        annotation_fields,
    )
    write_tsv(
        output_dir / 'confirmed_target_ranks.tsv',
        confirmed_rank_rows,
        [
            'method', 'compound_id', 'tcmip_id', 'compound_name',
            'support_stratum', 'protein_id', 'gene_symbol', 'rank', 'score',
            'hit@10', 'hit@%d' % top_k,
        ],
    )
    write_tsv(
        output_dir / 'herb_context_explanations.tsv',
        herb_rows,
        [
            'compound_id', 'tcmip_id', 'compound_name', 'support_stratum',
            'model_train_cp_degree', 'ingredient_herb_count',
            'model_hc_herb_count', 'pinyin_overlap_count',
            'ingredient_herbs', 'model_hc_herbs',
            'overlap_pinyin_normalized',
        ],
    )
    metric_fields = [
        'method', 'compound_id', 'compound_name', 'support_stratum',
        'confirmed_targets', 'first_confirmed_rank', 'MRR', 'Recall@10',
        'Recall@%d' % top_k,
    ]
    write_tsv(output_dir / 'case_metrics.tsv', metrics, metric_fields)
    aggregate_fields = [
        'method', 'cases', 'confirmed_targets', 'macro_MRR',
        'macro_Recall@10', 'macro_Recall@%d' % top_k,
        'confirmed_hits@10', 'confirmed_hits@%d' % top_k,
    ]
    write_tsv(
        output_dir / 'aggregate_case_metrics.tsv',
        aggregate_metrics,
        aggregate_fields,
    )

    report = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'evaluation_type': 'frozen_checkpoint_etcm_topk_case_explanation',
        'training_steps': 0,
        'optimizer_steps': 0,
        'checkpoint_updates': 0,
        'selection_manifest': str(selection_path),
        'selection_manifest_sha256': sha256_file(selection_path),
        'candidate_protocol': {
            'protein_universe': 'all_model_proteins',
            'protein_count': len(protein_mapping),
            'filter': 'all_known_ETCM2.0_core_mention10_C_P_pairs',
            'known_pairs_filtered': len(known_pairs),
            'top_k': top_k,
            'tie_break': 'protein_id_ascending',
            'unrecorded_pair_status': 'unlabeled',
        },
        'checkpoints': checkpoint_audits,
        'case_metrics': metrics,
        'aggregate_case_metrics': aggregate_metrics,
        'evidence_boundary': {
            'confirmed': 'external_confirmed_relation_with_activity_and_reference',
            'potential': 'ETCM_similarity_transfer_consistency_only',
            'no_evidence': 'unlabeled_not_verified_negative',
            'training_use_prohibited': True,
        },
    }
    write_json(output_dir / 'report.json', report)
    write_json(output_dir / 'checkpoint_manifest.json', {
        'selection_manifest_sha256': sha256_file(selection_path),
        'checkpoints': checkpoint_audits,
    })
    (output_dir / 'case_summary.md').write_text(
        build_markdown(selection, metrics, aggregate_metrics, top_k) + '\n',
        encoding='utf-8',
    )
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
