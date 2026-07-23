#!/usr/bin/env python3
"""Add raw ETCM Herb/Target/Disease context to frozen Top-K predictions.

This is a post-hoc annotation tool. It never restores a model, changes a
checkpoint, trains parameters, selects cases, or changes candidate ranks.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--selection-manifest',
        default='configs/etcm_topk_case_selection.json',
    )
    parser.add_argument(
        '--ranked-input',
        default='results/etcm_topk_cases/fold1/evidence_annotated_topk.tsv',
    )
    parser.add_argument(
        '--processed-dataset',
        default='dataset/ETCM2.0_processed',
    )
    parser.add_argument(
        '--model-dataset',
        default='dataset/ETCM2.0_core_mention10',
    )
    parser.add_argument('--raw-root', default='dataset/ETCM2.0')
    parser.add_argument(
        '--output-dir',
        default='results/etcm_topk_cases/fold1/context',
    )
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def relative_source(path):
    if path is None:
        return ''
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def joined(values):
    return ';'.join(sorted({str(value) for value in values if value}))


def entity_page(mapping, page_index, name_fields):
    from util.etcm_topk_cases import resolve_raw_page, split_values

    names = []
    for field in name_fields:
        names.extend(split_values(mapping.get(field, '')))
    return resolve_raw_page(page_index, names)


def build_summary(manifest, cases):
    lines = [
        '# ETCM2.0 Top-K 原始页面上下文注释',
        '',
        '该结果是在 Top-K 排名冻结后生成的 post-hoc 解释层。原始 Herbs、',
        'Targets 和 Diseases 页面没有参与训练、checkpoint 选择、案例选择或候选',
        '排序。',
        '',
        '## 覆盖统计',
        '',
        '| 指标 | 数值 |',
        '|---|---:|',
        '| 冻结 Top-K 记录 | %d |' % manifest['counts']['ranked_rows'],
        '| 唯一成分-蛋白候选 | %d |' % manifest['counts']['unique_pairs'],
        '| 存在 C-H-D-P 页面路径的候选 | %d |' % (
            manifest['counts']['path_supported_pairs']
        ),
        '| 唯一 C-H-D-P 路径 | %d |' % manifest['counts']['unique_paths'],
        '| Herb 原始页面覆盖 | %d/%d |' % (
            manifest['counts']['herb_raw_pages'],
            manifest['counts']['herb_entities'],
        ),
        '| Target 原始页面覆盖 | %d/%d |' % (
            manifest['counts']['target_raw_pages'],
            manifest['counts']['target_entities'],
        ),
        '| Disease 原始页面覆盖 | %d/%d |' % (
            manifest['counts']['disease_raw_pages'],
            manifest['counts']['disease_entities'],
        ),
        '',
        '## 五个冻结案例',
        '',
        '| 成分 | Herb 数 | 页面路径支持的 Top-K 候选 | 路径数 |',
        '|---|---:|---:|---:|',
    ]
    for row in cases:
        lines.append('| %s | %d | %d | %d |' % (
            row['compound_name'],
            row['herb_count'],
            row['path_supported_pairs'],
            row['path_count'],
        ))
    lines.extend([
        '',
        '## 证据边界',
        '',
        '- `H-C` 来自 Herb 页的 ingredient/network 关系，也是模型使用的固定侧信息；',
        '  它可解释药材来源，但不是独立外部验证。',
        '- `P-D` 来自 Disease 页的 target 关系，也是模型使用的固定侧信息；',
        '  它可解释疾病语义，但不是独立外部验证。',
        '- `H-D` 来自 Disease 页的 herb 关系。既有审计显示它与完整',
        '  `H-C-P-D` 网络高度耦合，因此这里只标为算法/数据库关联路径。',
        '- Target 原始页用于核对 Gene Symbol、UniProt 和 Target Name 身份，',
        '  不把页面存在本身解释为候选 C-P 关系成立。',
        '- Ingredient Confirmed Target 中带 Activity/Reference 的 A 级证据仍是',
        '  当前 Top-K 外部验证的主要直接证据。',
        '',
    ])
    return '\n'.join(lines)


def main():
    from util.etcm_topk_cases import (
        pair_context_paths,
        raw_page_index,
        read_csv_index,
        read_tsv,
        relation_adjacency,
        relation_pairs,
        sha256_file,
        split_values,
        write_json,
        write_tsv,
    )

    args = parse_args()
    selection_path = resolve_path(args.selection_manifest)
    ranked_path = resolve_path(args.ranked_input)
    processed = resolve_path(args.processed_dataset)
    model_dataset = resolve_path(args.model_dataset)
    raw_root = resolve_path(args.raw_root)
    output_dir = resolve_path(args.output_dir)

    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('selection_status') != 'frozen_before_checkpoint_scoring':
        raise ValueError('Selection manifest is not frozen.')
    if selection.get('model_scores_read'):
        raise ValueError('Selection manifest was created after reading scores.')

    ranked_rows = read_tsv(ranked_path)
    if not ranked_rows:
        raise ValueError('Ranked Top-K input is empty.')

    herbs = read_csv_index(
        processed / 'mappings' / 'herb_id_map.csv', 'herb_id'
    )
    proteins = read_csv_index(
        processed / 'mappings' / 'protein_id_map.csv', 'protein_id'
    )
    diseases = read_csv_index(
        processed / 'mappings' / 'disease_id_map.csv', 'disease_id'
    )
    herbs_by_compound = relation_adjacency(
        processed / 'H_C.txt', reverse=True
    )
    diseases_by_herb = relation_adjacency(processed / 'H_D.txt')
    diseases_by_protein = relation_adjacency(processed / 'P_D.txt')

    model_hc = relation_pairs(model_dataset / 'H_C.txt')
    model_pd = relation_pairs(model_dataset / 'P_D.txt')
    raw_indexes = {
        'herb': raw_page_index(raw_root / 'etcm_herbs'),
        'target': raw_page_index(raw_root / 'etcm_targets'),
        'disease': raw_page_index(raw_root / 'etcm_diseases'),
    }

    pair_rows = defaultdict(list)
    for row in ranked_rows:
        pair_rows[(str(row['compound_id']), str(row['protein_id']))].append(row)

    pair_context = {}
    all_paths = set()
    herb_ids = set()
    protein_ids = set()
    disease_ids = set()
    for compound_id, protein_id in sorted(pair_rows):
        paths = pair_context_paths(
            compound_id,
            protein_id,
            herbs_by_compound,
            diseases_by_herb,
            diseases_by_protein,
        )
        pair_context[(compound_id, protein_id)] = paths
        protein_ids.add(protein_id)
        for herb_id in herbs_by_compound.get(compound_id, set()):
            herb_ids.add(herb_id)
        for herb_id, disease_id in paths:
            all_paths.add((compound_id, protein_id, herb_id, disease_id))
            disease_ids.add(disease_id)

    herb_pages = {
        herb_id: entity_page(
            herbs.get(herb_id, {}),
            raw_indexes['herb'],
            ('herb_key', 'herb_name_pinyin', 'herb_name_english'),
        )
        for herb_id in herb_ids
    }
    target_pages = {
        protein_id: (
            entity_page(
                proteins.get(protein_id, {}),
                raw_indexes['target'],
                ('target_names', 'proteins', 'protein_key'),
            )
            if 'target_page_base' in split_values(
                proteins.get(protein_id, {}).get('source_types', '')
            )
            else None
        )
        for protein_id in protein_ids
    }
    disease_pages = {
        disease_id: entity_page(
            diseases.get(disease_id, {}),
            raw_indexes['disease'],
            ('disease_key', 'disease_names'),
        )
        for disease_id in disease_ids
    }

    annotated = []
    for row in ranked_rows:
        compound_id = str(row['compound_id'])
        protein_id = str(row['protein_id'])
        paths = pair_context[(compound_id, protein_id)]
        path_herbs = {herb_id for herb_id, _ in paths}
        path_diseases = {disease_id for _, disease_id in paths}
        protein = proteins.get(protein_id, {})
        annotated.append(dict(row, **{
            'raw_target_page': relative_source(target_pages.get(protein_id)),
            'target_page_available': int(bool(target_pages.get(protein_id))),
            'target_page_identity_only': 1,
            'hc_herb_count': len(
                herbs_by_compound.get(compound_id, set())
            ),
            'pd_disease_count': len(
                diseases_by_protein.get(protein_id, set())
            ),
            'chdp_path_count': len(paths),
            'chdp_path_herb_count': len(path_herbs),
            'chdp_path_disease_count': len(path_diseases),
            'context_evidence_role': (
                'posthoc_algorithmic_path'
                if paths else 'posthoc_side_information'
            ),
            'context_is_independent_cp_validation': 0,
            'context_used_for_case_selection': 0,
            'context_used_for_candidate_ranking': 0,
            'target_source_types': protein.get('source_types', ''),
        }))

    path_rows = []
    for compound_id, protein_id, herb_id, disease_id in sorted(all_paths):
        predictions = pair_rows[(compound_id, protein_id)]
        case = predictions[0]
        herb = herbs.get(herb_id, {})
        protein = proteins.get(protein_id, {})
        disease = diseases.get(disease_id, {})
        path_rows.append({
            'compound_id': compound_id,
            'tcmip_id': case.get('tcmip_id', ''),
            'compound_name': case.get('compound_name', ''),
            'protein_id': protein_id,
            'gene_symbol': (
                case.get('gene_symbol') or protein.get('gene_symbols', '')
            ),
            'target_name': protein.get('target_names', ''),
            'herb_id': herb_id,
            'herb_name_pinyin': herb.get('herb_name_pinyin', ''),
            'herb_name_latin': herb.get('herb_name_latin', ''),
            'disease_id': disease_id,
            'disease_name': disease.get('disease_names', ''),
            'disease_global_categories': disease.get(
                'global_categories', ''
            ),
            'disease_anatomical_categories': disease.get(
                'anatomical_categories', ''
            ),
            'methods': joined(row['method'] for row in predictions),
            'method_ranks': joined(
                '%s:%s' % (row['method'], row['rank'])
                for row in predictions
            ),
            'herb_raw_page': relative_source(herb_pages.get(herb_id)),
            'target_raw_page': relative_source(target_pages.get(protein_id)),
            'disease_raw_page': relative_source(
                disease_pages.get(disease_id)
            ),
            'hc_page_relation': 'herb_page_ingredient_or_network',
            'pd_page_relation': 'disease_page_target',
            'hd_page_relation': 'disease_page_herb',
            'hc_used_as_model_side_information': int(
                (herb_id, compound_id) in model_hc
            ),
            'pd_used_as_model_side_information': int(
                (protein_id, disease_id) in model_pd
            ),
            'hd_used_by_current_model': 0,
            'path_is_independent_cp_validation': 0,
            'path_evidence_role': 'posthoc_algorithmic_mechanism_hypothesis',
        })

    herb_entity_rows = []
    for herb_id in sorted(herb_ids):
        row = herbs.get(herb_id, {})
        herb_entity_rows.append({
            'herb_id': herb_id,
            'herb_key': row.get('herb_key', ''),
            'herb_name_pinyin': row.get('herb_name_pinyin', ''),
            'herb_name_latin': row.get('herb_name_latin', ''),
            'herb_name_english': row.get('herb_name_english', ''),
            'property': row.get('property', ''),
            'flavor': row.get('flavor', ''),
            'meridian_tropism': row.get('meridian_tropism', ''),
            'source_types': row.get('source_types', ''),
            'raw_page': relative_source(herb_pages.get(herb_id)),
        })
    target_entity_rows = []
    for protein_id in sorted(protein_ids):
        row = proteins.get(protein_id, {})
        target_entity_rows.append({
            'protein_id': protein_id,
            'protein_key': row.get('protein_key', ''),
            'gene_symbols': row.get('gene_symbols', ''),
            'uniprot_accessions': row.get('uniprot_accessions', ''),
            'target_names': row.get('target_names', ''),
            'proteins': row.get('proteins', ''),
            'organisms': row.get('organisms', ''),
            'source_types': row.get('source_types', ''),
            'raw_page': relative_source(target_pages.get(protein_id)),
        })
    disease_entity_rows = []
    for disease_id in sorted(disease_ids):
        row = diseases.get(disease_id, {})
        disease_entity_rows.append({
            'disease_id': disease_id,
            'disease_key': row.get('disease_key', ''),
            'disease_names': row.get('disease_names', ''),
            'global_categories': row.get('global_categories', ''),
            'anatomical_categories': row.get('anatomical_categories', ''),
            'source_types': row.get('source_types', ''),
            'raw_page': relative_source(disease_pages.get(disease_id)),
        })

    selected_case_index = {
        str(row['compound_id']): row for row in selection['selected_cases']
    }
    case_rows = []
    for compound_id, case in selected_case_index.items():
        case_pairs = [
            key for key in pair_context if key[0] == compound_id
        ]
        case_rows.append({
            'compound_id': compound_id,
            'compound_name': case['compound_name'],
            'herb_count': len(herbs_by_compound.get(compound_id, set())),
            'topk_unique_pairs': len(case_pairs),
            'path_supported_pairs': sum(
                bool(pair_context[key]) for key in case_pairs
            ),
            'path_count': sum(
                len(pair_context[key]) for key in case_pairs
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / 'context_annotated_topk.tsv'
    paths_path = output_dir / 'mechanism_paths.tsv'
    herbs_path = output_dir / 'herb_entities.tsv'
    targets_path = output_dir / 'target_entities.tsv'
    diseases_path = output_dir / 'disease_entities.tsv'
    cases_path = output_dir / 'case_context_summary.tsv'
    summary_path = output_dir / 'summary.md'

    write_tsv(
        annotated_path,
        annotated,
        list(ranked_rows[0].keys()) + [
            'raw_target_page', 'target_page_available',
            'target_page_identity_only', 'hc_herb_count',
            'pd_disease_count', 'chdp_path_count',
            'chdp_path_herb_count', 'chdp_path_disease_count',
            'context_evidence_role', 'context_is_independent_cp_validation',
            'context_used_for_case_selection',
            'context_used_for_candidate_ranking', 'target_source_types',
        ],
    )
    write_tsv(paths_path, path_rows, list(path_rows[0].keys()) if path_rows else [])
    write_tsv(
        herbs_path,
        herb_entity_rows,
        list(herb_entity_rows[0].keys()) if herb_entity_rows else [],
    )
    write_tsv(
        targets_path,
        target_entity_rows,
        list(target_entity_rows[0].keys()) if target_entity_rows else [],
    )
    write_tsv(
        diseases_path,
        disease_entity_rows,
        list(disease_entity_rows[0].keys()) if disease_entity_rows else [],
    )
    write_tsv(cases_path, case_rows, list(case_rows[0].keys()))

    manifest = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'purpose': 'posthoc_topk_case_explanation_only',
        'restrictions': {
            'used_for_training': False,
            'used_for_checkpoint_selection': False,
            'used_for_case_selection': False,
            'used_for_candidate_ranking': False,
            'hd_is_independent_cp_validation': False,
        },
        'source_semantics': {
            'H_C': 'Herb page ingredient/network relation; model side information',
            'P_D': 'Disease page target relation; model side information',
            'H_D': (
                'Disease page herb relation; algorithm/database mapping '
                'coupled to the full H-C-P-D network'
            ),
            'Target': 'Target page identity metadata only',
        },
        'counts': {
            'ranked_rows': len(ranked_rows),
            'unique_pairs': len(pair_rows),
            'path_supported_pairs': sum(bool(value) for value in pair_context.values()),
            'unique_paths': len(all_paths),
            'herb_entities': len(herb_entity_rows),
            'herb_raw_pages': sum(bool(path) for path in herb_pages.values()),
            'target_entities': len(target_entity_rows),
            'target_raw_pages': sum(bool(path) for path in target_pages.values()),
            'disease_entities': len(disease_entity_rows),
            'disease_raw_pages': sum(bool(path) for path in disease_pages.values()),
        },
        'inputs': {
            'selection_manifest': str(selection_path),
            'selection_manifest_sha256': sha256_file(selection_path),
            'ranked_input': str(ranked_path),
            'ranked_input_sha256': sha256_file(ranked_path),
            'processed_dataset': str(processed),
            'model_dataset': str(model_dataset),
            'raw_root': str(raw_root),
            'relation_hashes': {
                name: sha256_file(processed / name)
                for name in ('H_C.txt', 'P_D.txt', 'H_D.txt')
            },
        },
    }
    summary_path.write_text(
        build_summary(manifest, case_rows) + '\n',
        encoding='utf-8',
    )
    manifest['outputs'] = {
        str(path.name): sha256_file(path)
        for path in (
            annotated_path, paths_path, herbs_path, targets_path,
            diseases_path, cases_path, summary_path,
        )
    }
    write_json(output_dir / 'manifest.json', manifest)
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=2))
    print('Context annotations written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
