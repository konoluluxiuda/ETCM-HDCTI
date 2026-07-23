#!/usr/bin/env python3
"""Freeze E-level ETCM Top-K pairs before independent evidence searches."""

import argparse
import json
import sys
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
        '--context-manifest',
        default='results/etcm_topk_cases/fold1/context/manifest.json',
    )
    parser.add_argument(
        '--ranked-input',
        default=(
            'results/etcm_topk_cases/fold1/context/'
            'context_annotated_topk.tsv'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default='results/etcm_topk_cases/manual_validation',
    )
    parser.add_argument(
        '--frozen-config',
        default='configs/etcm_topk_manual_validation.json',
    )
    parser.add_argument('--per-compound', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def build_queries(candidate):
    compound = candidate['compound_name']
    gene = candidate['gene_symbol'].split('|')[0]
    target = candidate['target_name'].split('|')[0]
    return {
        'BindingDB': '"%s" AND (%s OR "%s")' % (
            compound, gene, target
        ),
        'ChEMBL': '"%s" AND (%s OR "%s")' % (
            compound, gene, target
        ),
        'PubMed': '("%s"[Title/Abstract]) AND (%s[Title/Abstract] OR "%s"[Title/Abstract])' % (
            compound, gene, target
        ),
    }


def build_summary(payload):
    lines = [
        '# ETCM2.0 Top-K 独立证据核验工作单',
        '',
        '候选在任何 BindingDB、ChEMBL 或 PubMed 检索开始前冻结。本工作单只用于',
        '对 E 级候选补充独立证据，不改变模型、checkpoint、案例或排名。',
        '',
        '## Material Passport',
        '',
        '- Schema: ARS-9',
        '- Artifact type: frozen evidence-verification worklist',
        '- Verification status: FROZEN_UNSEARCHED',
        '- Candidates: %d' % payload['candidate_count'],
        '- Selection seed: %d' % payload['selection_seed'],
        '',
        '## 选择规则',
        '',
        '每个已冻结成分选择 %d 个候选。候选必须为 E 级，并具有 Gene Symbol、'
        'Target Name 和 UniProt。成分内依次按模型重复数、reciprocal-rank 总和、'
        '最佳排名、平均排名和固定 SHA-256 tie-break 排序。C-H-D-P 路径数不参与'
        '选择。' % payload['per_compound'],
        '',
        '## 冻结候选',
        '',
        '| 序 | 成分 | Gene | Target | 模型数 | 最佳排名 | 模型排名 |',
        '|---:|---|---|---|---:|---:|---|',
    ]
    for row in payload['selected_candidates']:
        lines.append('| %d | %s | %s | %s | %d | %d | %s |' % (
            row['validation_order'],
            row['compound_name'],
            row['gene_symbol'],
            row['target_name'],
            row['method_count'],
            row['best_rank'],
            row['method_ranks'],
        ))
    lines.extend([
        '',
        '## 判定规则',
        '',
        '- B1：独立来源提供直接定量结合/活性证据，如 Kd、Ki、IC50、EC50。',
        '- B2：独立实验明确支持直接 C-P 作用，但定量信息不完整。',
        '- D：只有通路、表达、疾病共现、分子对接或间接机制支持。',
        '- E：未找到可核验的直接或间接支持。',
        '- Conflict：来源明确报告无活性或与候选关系相冲突。',
        '',
        '每条证据保留数据库记录号、URL、DOI/PMID、物种、实验类型、活性值、',
        '检索日期和人工备注。未记录关系仍是 unlabeled，不能按真实负例处理。',
        '',
    ])
    return '\n'.join(lines)


def main():
    from util.etcm_topk_cases import (
        read_tsv,
        select_manual_validation_candidates,
        sha256_file,
        write_json,
        write_tsv,
    )

    args = parse_args()
    selection_path = resolve_path(args.selection_manifest)
    context_manifest_path = resolve_path(args.context_manifest)
    ranked_path = resolve_path(args.ranked_input)
    output_dir = resolve_path(args.output_dir)
    frozen_config = resolve_path(args.frozen_config)

    existing = [
        path for path in (output_dir, frozen_config) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            'Frozen manual-validation artifacts already exist: %s'
            % ', '.join(str(path) for path in existing)
        )

    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    context_manifest = json.loads(
        context_manifest_path.read_text(encoding='utf-8')
    )
    if selection.get('selection_status') != (
            'frozen_before_checkpoint_scoring'):
        raise ValueError('Top-K case selection is not frozen.')
    if context_manifest.get('purpose') != (
            'posthoc_topk_case_explanation_only'):
        raise ValueError('Unexpected context annotation purpose.')
    if context_manifest['inputs']['selection_manifest_sha256'] != (
            sha256_file(selection_path)):
        raise ValueError('Context annotations use a different selection.')
    if context_manifest['outputs'].get(ranked_path.name) != (
            sha256_file(ranked_path)):
        raise ValueError('Context-ranked input hash does not match manifest.')

    compound_order = [
        str(row['compound_id']) for row in selection['selected_cases']
    ]
    selected = select_manual_validation_candidates(
        read_tsv(ranked_path),
        compound_order,
        per_compound=args.per_compound,
        seed=args.seed,
    )
    case_metadata = {
        str(row['compound_id']): row
        for row in selection['selected_cases']
    }
    for row in selected:
        metadata = case_metadata[row['compound_id']]
        row['all_tcmip_ids'] = metadata.get('all_tcmip_ids', '')
        row['cas_number'] = metadata.get('cas_number', '')
        row['pubchem_cids'] = metadata.get('pubchem_cids', '')
    expected = len(compound_order) * int(args.per_compound)
    if len(selected) != expected:
        raise ValueError(
            'Selected %d candidates; expected %d.' % (len(selected), expected)
        )

    payload = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'freeze_status': 'frozen_before_external_search',
        'external_search_started': False,
        'material_passport': {
            'schema': 'ARS-9',
            'artifact_type': 'frozen_evidence_verification_worklist',
            'verification_status': 'FROZEN_UNSEARCHED',
        },
        'dataset': selection['dataset'],
        'fold': selection['fold'],
        'candidate_count': len(selected),
        'per_compound': int(args.per_compound),
        'selection_seed': int(args.seed),
        'selection_rule': [
            'evidence_level_E_only',
            'gene_target_uniprot_identity_required',
            'equal_allocation_per_frozen_compound',
            'method_count_desc',
            'reciprocal_rank_sum_desc',
            'best_rank_asc',
            'mean_rank_asc',
            'seeded_sha256_tie_break',
            'context_path_count_not_used',
        ],
        'evidence_grades': {
            'B1': 'direct quantitative binding or activity evidence',
            'B2': 'direct experimental C-P support without complete quantitative data',
            'D': 'indirect pathway, expression, docking, or co-occurrence support',
            'E': 'no verified support found',
            'Conflict': 'verified evidence inconsistent with the predicted relation',
        },
        'restrictions': {
            'used_for_training': False,
            'used_for_checkpoint_selection': False,
            'used_for_case_selection': False,
            'used_for_candidate_ranking': False,
            'may_change_existing_ranks': False,
        },
        'inputs': {
            'case_selection_manifest': str(selection_path),
            'case_selection_sha256': sha256_file(selection_path),
            'context_manifest': str(context_manifest_path),
            'context_manifest_sha256': sha256_file(context_manifest_path),
            'ranked_input': str(ranked_path),
            'ranked_input_sha256': sha256_file(ranked_path),
        },
        'selected_candidates': selected,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / 'candidates.tsv'
    queries_path = output_dir / 'search_queries.tsv'
    review_path = output_dir / 'evidence_review.tsv'
    summary_path = output_dir / 'summary.md'
    manifest_path = output_dir / 'candidate_manifest.json'

    candidate_fields = list(selected[0].keys())
    write_tsv(candidates_path, selected, candidate_fields)

    query_rows = []
    review_rows = []
    for candidate in selected:
        for database, query in build_queries(candidate).items():
            base = {
                'validation_order': candidate['validation_order'],
                'compound_id': candidate['compound_id'],
                'tcmip_id': candidate['tcmip_id'],
                'compound_name': candidate['compound_name'],
                'all_tcmip_ids': candidate['all_tcmip_ids'],
                'cas_number': candidate['cas_number'],
                'pubchem_cids': candidate['pubchem_cids'],
                'protein_id': candidate['protein_id'],
                'gene_symbol': candidate['gene_symbol'],
                'target_name': candidate['target_name'],
                'uniprot_accessions': candidate['uniprot_accessions'],
                'source_database': database,
                'query': query,
            }
            query_rows.append(base)
            review_rows.append(dict(base, **{
                'search_status': 'pending',
                'searched_at': '',
                'record_id': '',
                'source_url': '',
                'doi': '',
                'pmid': '',
                'species': '',
                'assay_or_study_type': '',
                'relation_type': '',
                'activity_type': '',
                'activity_value': '',
                'activity_unit': '',
                'evidence_grade': '',
                'verification_verdict': '',
                'reviewer_notes': '',
            }))
    query_fields = list(query_rows[0].keys())
    review_fields = list(review_rows[0].keys())
    write_tsv(queries_path, query_rows, query_fields)
    write_tsv(review_path, review_rows, review_fields)
    summary_path.write_text(
        build_summary(payload) + '\n', encoding='utf-8'
    )

    payload['outputs'] = {
        str(path.name): sha256_file(path)
        for path in (candidates_path, queries_path, review_path, summary_path)
    }
    write_json(manifest_path, payload)
    payload['candidate_manifest_sha256'] = sha256_file(manifest_path)
    write_json(frozen_config, payload)

    print(json.dumps({
        'freeze_status': payload['freeze_status'],
        'candidate_count': payload['candidate_count'],
        'queries': len(query_rows),
        'candidate_manifest_sha256': payload[
            'candidate_manifest_sha256'
        ],
        'frozen_config': str(frozen_config),
        'output_dir': str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
