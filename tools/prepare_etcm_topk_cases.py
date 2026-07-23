#!/usr/bin/env python3
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
    parser = argparse.ArgumentParser(
        description=(
            'Deterministically select ETCM2.0 mention10 case-study compounds '
            'before any checkpoint score is inspected.'
        )
    )
    parser.add_argument(
        '--checkpoint-manifest',
        default='configs/etcm_topk_case_checkpoints.json',
    )
    parser.add_argument(
        '--selection-manifest',
        default='configs/etcm_topk_case_selection.json',
    )
    parser.add_argument(
        '--output-dir',
        default='results/etcm_topk_cases/selection',
    )
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def aggregate_candidates(
        unseen_rows,
        ingredients,
        herbs_by_tcmip,
        compound_mappings,
        model_train_degrees):
    grouped = defaultdict(list)
    for row in unseen_rows:
        grouped[str(row['compound_id'])].append(row)

    candidates = []
    for compound_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (
            str(row['tcmip_id']), str(row['protein_id'])
        ))
        tcmip_ids = sorted({str(row['tcmip_id']) for row in rows})
        ingredient_rows = [
            ingredients[tcmip_id]
            for tcmip_id in tcmip_ids
            if tcmip_id in ingredients
        ]
        independent_herbs = {
            (
                herb.get('herb_name_pinyin', ''),
                herb.get('herb_name_latin', ''),
            )
            for tcmip_id in tcmip_ids
            for herb in herbs_by_tcmip.get(tcmip_id, [])
            if herb.get('herb_name_pinyin') or herb.get('herb_name_latin')
        }
        if not ingredient_rows or not independent_herbs:
            continue
        identity_rows = [
            row for row in ingredient_rows
            if row.get('cas_number') or row.get('pubchem_cids')
        ]
        representative = identity_rows[0] if identity_rows else ingredient_rows[0]
        mapping = compound_mappings.get(compound_id, {})
        candidates.append({
            'compound_id': compound_id,
            'tcmip_id': representative['tcmip_id'],
            'all_tcmip_ids': ';'.join(tcmip_ids),
            'compound_name': representative.get('compound_name', rows[0]['compound_name']),
            'model_train_cp_degree': int(model_train_degrees.get(compound_id, 0)),
            'unseen_confirmed_targets': len({
                str(row['protein_id']) for row in rows
            }),
            'confirmed_evidence_count': sum(
                int(row.get('evidence_count') or 0) for row in rows
            ),
            'confirmed_reference_count': sum(
                int(row.get('reference_count') or 0) for row in rows
            ),
            'independent_herb_count': len(independent_herbs),
            'mention_count': int(float(mapping.get('mention_count') or 0)),
            'cas_number': representative.get('cas_number', ''),
            'pubchem_cids': representative.get('pubchem_cids', ''),
            'has_identity_identifier': bool(identity_rows),
        })
    return candidates


def main():
    args = parse_args()
    from tools.analyze_context_subgroups import (
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
    )
    from util.etcm_topk_cases import (
        compound_degrees,
        read_csv_index,
        read_tsv,
        select_cases,
        sha256_file,
        write_json,
        write_tsv,
    )

    checkpoint_manifest_path = resolve_path(args.checkpoint_manifest)
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding='utf-8')
    )
    fold = int(checkpoint_manifest['fold'])
    case_count = int(checkpoint_manifest['case_count'])
    selection_seed = int(checkpoint_manifest['selection_seed'])

    evidence_manifest_spec = checkpoint_manifest['external_evidence_manifest']
    evidence_manifest_path = resolve_path(evidence_manifest_spec['path'])
    if sha256_file(evidence_manifest_path) != evidence_manifest_spec['sha256']:
        raise ValueError('External evidence manifest SHA-256 mismatch.')
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding='utf-8')
    )
    if not evidence_manifest.get('training_use_prohibited'):
        raise ValueError('External evidence manifest does not prohibit training use.')

    protocol_entries = []
    checkpoint_entries = []
    reference_protocol = None
    matched_fields = (
        'datapath',
        'strict_assignments_sha256',
        'outer_train_sha256',
        'outer_test_sha256',
        'model_train_sha256',
        'validation_sha256',
    )
    for model_spec in checkpoint_manifest['models']:
        config_path = resolve_path(model_spec['config'])
        if sha256_file(config_path) != model_spec['config_sha256']:
            raise ValueError('Config SHA-256 mismatch: %s' % config_path)
        protocol = prepare_protocol(config_path, fold)
        audit = protocol_audit(protocol)
        if reference_protocol is None:
            reference_protocol = protocol
            reference_audit = audit
        else:
            mismatches = {
                field: (reference_audit[field], audit[field])
                for field in matched_fields
                if reference_audit[field] != audit[field]
            }
            if mismatches:
                raise ValueError(
                    'Case-study model protocols do not match: %s' % mismatches
                )
        checkpoint_prefix, checkpoint_files = normalize_checkpoint(
            resolve_path(model_spec['checkpoint'])
        )
        index_hash = sha256_file(str(checkpoint_prefix) + '.index')
        data_hashes = [sha256_file(path) for path in checkpoint_files]
        if index_hash != model_spec['checkpoint_index_sha256']:
            raise ValueError(
                'Checkpoint index SHA-256 mismatch: %s' % checkpoint_prefix
            )
        if data_hashes != [model_spec['checkpoint_data_sha256']]:
            raise ValueError(
                'Checkpoint data SHA-256 mismatch: %s' % checkpoint_prefix
            )
        protocol_entries.append({
            'method': model_spec['method'],
            'variant': model_spec['variant'],
            'protocol': audit,
        })
        checkpoint_entries.append({
            'method': model_spec['method'],
            'variant': model_spec['variant'],
            'config': str(config_path),
            'config_sha256': model_spec['config_sha256'],
            'checkpoint': str(checkpoint_prefix),
            'checkpoint_index_sha256': index_hash,
            'checkpoint_data_sha256': data_hashes[0],
        })

    dataset_path = Path(reference_audit['datapath']).parent
    validation_root = evidence_manifest_path.parent
    unseen_path = (
        validation_root / 'validation' / checkpoint_manifest['dataset'] /
        'unseen_confirmed.tsv'
    )
    ingredient_path = validation_root / 'entities' / 'ingredient.tsv'
    ingredient_herb_path = (
        validation_root / 'relations' / 'ingredient_herb.tsv'
    )
    compound_mapping_path = (
        dataset_path / 'mappings' / 'compound_id_map.csv'
    )

    unseen_rows = read_tsv(unseen_path)
    ingredients = {
        str(row['tcmip_id']): row for row in read_tsv(ingredient_path)
    }
    herbs_by_tcmip = defaultdict(list)
    for row in read_tsv(ingredient_herb_path):
        herbs_by_tcmip[str(row['tcmip_id'])].append(row)
    compound_mappings = read_csv_index(compound_mapping_path, 'compound_id')
    model_train_degrees = compound_degrees(reference_protocol['model_train'])
    candidates = aggregate_candidates(
        unseen_rows,
        ingredients,
        herbs_by_tcmip,
        compound_mappings,
        model_train_degrees,
    )
    eligible, selected, allocation = select_cases(
        candidates, case_count=case_count, seed=selection_seed
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_fields = [
        'compound_id', 'tcmip_id', 'all_tcmip_ids', 'compound_name',
        'model_train_cp_degree', 'support_stratum', 'support_percentile',
        'unseen_confirmed_targets', 'confirmed_evidence_count',
        'confirmed_reference_count', 'independent_herb_count',
        'mention_count', 'cas_number', 'pubchem_cids',
        'has_identity_identifier', 'selection_order',
    ]
    eligible_path = output_dir / 'eligible_candidates.tsv'
    selected_path = output_dir / 'selected_cases.tsv'
    write_tsv(eligible_path, eligible, candidate_fields)
    write_tsv(selected_path, selected, candidate_fields)

    stratum_summary = {}
    for stratum in ('low', 'medium', 'high'):
        members = [
            row for row in eligible if row['support_stratum'] == stratum
        ]
        degrees = [int(row['model_train_cp_degree']) for row in members]
        stratum_summary[stratum] = {
            'eligible': len(members),
            'selected': allocation[stratum],
            'degree_min': min(degrees),
            'degree_max': max(degrees),
        }

    selection_manifest = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'selection_status': 'frozen_before_checkpoint_scoring',
        'model_scores_read': False,
        'dataset': checkpoint_manifest['dataset'],
        'fold': fold,
        'case_count': case_count,
        'selection_seed': selection_seed,
        'selection_algorithm': {
            'name': 'rank_based_training_support_tertiles_v1',
            'strata': ['low', 'medium', 'high'],
            'allocation': allocation,
            'eligibility': [
                'compound_and_protein_in_mention10_model_space',
                'at_least_one_unseen_confirmed_target',
                'ingredient_identity_row_available',
                'at_least_one_independent_ingredient_herb_source',
            ],
            'within_stratum_priority': [
                'CAS_or_PubChem_available',
                'unseen_confirmed_target_count_desc',
                'confirmed_evidence_count_desc',
                'independent_herb_count_desc',
                'mention_count_desc',
                'seeded_SHA256_tie_break',
            ],
        },
        'candidate_pool': {
            'unseen_confirmed_pairs': len(unseen_rows),
            'unseen_confirmed_compounds': len({
                str(row['compound_id']) for row in unseen_rows
            }),
            'eligible_compounds': len(eligible),
            'strata': stratum_summary,
        },
        'selected_cases': selected,
        'inputs': {
            'checkpoint_manifest': str(checkpoint_manifest_path),
            'checkpoint_manifest_sha256': sha256_file(checkpoint_manifest_path),
            'external_evidence_manifest': str(evidence_manifest_path),
            'external_evidence_manifest_sha256': sha256_file(
                evidence_manifest_path
            ),
            'unseen_confirmed': str(unseen_path),
            'unseen_confirmed_sha256': sha256_file(unseen_path),
            'ingredient_entities': str(ingredient_path),
            'ingredient_entities_sha256': sha256_file(ingredient_path),
            'ingredient_herb': str(ingredient_herb_path),
            'ingredient_herb_sha256': sha256_file(ingredient_herb_path),
            'compound_mapping': str(compound_mapping_path),
            'compound_mapping_sha256': sha256_file(compound_mapping_path),
            'eligible_candidates': str(eligible_path),
            'eligible_candidates_sha256': sha256_file(eligible_path),
            'selected_cases': str(selected_path),
            'selected_cases_sha256': sha256_file(selected_path),
        },
        'protocols': protocol_entries,
        'checkpoints': checkpoint_entries,
        'interpretation_boundary': [
            'Selection does not inspect checkpoint scores or Top-K outcomes.',
            'Ingredient evidence is prohibited from training and model selection.',
            'Potential Targets are explanatory consistency evidence, not independent truth.',
        ],
    }
    manifest_output = resolve_path(args.selection_manifest)
    write_json(manifest_output, selection_manifest)
    write_json(output_dir / 'case_selection_manifest.json', selection_manifest)

    print('ETCM Top-K cases frozen before scoring')
    print('  eligible compounds: %d' % len(eligible))
    for row in selected:
        print(
            '  #%d %s [%s] train_degree=%d unseen_confirmed=%d' % (
                row['selection_order'],
                row['compound_name'],
                row['support_stratum'],
                row['model_train_cp_degree'],
                row['unseen_confirmed_targets'],
            )
        )
    print('Selection manifest: %s' % manifest_output)
    print('Selection tables: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
