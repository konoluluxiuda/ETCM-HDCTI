#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DATASETS = {
    'TCM-Suite': REPOSITORY_ROOT / 'dataset/TCMsuite/ONE_indices.txt',
    'TCMSP': REPOSITORY_ROOT / 'dataset/TCMSP/one1.txt',
    'SymMap2.0': REPOSITORY_ROOT / 'dataset/Symmap/one.txt',
    'ETCM2.0-mention10': (
        REPOSITORY_ROOT / 'dataset/ETCM2.0_core_mention10/ONE_indices.txt'
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Frozen side-hypergraph role audit with four-dataset '
            'leave-one-dataset-out evaluation.'
        )
    )
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-positive-pairs', type=int, default=30000)
    parser.add_argument('--degree-bin-count', type=int, default=10)
    parser.add_argument('--minimum-role-gain', type=float, default=0.01)
    parser.add_argument('--minimum-matched-auc', type=float, default=0.55)
    parser.add_argument('--minimum-target-auc', type=float, default=0.49)
    parser.add_argument('--permutation-tolerance', type=float, default=0.05)
    parser.add_argument('--output-dir', default=(
        'results/side_hypergraph_role_dg/frozen_role_seed2026'
    ))
    return parser.parse_args()


def read_unique_pairs(path):
    pairs = set()
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            values = line.split()
            if not values:
                continue
            if len(values) < 2:
                raise ValueError('Invalid pair row %d in %s.' % (line_number, path))
            if len(values) >= 3 and float(values[2]) <= 0:
                continue
            pairs.add((str(values[0]), str(values[1])))
    return sorted(pairs)


def deterministic_subsample(values, maximum, seed):
    values = list(values)
    if maximum <= 0 or len(values) <= maximum:
        return values
    rng = np.random.RandomState(int(seed))
    indices = np.sort(rng.choice(len(values), size=int(maximum), replace=False))
    return [values[index] for index in indices]


def reservoir_sample_pairs(path, count, excluded_pairs, seed):
    rng = np.random.RandomState(int(seed))
    reservoir = []
    eligible = 0
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            values = line.split()
            if not values:
                continue
            if len(values) < 2:
                raise ValueError('Invalid pair row %d in %s.' % (line_number, path))
            pair = (str(values[0]), str(values[1]))
            if pair in excluded_pairs:
                continue
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(pair)
            else:
                replacement = rng.randint(eligible)
                if replacement < count:
                    reservoir[replacement] = pair
    if len(reservoir) < count:
        raise ValueError(
            '%s supplies only %d eligible negatives; %d required.' %
            (path, len(reservoir), count)
        )
    return reservoir, eligible


def generate_unobserved_pairs(positive_pairs, count, seed):
    positives = set(positive_pairs)
    compounds = sorted({left for left, _ in positives})
    proteins = sorted({right for _, right in positives})
    available = len(compounds) * len(proteins) - len(positives)
    if available < count:
        raise ValueError('Not enough unobserved pairs for deterministic sampling.')
    rng = np.random.RandomState(int(seed))
    negatives = set()
    while len(negatives) < count:
        candidate = (
            compounds[rng.randint(len(compounds))],
            proteins[rng.randint(len(proteins))],
        )
        if candidate not in positives:
            negatives.add(candidate)
    return sorted(negatives)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def lookup_rows(role, node_ids):
    return role['features'][[role['node_index'][str(value)] for value in node_ids]]


def lookup_degrees(role, node_ids):
    return role['degrees'][[role['node_index'][str(value)] for value in node_ids]]


def features_for_pairs(compound_role, protein_role, pairs):
    from util.side_hypergraph_roles import degree_pair_features, pair_role_features

    compounds = [left for left, _ in pairs]
    proteins = [right for _, right in pairs]
    compound_features = lookup_rows(compound_role, compounds)
    protein_features = lookup_rows(protein_role, proteins)
    return {
        'full': pair_role_features(compound_features, protein_features),
        'degree': degree_pair_features(
            lookup_degrees(compound_role, compounds),
            lookup_degrees(protein_role, proteins),
        ),
    }


def load_dataset(name, positive_path, args, output_dir, dataset_index):
    from rating import resolve_dataset_file
    from util.io import NEGATIVE_FILE_CANDIDATES, resolve_optional_dataset_file
    from util.side_hypergraph_roles import (
        build_role_features,
        read_incidence,
        sample_degree_matched_negatives,
    )

    dataset_dir = positive_path.resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C')).resolve()
    pd_path = Path(resolve_dataset_file(str(dataset_dir), 'P_D')).resolve()
    negative_path_value = resolve_optional_dataset_file(
        str(dataset_dir), NEGATIVE_FILE_CANDIDATES
    )
    negative_path = Path(negative_path_value).resolve() if negative_path_value else None
    all_positives = read_unique_pairs(positive_path)
    selected_positives = deterministic_subsample(
        all_positives,
        args.max_positive_pairs,
        args.seed + dataset_index * 101,
    )
    if negative_path:
        standard_negatives, negative_pool_size = reservoir_sample_pairs(
            negative_path,
            len(selected_positives),
            set(all_positives),
            args.seed + dataset_index * 101 + 1,
        )
        negative_source = str(negative_path)
    else:
        standard_negatives = generate_unobserved_pairs(
            all_positives,
            len(selected_positives),
            args.seed + dataset_index * 101 + 1,
        )
        negative_pool_size = (
            len({left for left, _ in all_positives})
            * len({right for _, right in all_positives})
            - len(all_positives)
        )
        negative_source = 'deterministic_unobserved_sampling'

    compound_universe = {
        left for left, _ in all_positives
    }.union(left for left, _ in standard_negatives)
    protein_universe = {
        right for _, right in all_positives
    }.union(right for _, right in standard_negatives)
    compound_role = build_role_features(
        read_incidence(hc_path, node_column=1, edge_column=0),
        node_universe=compound_universe,
    )
    protein_role = build_role_features(
        read_incidence(pd_path, node_column=0, edge_column=1),
        node_universe=protein_universe,
    )

    cp_compounds = sorted({left for left, _ in all_positives})
    cp_proteins = sorted({right for _, right in all_positives})
    matched_positives, matched_negatives, matched_audit = (
        sample_degree_matched_negatives(
            selected_positives,
            compound_ids=cp_compounds,
            compound_degrees=lookup_degrees(compound_role, cp_compounds),
            protein_ids=cp_proteins,
            protein_degrees=lookup_degrees(protein_role, cp_proteins),
            seed=args.seed + dataset_index * 101 + 2,
            bin_count=args.degree_bin_count,
            excluded_pairs=all_positives,
        )
    )
    if matched_audit['coverage'] < 0.95:
        raise ValueError(
            '%s degree-matched negative coverage is only %.4f.' %
            (name, matched_audit['coverage'])
        )

    standard_pairs = selected_positives + standard_negatives
    standard_labels = np.concatenate((
        np.ones(len(selected_positives), dtype=np.int32),
        np.zeros(len(standard_negatives), dtype=np.int32),
    ))
    matched_pairs = matched_positives + matched_negatives
    matched_labels = np.concatenate((
        np.ones(len(matched_positives), dtype=np.int32),
        np.zeros(len(matched_negatives), dtype=np.int32),
    ))
    standard_features = features_for_pairs(compound_role, protein_role, standard_pairs)
    matched_features = features_for_pairs(compound_role, protein_role, matched_pairs)

    positive_compounds = [left for left, _ in selected_positives]
    positive_proteins = [right for _, right in selected_positives]
    compound_supported = lookup_degrees(compound_role, positive_compounds) > 0
    protein_supported = lookup_degrees(protein_role, positive_proteins) > 0
    finite_coverage = float(np.mean(np.all(np.isfinite(standard_features['full']), axis=1)))

    feature_dir = output_dir / 'role_features'
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_dir / ('%s.npz' % name.lower().replace('.', '').replace('-', '_')),
        compound_ids=np.asarray(compound_role['node_ids'], dtype=str),
        compound_features=compound_role['features'].astype(np.float32),
        compound_degrees=compound_role['degrees'],
        protein_ids=np.asarray(protein_role['node_ids'], dtype=str),
        protein_features=protein_role['features'].astype(np.float32),
        protein_degrees=protein_role['degrees'],
    )

    input_files = [positive_path, hc_path, pd_path]
    if negative_path:
        input_files.append(negative_path)
    return {
        'name': name,
        'standard_full': standard_features['full'].astype(np.float32),
        'standard_degree': standard_features['degree'].astype(np.float32),
        'standard_labels': standard_labels,
        'matched_full': matched_features['full'].astype(np.float32),
        'matched_degree': matched_features['degree'].astype(np.float32),
        'matched_labels': matched_labels,
        'audit': {
            'positive_file': str(positive_path),
            'hc_file': str(hc_path),
            'pd_file': str(pd_path),
            'negative_source': negative_source,
            'input_sha256': {str(path): sha256_file(path) for path in input_files},
            'all_positive_pairs': len(all_positives),
            'selected_positive_pairs': len(selected_positives),
            'standard_negative_pairs': len(standard_negatives),
            'negative_candidate_pool': int(negative_pool_size),
            'compound_entities': len(compound_role['node_ids']),
            'protein_entities': len(protein_role['node_ids']),
            'compound_side_support_positive_pairs': float(np.mean(compound_supported)),
            'protein_side_support_positive_pairs': float(np.mean(protein_supported)),
            'both_sides_supported_positive_pairs': float(np.mean(
                compound_supported & protein_supported
            )),
            'role_vector_finite_coverage': finite_coverage,
            'degree_matched': matched_audit,
        },
    }


def domain_weights(datasets, source_names):
    total = sum(len(datasets[name]['standard_labels']) for name in source_names)
    domain_count = len(source_names)
    return np.concatenate([
        np.full(
            len(datasets[name]['standard_labels']),
            total / float(domain_count * len(datasets[name]['standard_labels'])),
            dtype=np.float64,
        )
        for name in source_names
    ])


def fit_probe(datasets, source_names, feature_key, seed, permute=False):
    features = np.concatenate([
        datasets[name]['standard_%s' % feature_key] for name in source_names
    ])
    labels = np.concatenate([
        datasets[name]['standard_labels'] for name in source_names
    ])
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    if permute:
        labels = np.random.RandomState(int(seed)).permutation(labels)
    model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=2000,
        tol=1e-6,
        random_state=int(seed),
        average=True,
    )
    model.fit(
        transformed,
        labels,
        sample_weight=domain_weights(datasets, source_names),
    )
    return scaler, model


def metrics(labels, scores):
    return {
        'AUC': float(roc_auc_score(labels, scores)),
        'AUPR': float(average_precision_score(labels, scores)),
    }


def evaluate_target(dataset, scaler, model, feature_key, matched=False):
    prefix = 'matched' if matched else 'standard'
    features = dataset['%s_%s' % (prefix, feature_key)]
    labels = dataset['%s_labels' % prefix]
    scores = model.decision_function(scaler.transform(features))
    return metrics(labels, scores)


def markdown_report(report):
    lines = [
        '# RG-SHADG 四库冻结角色审计',
        '',
        '## 判定',
        '',
        '**%s**' % report['decision'],
        '',
        '- 角色特征仅来自 H-C/P-D 侧超图，不使用 C-P degree、PageRank 或测试标签。',
        '- 每轮使用三个源数据库拟合容量受限线性 probe，第四个数据库只做外部评价。',
        '- isolated 实体保留为零结构角色并设置 support 指示位，不因侧关系缺失而删除。',
        '- 标准化器和分类器均只在源数据库拟合。',
        '',
        '## Leave-one-dataset-out',
        '',
        '| 目标库 | Degree AUPR | Full-role AUPR | Delta | Full AUC | Matched AUC | Permuted AUC | Permuted AUPR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for target, values in report['targets'].items():
        lines.append('| %s | %.6f | %.6f | %+.6f | %.6f | %.6f | %.6f | %.6f |' % (
            target,
            values['degree']['AUPR'],
            values['full_role']['AUPR'],
            values['full_role_gain_AUPR'],
            values['full_role']['AUC'],
            values['degree_matched_full_role']['AUC'],
            values['label_permutation']['AUC'],
            values['label_permutation']['AUPR'],
        ))
    lines.extend([
        '',
        '## 侧超图覆盖',
        '',
        '| 数据集 | Compound support | Protein support | Both support | Role-vector finite | Matched coverage |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for name, values in report['dataset_audits'].items():
        lines.append('| %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |' % (
            name,
            100.0 * values['compound_side_support_positive_pairs'],
            100.0 * values['protein_side_support_positive_pairs'],
            100.0 * values['both_sides_supported_positive_pairs'],
            100.0 * values['role_vector_finite_coverage'],
            100.0 * values['degree_matched']['coverage'],
        ))
    lines.extend([
        '',
        '## 预注册条件',
        '',
        '| 条件 | 是否通过 |',
        '|---|---|',
    ])
    for criterion, passed in report['criteria'].items():
        lines.append('| %s | %s |' % (criterion, '是' if passed else '否'))
    lines.extend([
        '',
        '## 边界',
        '',
        '- 该结果只筛选标签独立结构角色是否值得进入神经模型，不是 RG-SHADG 最终性能。',
        '- 未对目标库使用标签调参、早停或阈值选择。',
        '- 未观测 C-P pair 不是已确认生物学负例；degree-matched 对照仅用于排查度数捷径。',
        '- 单次标签置换在强跨域偏移下可能偏离 0.5；最终判定同时要求 role gain、degree-matched AUC 和最低目标 AUC，不能只依赖置换结果。',
        '- 若审计 No-Go，不搜索 role 指标、bin 数、分类器容量或域损失。',
        '',
    ])
    return '\n'.join(lines)


def write_metrics_csv(path, targets):
    rows = []
    for target, values in targets.items():
        for model_name in ('degree', 'full_role', 'degree_matched_full_role', 'label_permutation'):
            rows.append({
                'target_dataset': target,
                'model': model_name,
                'AUC': values[model_name]['AUC'],
                'AUPR': values[model_name]['AUPR'],
            })
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.max_positive_pairs <= 0:
        raise ValueError('--max-positive-pairs must be positive.')
    output_dir = (REPOSITORY_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for index, (name, path) in enumerate(DATASETS.items()):
        print('Preparing side-hypergraph roles: %s' % name, flush=True)
        datasets[name] = load_dataset(name, path, args, output_dir, index)
        audit = datasets[name]['audit']
        print(
            '  pairs=%d support(C/P/both)=%.3f/%.3f/%.3f matched=%.3f' % (
                audit['selected_positive_pairs'],
                audit['compound_side_support_positive_pairs'],
                audit['protein_side_support_positive_pairs'],
                audit['both_sides_supported_positive_pairs'],
                audit['degree_matched']['coverage'],
            ),
            flush=True,
        )

    targets = {}
    names = list(datasets)
    for target_index, target_name in enumerate(names):
        source_names = [name for name in names if name != target_name]
        degree_scaler, degree_model = fit_probe(
            datasets, source_names, 'degree', args.seed + target_index * 17
        )
        full_scaler, full_model = fit_probe(
            datasets, source_names, 'full', args.seed + target_index * 17 + 1
        )
        permutation_scaler, permutation_model = fit_probe(
            datasets,
            source_names,
            'full',
            args.seed + target_index * 17 + 2,
            permute=True,
        )
        degree_result = evaluate_target(
            datasets[target_name], degree_scaler, degree_model, 'degree'
        )
        full_result = evaluate_target(
            datasets[target_name], full_scaler, full_model, 'full'
        )
        matched_result = evaluate_target(
            datasets[target_name], full_scaler, full_model, 'full', matched=True
        )
        permutation_result = evaluate_target(
            datasets[target_name], permutation_scaler, permutation_model, 'full'
        )
        targets[target_name] = {
            'source_datasets': source_names,
            'degree': degree_result,
            'full_role': full_result,
            'full_role_gain_AUPR': (
                full_result['AUPR'] - degree_result['AUPR']
            ),
            'degree_matched_full_role': matched_result,
            'label_permutation': permutation_result,
        }
        print(
            'Target %s: degree AUPR=%.4f full AUPR=%.4f delta=%+.4f matched AUC=%.4f' % (
                target_name,
                degree_result['AUPR'],
                full_result['AUPR'],
                targets[target_name]['full_role_gain_AUPR'],
                matched_result['AUC'],
            ),
            flush=True,
        )

    role_gain_passes = sum(
        values['full_role_gain_AUPR'] >= args.minimum_role_gain
        for values in targets.values()
    )
    matched_auc_passes = sum(
        values['degree_matched_full_role']['AUC'] >= args.minimum_matched_auc
        for values in targets.values()
    )
    permutation_passes = all(
        abs(values['label_permutation']['AUC'] - 0.5) <= args.permutation_tolerance
        and abs(values['label_permutation']['AUPR'] - 0.5) <= args.permutation_tolerance
        for values in targets.values()
    )
    criteria = {
        'all_role_vectors_finite': all(
            values['audit']['role_vector_finite_coverage'] >= 0.95
            for values in datasets.values()
        ),
        'full_role_gain_at_least_0.01_on_3_of_4_targets': role_gain_passes >= 3,
        'degree_matched_auc_at_least_0.55_on_3_of_4_targets': matched_auc_passes >= 3,
        'no_target_full_role_auc_below_0.49': all(
            values['full_role']['AUC'] >= args.minimum_target_auc
            for values in targets.values()
        ),
        'label_permutation_returns_to_random': permutation_passes,
        'source_only_preprocessing_and_model_selection': True,
    }
    decision = (
        'GO: implement one fixed RG-SHADG neural pilot.'
        if all(criteria.values())
        else 'NO-GO: stop RG-SHADG before neural implementation.'
    )
    report = {
        'audit_type': 'side_hypergraph_role_domain_generalization_frozen_probe',
        'created_at': datetime.now().astimezone().isoformat(),
        'decision': decision,
        'configuration': vars(args),
        'role_feature_names': list(
            __import__(
                'util.side_hypergraph_roles', fromlist=['ROLE_FEATURE_NAMES']
            ).ROLE_FEATURE_NAMES
        ),
        'dataset_audits': {
            name: values['audit'] for name, values in datasets.items()
        },
        'targets': targets,
        'criteria': criteria,
    }
    with (output_dir / 'report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    (output_dir / 'report.md').write_text(
        markdown_report(report), encoding='utf-8'
    )
    write_metrics_csv(output_dir / 'metrics.csv', targets)
    print('\n%s' % decision)
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()
