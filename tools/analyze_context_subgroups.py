#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Compare no-context and HerbOnly checkpoints on the same Strict fold, '
            'then audit score changes by H-C degree, training C-P degree, and mention count.'
        )
    )
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--herb-config', required=True)
    parser.add_argument('--herb-checkpoint', required=True)
    parser.add_argument('--fold', type=int, default=1, help='One-based Strict outer fold.')
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate configs, split identity, and checkpoint files without TensorFlow.',
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records):
    lines = [
        '%s\t%s\t%d' % (str(left), str(right), int(float(label) > 0))
        for left, right, label in records
    ]
    return hashlib.sha256(
        ('\n'.join(sorted(lines)) + '\n').encode('utf-8')
    ).hexdigest()


def normalize_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        preferred = path / 'hdcti_model.ckpt.index'
        if preferred.exists():
            path = preferred
        else:
            index_files = sorted(path.glob('*.index'))
            if len(index_files) != 1:
                raise FileNotFoundError(
                    'Expected exactly one *.index file in %s, found %d.' %
                    (path, len(index_files))
                )
            path = index_files[0]
    text = str(path)
    for suffix in ('.index', '.meta'):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    if '.data-' in text:
        text = text.split('.data-', 1)[0]
    prefix = Path(text)
    if not Path(str(prefix) + '.index').exists():
        raise FileNotFoundError('Checkpoint index not found: %s.index' % prefix)
    data_files = sorted(prefix.parent.glob(prefix.name + '.data-*'))
    if not data_files:
        raise FileNotFoundError('Checkpoint data shard not found for %s.' % prefix)
    return prefix, data_files


def prepare_protocol(config_path, fold_one_based):
    from util.config import ModelConf, OptionConf
    from util.dataSplit import DataSplit
    from util.model_components import resolve_early_stopping

    config_path = Path(config_path).expanduser().resolve()
    conf = ModelConf(str(config_path))
    protocol = conf['experiment.protocol'].strip().lower()
    if protocol != 'strict':
        raise ValueError('Context subgroup analysis requires experiment.protocol=strict.')
    evaluation = OptionConf(conf['evaluation.setup'])
    if not evaluation.contains('-cv'):
        raise ValueError('Context subgroup analysis requires evaluation.setup=-cv K.')
    fold_count = int(evaluation['-cv'])
    if fold_one_based < 1 or fold_one_based > fold_count:
        raise ValueError('--fold must be between 1 and %d.' % fold_count)

    folds, manifest = DataSplit.prepareStrictFolds(conf, conf['datapath'], fold_count)
    outer_train, outer_test = folds[fold_one_based - 1]
    early_stopping = resolve_early_stopping(conf)
    model_train = outer_train
    validation = []
    validation_info = None
    if early_stopping['enabled']:
        base_seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
        validation_seed = (
            int(conf['validation.seed'])
            if conf.contains('validation.seed') else base_seed + 100000
        )
        model_train, validation, validation_info = DataSplit.innerValidationSplit(
            outer_train, early_stopping['ratio'], validation_seed + fold_one_based - 1
        )
    return {
        'config_path': config_path,
        'conf': conf,
        'fold_count': fold_count,
        'manifest': manifest,
        'outer_train': outer_train,
        'outer_test': outer_test,
        'model_train': model_train,
        'validation': validation,
        'validation_info': validation_info,
    }


def protocol_audit(protocol):
    manifest = protocol['manifest']
    return {
        'config_path': str(protocol['config_path']),
        'config_sha256': sha256_file(protocol['config_path']),
        'datapath': str(Path(protocol['conf']['datapath']).resolve()),
        'strict_assignments': manifest.get('assignments_path'),
        'strict_assignments_sha256': manifest.get('assignments_sha256'),
        'outer_train_sha256': records_sha256(protocol['outer_train']),
        'outer_test_sha256': records_sha256(protocol['outer_test']),
        'model_train_sha256': records_sha256(protocol['model_train']),
        'validation_sha256': records_sha256(protocol['validation']),
        'validation_info': protocol['validation_info'],
    }


def validate_matched_protocols(baseline, herb):
    baseline_audit = protocol_audit(baseline)
    herb_audit = protocol_audit(herb)
    matched_fields = (
        'datapath', 'strict_assignments_sha256', 'outer_train_sha256',
        'outer_test_sha256', 'model_train_sha256', 'validation_sha256',
    )
    mismatches = {
        field: (baseline_audit[field], herb_audit[field])
        for field in matched_fields
        if baseline_audit[field] != herb_audit[field]
    }
    if mismatches:
        raise ValueError('Baseline and HerbOnly protocols do not match: %s' % mismatches)
    return baseline_audit, herb_audit


def checkpoint_audit(prefix, data_files):
    return {
        'prefix': str(prefix),
        'index_sha256': sha256_file(str(prefix) + '.index'),
        'data_shards': {path.name: sha256_file(path) for path in data_files},
    }


def weight_audit(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'size': int(values.size),
        'mean': float(np.mean(values)),
        'mean_abs': float(np.mean(np.abs(values))),
        'minimum': float(np.min(values)),
        'maximum': float(np.max(values)),
        'l2_norm': float(np.linalg.norm(values)),
    }


def restore_snapshot(tf, HDCTI, set_global_seed, protocol, checkpoint_prefix, fold):
    tf.reset_default_graph()
    conf = protocol['conf']
    base_seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
    set_global_seed(base_seed + fold - 1, reset_tensorflow_graph=False)
    model = HDCTI(
        conf, protocol['model_train'], protocol['outer_test'], '[%d]' % fold
    )
    model.validationData = protocol['validation']
    model.readConfiguration()
    model.initModel()

    checkpoint_variables = dict(tf.train.list_variables(str(checkpoint_prefix)))
    graph_variables = {
        variable.name.split(':', 1)[0]: variable
        for variable in tf.global_variables()
    }
    missing = sorted(set(graph_variables) - set(checkpoint_variables))
    shape_mismatches = []
    for name, variable in graph_variables.items():
        if name not in checkpoint_variables:
            continue
        graph_shape = variable.shape.as_list()
        checkpoint_shape = list(checkpoint_variables[name])
        if graph_shape != checkpoint_shape:
            shape_mismatches.append({
                'name': name,
                'graph': graph_shape,
                'checkpoint': checkpoint_shape,
            })
    if missing or shape_mismatches:
        model.sess.close()
        raise ValueError(
            'Checkpoint/config mismatch: missing=%s; shape mismatches=%s.' %
            (missing, shape_mismatches)
        )

    saver = tf.train.Saver(var_list=graph_variables)
    saver.restore(model.sess, str(checkpoint_prefix))
    state = model.fetchModelState()
    snapshot = {
        'compound': state['compound'],
        'protein': state['protein'],
        'compound_context': state['compound_context'],
        'protein_context': state['protein_context'],
        'herb_edge': state['herb_edge'],
        'disease_edge': state['disease_edge'],
        'weights': state['weights'],
        'compound_map': dict(model.data.compound),
        'protein_map': dict(model.data.protein),
        'herb_map': dict(model.data.herb),
        'disease_map': dict(model.data.disease),
        'context_terms': dict(model.context_terms),
        'pair_decoder': dict(model.pair_decoder),
        'num_compounds': model.num_compounds,
        'num_proteins': model.num_proteins,
    }
    model.sess.close()
    return snapshot


def score_snapshot(snapshot, records, include_context):
    from util.model_components import (
        context_interaction_pair_scores,
        pair_decoder_scores,
    )

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
    compounds = snapshot['compound'][compound_indices]
    proteins = snapshot['protein'][protein_indices]
    base_logits = pair_decoder_scores(
        compounds,
        proteins,
        decoder_type=snapshot['pair_decoder']['type'],
        decoder_weights=snapshot['weights'],
    )
    if not include_context:
        return np.asarray(base_logits, dtype=np.float64), np.asarray(base_logits, dtype=np.float64)

    dimension = snapshot['compound'].shape[1]
    zero_weight = np.zeros(dimension, dtype=np.float32)
    weights = snapshot['weights']
    total_logits = context_interaction_pair_scores(
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
    )
    return np.asarray(base_logits, dtype=np.float64), np.asarray(total_logits, dtype=np.float64)


def read_relation_degrees(path, entity_column):
    degrees = Counter()
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                degrees[str(parts[entity_column])] += 1
    return degrees


def read_mention_counts(dataset_dir):
    path = dataset_dir / 'mappings' / 'compound_id_map.csv'
    if not path.exists():
        return {}, None
    values = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'compound_id' not in reader.fieldnames:
            return {}, str(path)
        for row in reader:
            raw = row.get('mention_count')
            if raw not in (None, ''):
                values[str(row['compound_id'])] = int(float(raw))
    return values, str(path)


def flatten_subgroups(report):
    rows = []
    for subgroup, groups in report['subgroups'].items():
        for group, summary in groups.items():
            delta = summary['herb_total_minus_baseline']
            context = summary['context_logit']
            transitions = summary['prediction_transitions']
            rows.append({
                'subgroup': subgroup,
                'group': group,
                'records': summary['records'],
                'positives': summary['positives'],
                'negatives': summary['negatives'],
                'baseline_AUPR': summary['baseline']['AUPR'],
                'herb_total_AUPR': summary['herb_total']['AUPR'],
                'delta_AUC': delta['AUC'],
                'delta_AUPR': delta['AUPR'],
                'delta_Recall': delta['Recall'],
                'delta_Precision': delta['Precision'],
                'delta_F1': delta['F1-score'],
                'context_positive_mean': context['positive_mean'],
                'context_negative_mean': context['negative_mean'],
                'context_mean_abs': context['mean_abs'],
                'FN_to_TP': transitions['FN_to_TP'],
                'TP_to_FN': transitions['TP_to_FN'],
                'FP_to_TN': transitions['FP_to_TN'],
                'TN_to_FP': transitions['TN_to_FP'],
            })
    return rows


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def markdown_value(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return '%.6f' % value
    return str(value)


def build_markdown(report, metadata, calibration=None, calibrated_report=None):
    overall = report['overall']
    lines = [
        '# Hctx-P 分组机制分析',
        '',
        '- Fold: `%d/%d`' % (metadata['fold'], metadata['fold_count']),
        '- Baseline: `%s`' % metadata['baseline_checkpoint']['prefix'],
        '- HerbOnly: `%s`' % metadata['herb_checkpoint']['prefix'],
        '- 评价范围：固定 Strict outer-test pairs；纯推理，不训练、不选择 checkpoint。',
        '',
        '## 总体结果',
        '',
        '| 指标 | Baseline | Herb base only | Herb total | Herb total - Baseline |',
        '|---|---:|---:|---:|---:|',
    ]
    for metric in ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score'):
        lines.append('| %s | %s | %s | %s | %s |' % (
            metric,
            markdown_value(overall['baseline'][metric]),
            markdown_value(overall['herb_base_only'][metric]),
            markdown_value(overall['herb_total'][metric]),
            markdown_value(overall['herb_total_minus_baseline'][metric]),
        ))
    lines.extend([
        '',
        '## 分组结果',
        '',
        '| 分组 | 区间 | 样本 | 正例 | AUPR Δ | Recall Δ | Precision Δ | F1 Δ | FP→TN | TP→FN |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in flatten_subgroups(report):
        lines.append('| %s | %s | %d | %d | %s | %s | %s | %s | %d | %d |' % (
            row['subgroup'], row['group'], row['records'], row['positives'],
            markdown_value(row['delta_AUPR']), markdown_value(row['delta_Recall']),
            markdown_value(row['delta_Precision']), markdown_value(row['delta_F1']),
            row['FP_to_TN'], row['TP_to_FN'],
        ))
    if calibration is not None and calibrated_report is not None:
        lines.extend([
            '',
            '## Inner-validation 阈值校准',
            '',
            '| 模型 | 选择目标 | 阈值 | Validation Precision | Validation Recall | Validation F1 |',
            '|---|---|---:|---:|---:|---:|',
        ])
        for name, label in (('baseline', 'Baseline'), ('herb', 'HerbOnly')):
            item = calibration[name]
            metrics = item['validation_metrics']
            lines.append('| %s | %s | %s | %s | %s | %s |' % (
                label,
                item['objective'],
                markdown_value(item['threshold']),
                markdown_value(metrics['Precision']),
                markdown_value(metrics['Recall']),
                markdown_value(metrics['F1-score']),
            ))
        calibrated_overall = calibrated_report['overall']
        lines.extend([
            '',
            '阈值只由 inner validation 确定，以下为原样应用于 outer-test 的结果。',
            '',
            '| 指标 | Baseline calibrated | HerbOnly calibrated | Herb - Baseline |',
            '|---|---:|---:|---:|',
        ])
        for metric in ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score'):
            lines.append('| %s | %s | %s | %s |' % (
                metric,
                markdown_value(calibrated_overall['baseline'][metric]),
                markdown_value(calibrated_overall['herb_total'][metric]),
                markdown_value(calibrated_overall['herb_total_minus_baseline'][metric]),
            ))
        lines.extend([
            '',
            '### 校准后的训练 C-P degree 分组',
            '',
            '| 区间 | 样本 | 正例 | Recall Δ | Precision Δ | F1 Δ | FP→TN | TP→FN |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for group, summary in calibrated_report['subgroups']['training C-P degree'].items():
            delta = summary['herb_total_minus_baseline']
            transitions = summary['prediction_transitions']
            lines.append('| %s | %d | %d | %s | %s | %s | %d | %d |' % (
                group,
                summary['records'],
                summary['positives'],
                markdown_value(delta['Recall']),
                markdown_value(delta['Precision']),
                markdown_value(delta['F1-score']),
                transitions['FP_to_TN'],
                transitions['TP_to_FN'],
            ))
    lines.extend([
        '',
        '## 解释边界',
        '',
        '- `Herb base only` 使用 HerbOnly checkpoint 的节点表示，但关闭显式 Hctx-P 打分项；它不是独立训练的消融模型。',
        '- `Herb total - Baseline` 同时包含训练期间表示变化与显式上下文项，不能只归因于单个加法分数。',
        '- 分组使用外层测试标签进行事后机制分析，不能据此调参后再把同一外层结果作为无偏确认性证据。',
        '- 校准阈值分别由相同 fold 的 inner validation 确定；outer-test 从未参与阈值选择。',
        '- 只有当低证据组持续出现更大的 Recall 损失或 TP→FN，才为可靠性门控提供直接支持。',
        '',
    ])
    return '\n'.join(lines)


def prediction_transition(label, baseline_prediction, herb_prediction):
    if label == 1 and baseline_prediction == 0 and herb_prediction == 1:
        return 'FN_to_TP'
    if label == 1 and baseline_prediction == 1 and herb_prediction == 0:
        return 'TP_to_FN'
    if label == 0 and baseline_prediction == 1 and herb_prediction == 0:
        return 'FP_to_TN'
    if label == 0 and baseline_prediction == 0 and herb_prediction == 1:
        return 'TN_to_FP'
    return 'unchanged'


def main():
    args = parse_args()
    baseline_protocol = prepare_protocol(args.baseline_config, args.fold)
    herb_protocol = prepare_protocol(args.herb_config, args.fold)
    baseline_audit, herb_audit = validate_matched_protocols(
        baseline_protocol, herb_protocol
    )
    baseline_checkpoint, baseline_files = normalize_checkpoint(args.baseline_checkpoint)
    herb_checkpoint, herb_files = normalize_checkpoint(args.herb_checkpoint)

    metadata = {
        'evaluation_type': 'checkpoint_context_subgroup_pure_inference',
        'created_at': datetime.now().astimezone().isoformat(),
        'fold': args.fold,
        'fold_count': baseline_protocol['fold_count'],
        'baseline_protocol': baseline_audit,
        'herb_protocol': herb_audit,
        'baseline_checkpoint': checkpoint_audit(baseline_checkpoint, baseline_files),
        'herb_checkpoint': checkpoint_audit(herb_checkpoint, herb_files),
    }
    print('Context subgroup checkpoint analysis')
    print('  fold: %d/%d' % (args.fold, baseline_protocol['fold_count']))
    print('  test pairs: %d' % len(baseline_protocol['outer_test']))
    print('  protocol hashes: matched')
    print('  optimizer/training steps: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.gpu import configure_cuda_environment
    configure_cuda_environment(baseline_protocol['conf'])
    from util.reproducibility import set_global_seed
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI
    from rating import resolve_dataset_file
    from util.context_subgroups import (
        build_subgroup_report,
        herb_degree_bin,
        mention_count_bin,
        select_f1_threshold,
        sigmoid,
        training_cp_degree_bin,
    )

    baseline_snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, baseline_protocol, baseline_checkpoint, args.fold
    )
    print('Baseline checkpoint restored.')
    herb_snapshot = restore_snapshot(
        tf, HDCTI, set_global_seed, herb_protocol, herb_checkpoint, args.fold
    )
    print('HerbOnly checkpoint restored.')
    if (
        baseline_snapshot['compound_map'] != herb_snapshot['compound_map']
        or baseline_snapshot['protein_map'] != herb_snapshot['protein_map']
    ):
        raise ValueError('Baseline and HerbOnly entity mappings do not match.')
    if 'context_herb_protein' not in herb_snapshot['weights']:
        raise ValueError('HerbOnly checkpoint does not contain context_herb_protein.')
    metadata['herb_context'] = {
        'enabled_terms': herb_snapshot['context_terms'],
        'weight': weight_audit(herb_snapshot['weights']['context_herb_protein']),
    }

    records = baseline_protocol['outer_test']
    baseline_base, baseline_total = score_snapshot(
        baseline_snapshot, records, include_context=False
    )
    herb_base, herb_total = score_snapshot(
        herb_snapshot, records, include_context=True
    )
    context_logits = herb_total - herb_base
    labels = np.asarray([int(float(row[2]) > 0) for row in records], dtype=np.int32)
    compound_ids = [str(row[0]) for row in records]
    protein_ids = [str(row[1]) for row in records]

    validation_records = baseline_protocol['validation']
    if not validation_records:
        raise ValueError(
            'Threshold calibration requires early stopping with a non-empty inner validation set.'
        )
    _, baseline_validation_logits = score_snapshot(
        baseline_snapshot, validation_records, include_context=False
    )
    _, herb_validation_logits = score_snapshot(
        herb_snapshot, validation_records, include_context=True
    )
    validation_labels = np.asarray(
        [int(float(row[2]) > 0) for row in validation_records], dtype=np.int32
    )
    calibration = {
        'selection_data': 'inner_validation',
        'validation_records': len(validation_records),
        'validation_sha256': baseline_audit['validation_sha256'],
        'baseline': select_f1_threshold(
            validation_labels, baseline_validation_logits
        ),
        'herb': select_f1_threshold(validation_labels, herb_validation_logits),
    }

    dataset_dir = Path(baseline_protocol['conf']['datapath']).resolve().parent
    hc_path = Path(resolve_dataset_file(str(dataset_dir), 'H_C'))
    hc_degrees = read_relation_degrees(hc_path, entity_column=1)
    train_cp_degrees = Counter(
        str(row[0]) for row in baseline_protocol['model_train'] if float(row[2]) > 0
    )
    mention_counts, mention_path = read_mention_counts(dataset_dir)
    raw_hc_degrees = [hc_degrees.get(compound_id, 0) for compound_id in compound_ids]
    raw_train_cp_degrees = [
        train_cp_degrees.get(compound_id, 0) for compound_id in compound_ids
    ]
    raw_mentions = [mention_counts.get(compound_id) for compound_id in compound_ids]
    subgroup_values = {
        'H-C degree': [herb_degree_bin(value) for value in raw_hc_degrees],
        'training C-P degree': [
            training_cp_degree_bin(value) for value in raw_train_cp_degrees
        ],
    }
    if mention_counts:
        subgroup_values['mention_count'] = [
            mention_count_bin(value) for value in raw_mentions
        ]

    report = build_subgroup_report(
        labels,
        baseline_total,
        herb_base,
        herb_total,
        context_logits,
        subgroup_values,
    )
    calibrated_report = build_subgroup_report(
        labels,
        baseline_total,
        herb_base,
        herb_total,
        context_logits,
        subgroup_values,
        baseline_threshold=calibration['baseline']['threshold'],
        herb_threshold=calibration['herb']['threshold'],
    )
    metadata['side_information'] = {
        'hc_path': str(hc_path),
        'hc_sha256': sha256_file(hc_path),
        'mention_path': mention_path,
        'mention_sha256': sha256_file(mention_path) if mention_path else None,
    }

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'context_subgroups' / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata': metadata,
        'calibration': calibration,
        'analysis': report,
        'calibrated_analysis': calibrated_report,
    }
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (output_dir / 'report.md').write_text(
        build_markdown(report, metadata, calibration, calibrated_report),
        encoding='utf-8',
    )

    pair_rows = []
    baseline_scores = sigmoid(baseline_total)
    herb_base_scores = sigmoid(herb_base)
    herb_scores = sigmoid(herb_total)
    baseline_calibrated_predictions = (
        baseline_scores >= calibration['baseline']['threshold']
    ).astype(np.int32)
    herb_calibrated_predictions = (
        herb_scores >= calibration['herb']['threshold']
    ).astype(np.int32)
    for index, (compound_id, protein_id, label) in enumerate(
            zip(compound_ids, protein_ids, labels)):
        baseline_prediction = int(baseline_scores[index] >= 0.5)
        herb_prediction = int(herb_scores[index] >= 0.5)
        transition = prediction_transition(
            label, baseline_prediction, herb_prediction
        )
        baseline_calibrated_prediction = int(
            baseline_calibrated_predictions[index]
        )
        herb_calibrated_prediction = int(herb_calibrated_predictions[index])
        calibrated_transition = prediction_transition(
            label, baseline_calibrated_prediction, herb_calibrated_prediction
        )
        pair_rows.append({
            'compound_id': compound_id,
            'protein_id': protein_id,
            'label': int(label),
            'hc_degree': raw_hc_degrees[index],
            'hc_degree_group': subgroup_values['H-C degree'][index],
            'training_cp_degree': raw_train_cp_degrees[index],
            'training_cp_degree_group': subgroup_values['training C-P degree'][index],
            'mention_count': raw_mentions[index],
            'mention_count_group': mention_count_bin(raw_mentions[index]),
            'baseline_logit': baseline_total[index],
            'baseline_score': baseline_scores[index],
            'herb_base_logit': herb_base[index],
            'herb_base_score': herb_base_scores[index],
            'context_logit': context_logits[index],
            'herb_total_logit': herb_total[index],
            'herb_total_score': herb_scores[index],
            'transition': transition,
            'baseline_calibrated_prediction': baseline_calibrated_prediction,
            'herb_calibrated_prediction': herb_calibrated_prediction,
            'calibrated_transition': calibrated_transition,
        })
    write_tsv(
        output_dir / 'pair_scores.tsv',
        pair_rows,
        list(pair_rows[0].keys()),
    )
    subgroup_rows = flatten_subgroups(report)
    write_tsv(
        output_dir / 'subgroup_metrics.tsv',
        subgroup_rows,
        list(subgroup_rows[0].keys()),
    )
    calibrated_subgroup_rows = flatten_subgroups(calibrated_report)
    write_tsv(
        output_dir / 'subgroup_metrics_calibrated.tsv',
        calibrated_subgroup_rows,
        list(calibrated_subgroup_rows[0].keys()),
    )

    overall = report['overall']
    print('\nOverall HerbOnly - baseline:')
    for metric, value in overall['herb_total_minus_baseline'].items():
        print('  %s: %s' % (metric, markdown_value(value)))
    print('\nInner-validation F1 thresholds:')
    print('  Baseline: %.6f' % calibration['baseline']['threshold'])
    print('  HerbOnly: %.6f' % calibration['herb']['threshold'])
    print('Calibrated outer-test HerbOnly - baseline:')
    for metric, value in calibrated_report['overall'][
            'herb_total_minus_baseline'].items():
        print('  %s: %s' % (metric, markdown_value(value)))
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
