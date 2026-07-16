#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

METRIC_NAMES = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Restore one checkpoint per Strict fold, select an F1 threshold only '
            'from that fold inner-validation records, and evaluate outer-test pairs.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--checkpoint',
        action='append',
        required=True,
        help='Checkpoint prefix or directory, repeated once in one-based fold order.',
    )
    parser.add_argument('--output-dir')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate the split and checkpoint list without importing TensorFlow.',
    )
    return parser.parse_args()


def records_sha256(records):
    import hashlib

    rows = [
        '%s\t%s\t%d' % (str(left), str(right), int(float(label) > 0))
        for left, right, label in records
    ]
    return hashlib.sha256(
        ('\n'.join(sorted(rows)) + '\n').encode('utf-8')
    ).hexdigest()


def summarize_fold_metrics(fold_results, result_key):
    summary = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [row[result_key][metric_name] for row in fold_results],
            dtype=np.float64,
        )
        summary[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def format_summary(summary):
    return '\n'.join(
        '%s:%.6f(+-%.6f)' %
        (metric_name, summary[metric_name]['mean'], summary[metric_name]['std'])
        for metric_name in METRIC_NAMES
    )


def build_markdown(payload):
    lines = [
        '# Checkpoint Fold Threshold Calibration',
        '',
        '- Config: `%s`' % payload['config_path'],
        '- Split strategy: `%s`' % payload['split_strategy'],
        '- Threshold selection: inner-validation F1; outer-test is evaluation only.',
        '- Training/optimizer steps: 0.',
        '',
        '## Fold Results',
        '',
        '| Fold | Best threshold | Validation F1 | Fixed AUC | Fixed AUPR | Fixed F1 | Calibrated Recall | Calibrated Precision | Calibrated F1 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in payload['fold_results']:
        lines.append(
            '| {fold} | {threshold:.6f} | {validation_f1:.6f} | '
            '{fixed_auc:.6f} | {fixed_aupr:.6f} | {fixed_f1:.6f} | '
            '{calibrated_recall:.6f} | {calibrated_precision:.6f} | '
            '{calibrated_f1:.6f} |'.format(
                fold=row['fold'],
                threshold=row['threshold'],
                validation_f1=row['validation_metrics']['F1-score'],
                fixed_auc=row['fixed_metrics']['AUC'],
                fixed_aupr=row['fixed_metrics']['AUPR'],
                fixed_f1=row['fixed_metrics']['F1-score'],
                calibrated_recall=row['calibrated_metrics']['Recall'],
                calibrated_precision=row['calibrated_metrics']['Precision'],
                calibrated_f1=row['calibrated_metrics']['F1-score'],
            )
        )
    lines.extend([
        '',
        '## Fixed Threshold Summary',
        '',
        '```text',
        format_summary(payload['fixed_summary']),
        '```',
        '',
        '## Calibrated Threshold Summary',
        '',
        '```text',
        format_summary(payload['calibrated_summary']),
        '```',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()

    from util.config import ModelConf, OptionConf
    from util.model_components import resolve_early_stopping
    from tools.analyze_context_subgroups import (
        checkpoint_audit,
        normalize_checkpoint,
        prepare_protocol,
        protocol_audit,
    )

    config_path = Path(args.config).expanduser().resolve()
    conf = ModelConf(str(config_path))
    protocol_name = (
        conf['experiment.protocol'].strip().lower()
        if conf.contains('experiment.protocol') else 'legacy'
    )
    if protocol_name != 'strict':
        raise ValueError('Fold calibration requires experiment.protocol=strict.')
    early_stopping = resolve_early_stopping(conf)
    if not early_stopping['enabled']:
        raise ValueError('Fold calibration requires a fixed inner-validation split.')
    evaluation = OptionConf(conf['evaluation.setup'])
    if not evaluation.contains('-cv'):
        raise ValueError('Fold calibration requires evaluation.setup=-cv K.')
    fold_count = int(evaluation['-cv'])
    if len(args.checkpoint) != fold_count:
        raise ValueError(
            'Expected %d checkpoints in fold order; received %d.' %
            (fold_count, len(args.checkpoint))
        )

    protocols = [
        prepare_protocol(config_path, fold) for fold in range(1, fold_count + 1)
    ]
    checkpoints = [normalize_checkpoint(value) for value in args.checkpoint]
    metadata = {
        'evaluation_type': 'strict_fold_checkpoint_threshold_calibration',
        'created_at': datetime.now().astimezone().isoformat(),
        'config_path': str(config_path),
        'model_variant': (
            conf['model.variant'] if conf.contains('model.variant') else conf['model.name']
        ),
        'split_strategy': protocols[0]['manifest'].get(
            'split_strategy', 'pair_stratified'
        ),
        'fold_count': fold_count,
        'folds': [],
    }
    for fold, (protocol, checkpoint) in enumerate(
            zip(protocols, checkpoints), start=1):
        prefix, data_files = checkpoint
        metadata['folds'].append({
            'fold': fold,
            'protocol': protocol_audit(protocol),
            'checkpoint': checkpoint_audit(prefix, data_files),
            'validation_records_sha256': records_sha256(protocol['validation']),
            'outer_test_records_sha256': records_sha256(protocol['outer_test']),
        })

    print('Strict fold checkpoint threshold calibration')
    print('  config: %s' % config_path)
    print('  folds/checkpoints: %d/%d' % (fold_count, len(checkpoints)))
    print('  split strategy: %s' % metadata['split_strategy'])
    print('  optimizer/training steps: disabled')
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    from util.context_subgroups import binary_metrics, select_f1_threshold
    from util.gpu import configure_cuda_environment
    from util.reproducibility import set_global_seed
    from tools.analyze_context_subgroups import restore_snapshot, score_snapshot
    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI

    configure_cuda_environment(conf)
    fold_results = []
    for fold, (protocol, checkpoint) in enumerate(
            zip(protocols, checkpoints), start=1):
        prefix, _ = checkpoint
        snapshot = restore_snapshot(
            tf, HDCTI, set_global_seed, protocol, prefix, fold
        )
        include_context = any(bool(value) for value in snapshot['context_terms'].values())
        _, validation_logits = score_snapshot(
            snapshot, protocol['validation'], include_context=include_context
        )
        _, test_logits = score_snapshot(
            snapshot, protocol['outer_test'], include_context=include_context
        )
        validation_labels = np.asarray(
            [int(float(row[2]) > 0) for row in protocol['validation']],
            dtype=np.int32,
        )
        test_labels = np.asarray(
            [int(float(row[2]) > 0) for row in protocol['outer_test']],
            dtype=np.int32,
        )
        threshold_info = select_f1_threshold(validation_labels, validation_logits)
        fixed_metrics = binary_metrics(test_labels, test_logits, threshold=0.5)
        calibrated_metrics = binary_metrics(
            test_labels, test_logits, threshold=threshold_info['threshold']
        )
        result = {
            'fold': fold,
            'checkpoint': str(prefix),
            'context_enabled': include_context,
            'threshold': threshold_info['threshold'],
            'validation_metrics': threshold_info['validation_metrics'],
            'fixed_metrics': fixed_metrics,
            'calibrated_metrics': calibrated_metrics,
        }
        fold_results.append(result)
        print(
            '  fold %d threshold=%.6f fixed_F1=%.6f calibrated_F1=%.6f' %
            (fold, result['threshold'], fixed_metrics['F1-score'],
             calibrated_metrics['F1-score'])
        )

    payload = dict(metadata)
    payload['fold_results'] = fold_results
    payload['fixed_summary'] = summarize_fold_metrics(fold_results, 'fixed_metrics')
    payload['calibrated_summary'] = summarize_fold_metrics(
        fold_results, 'calibrated_metrics'
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'checkpoint_calibration' / metadata['model_variant']
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'report.md').write_text(
        build_markdown(payload), encoding='utf-8'
    )
    print('\nFixed threshold summary:\n%s' % format_summary(payload['fixed_summary']))
    print(
        '\nCalibrated threshold summary:\n%s' %
        format_summary(payload['calibrated_summary'])
    )
    print('\nResults written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
