#!/usr/bin/env python3
"""Train an isolated Hctx-P head on one frozen NoContext checkpoint."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.audit_four_state_cc_residual import restore_model  # noqa: E402
from tools.evaluate_four_state_checkpoint import (  # noqa: E402
    STATE_NAMES,
    compare_to_baseline,
    normalize_checkpoint,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Keep a NoContext checkpoint frozen, train only a deterministic '
            'Hctx-P head, and evaluate fixed four-state routing.'
        )
    )
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def checkpoint_hashes(prefix):
    paths = sorted(prefix.parent.glob(prefix.name + '.*'))
    if not paths:
        raise FileNotFoundError('Checkpoint files not found: %s' % prefix)
    return {str(path): sha256_file(path) for path in paths}


def records_to_arrays(model, state, records):
    compound_indices = np.asarray([
        model.data.compound[str(row[0])] for row in records
    ], dtype=np.int64)
    protein_indices = np.asarray([
        model.data.protein[str(row[1])] for row in records
    ], dtype=np.int64)
    labels = np.asarray([
        1 if float(row[2]) > 0 else 0 for row in records
    ], dtype=np.float64)
    compounds = state['compound'][compound_indices]
    proteins = state['protein'][protein_indices]
    contexts = state['compound_context'][compound_indices]
    return {
        'compound_indices': compound_indices,
        'protein_indices': protein_indices,
        'labels': labels,
        'base_logits': np.sum(compounds * proteins, axis=1),
        'hctx_p_features': contexts * proteins,
    }


def binary_metrics(labels, logits):
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return {
        'AUPR': float(average_precision_score(labels, scores)),
        'AUC': float(roc_auc_score(labels, scores)),
        'records': int(labels.size),
    }


def evaluate_router(state_arrays, head, preserved_metrics=None):
    metrics = {}
    for state_name in STATE_NAMES:
        arrays = state_arrays[state_name]
        if preserved_metrics and state_name in preserved_metrics:
            metrics[state_name] = dict(preserved_metrics[state_name])
            continue
        hp_logits = arrays['hctx_p_features'].dot(head)
        if state_name == 'warm_warm':
            logits = arrays['base_logits'] + hp_logits
        elif state_name == 'cold_warm':
            logits = hp_logits
        else:
            logits = arrays['base_logits']
        metrics[state_name] = binary_metrics(arrays['labels'], logits)
    metrics['macro'] = {
        metric: float(np.mean([
            metrics[name][metric] for name in STATE_NAMES
        ]))
        for metric in ('AUPR', 'AUC')
    }
    return metrics


def recompute_frozen_base_metrics(state_arrays):
    metrics = {
        name: binary_metrics(
            state_arrays[name]['labels'],
            state_arrays[name]['base_logits'],
        )
        for name in STATE_NAMES
    }
    metrics['macro'] = {
        metric: float(np.mean([
            metrics[name][metric] for name in STATE_NAMES
        ]))
        for metric in ('AUPR', 'AUC')
    }
    return metrics


def verify_baseline_metrics(recomputed, reported, tolerance=1e-4):
    differences = {}
    for state_name in STATE_NAMES + ('macro',):
        differences[state_name] = {}
        for metric in ('AUPR', 'AUC'):
            difference = abs(
                recomputed[state_name][metric]
                - reported[state_name][metric]
            )
            differences[state_name][metric] = float(difference)
            if difference > tolerance:
                raise ValueError(
                    'Frozen base %s %s differs from baseline report by %.12g.'
                    % (state_name, metric, difference)
                )
    return differences


def train_head(
        features,
        labels,
        validation_arrays,
        preserved_metrics,
        settings):
    dimensions = features.shape[1]
    head = np.zeros(dimensions, dtype=np.float64)
    first_moment = np.zeros_like(head)
    second_moment = np.zeros_like(head)
    rng = np.random.RandomState(int(settings['seed']))
    learning_rate = float(settings['learning_rate'])
    regularization = float(settings['l2'])
    batch_size = int(settings['batch_size'])
    max_epochs = int(settings['max_epochs'])
    interval = int(settings['validation_interval'])
    patience = int(settings['patience'])
    min_delta = float(settings['min_delta'])
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    optimizer_steps = 0
    best_metric = float('-inf')
    best_epoch = 0
    best_head = head.copy()
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(labels.size)
        for start in range(0, labels.size, batch_size):
            indices = order[start:start + batch_size]
            batch_features = features[indices]
            batch_labels = labels[indices]
            logits = batch_features.dot(head)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(logits, -50, 50))
            )
            gradient = (
                batch_features.T.dot(probabilities - batch_labels)
                / batch_labels.size
                + regularization * head
            )
            optimizer_steps += 1
            first_moment = (
                beta1 * first_moment + (1.0 - beta1) * gradient
            )
            second_moment = (
                beta2 * second_moment
                + (1.0 - beta2) * np.square(gradient)
            )
            corrected_first = first_moment / (
                1.0 - beta1 ** optimizer_steps
            )
            corrected_second = second_moment / (
                1.0 - beta2 ** optimizer_steps
            )
            head -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )

        if epoch % interval != 0 and epoch != max_epochs:
            continue
        training_logits = features.dot(head)
        training_loss = float(np.mean(
            np.logaddexp(0.0, training_logits)
            - labels * training_logits
        ) + 0.5 * regularization * head.dot(head))
        metrics = evaluate_router(
            validation_arrays,
            head,
            preserved_metrics=preserved_metrics,
        )
        value = metrics['macro']['AUPR']
        improved = value > best_metric + min_delta
        if improved:
            best_metric = value
            best_epoch = epoch
            best_head = head.copy()
            stale = 0
        else:
            stale += 1
        history.append({
            'epoch': epoch,
            'training_head_loss': training_loss,
            'macro_aupr': value,
            'warm_warm_aupr': metrics['warm_warm']['AUPR'],
            'cold_warm_aupr': metrics['cold_warm']['AUPR'],
            'stale': stale,
            'improved': improved,
        })
        print(
            'head validation: epoch %d macro-AUPR=%.6f '
            'WW=%.6f CW=%.6f best=%.6f best_epoch=%d stale=%d/%d%s'
            % (
                epoch,
                value,
                metrics['warm_warm']['AUPR'],
                metrics['cold_warm']['AUPR'],
                best_metric,
                best_epoch,
                stale,
                patience,
                ' improved' if improved else '',
            )
        )
        if stale >= patience:
            print('Frozen-head early stopping triggered at epoch %d.' % epoch)
            break

    return {
        'head': best_head,
        'best_epoch': best_epoch,
        'best_macro_aupr': best_metric,
        'optimizer_steps': optimizer_steps,
        'history': history,
    }


def markdown_report(report):
    lines = [
        '# Frozen-Base Hctx-P Router Pilot',
        '',
        '- Dataset: `%s`' % report['dataset'],
        '- Protocol: `%s`' % report['protocol'],
        '- Base checkpoint: `%s`' % report['base_checkpoint']['prefix'],
        '- Base model optimizer steps: `0`',
        '- Hctx-P head optimizer steps: `%d`' %
        report['head_training']['optimizer_steps'],
        '- Best head epoch: `%d`' % report['head_training']['best_epoch'],
        '- Assignment hash: `%s`' %
        report['support_unit']['assignments_sha256'],
        '',
        '| State | Baseline AUPR | V3 AUPR | Delta | Baseline AUC | '
        'V3 AUC | Delta |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in STATE_NAMES + ('macro',):
        baseline = report['baseline_metrics'][name]
        candidate = report['metrics'][name]
        delta = report['comparison']['deltas'][name]
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %.6f | %.6f | %+.6f |'
            % (
                name,
                baseline['AUPR'],
                candidate['AUPR'],
                delta['AUPR'],
                baseline['AUC'],
                candidate['AUC'],
                delta['AUC'],
            )
        )
    lines.extend([
        '',
        'Gate: `%s`' % (
            'PASS' if report['comparison']['passed'] else 'FAIL'
        ),
        '',
        '- Frozen WC exact preservation: `%s`' %
        report['preservation_checks']['warm_cold_exact'],
        '- Frozen CC exact preservation: `%s`' %
        report['preservation_checks']['cold_cold_exact'],
        '- Base checkpoint hashes unchanged: `%s`' %
        report['preservation_checks']['checkpoint_hashes_unchanged'],
        '',
        'This is an inner-validation pilot. No outer-test records were used.',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    manifest = load_json(manifest_path)
    if manifest.get('protocol') != 'frozen_base_hctx_router_pilot_v3':
        raise ValueError('Unexpected frozen-head manifest protocol.')
    if args.dataset not in manifest['datasets']:
        raise ValueError('Dataset is absent from manifest: %s' % args.dataset)
    dataset_spec = manifest['datasets'][args.dataset]
    settings = manifest['head_training']
    config_path = resolve_path(dataset_spec['config'])
    baseline_report_path = resolve_path(dataset_spec['baseline_report'])
    checkpoint = normalize_checkpoint(dataset_spec['checkpoint'])
    baseline_report = load_json(baseline_report_path)

    if baseline_report['config']['sha256'] != sha256_file(config_path):
        raise ValueError('Baseline report/config hash mismatch.')
    if baseline_report['checkpoint']['prefix'] != str(checkpoint):
        raise ValueError('Manifest and baseline report checkpoints differ.')
    if (
        baseline_report['checkpoint']['index_sha256']
        != dataset_spec['checkpoint_index_sha256']
    ):
        raise ValueError('Manifest and baseline checkpoint hashes differ.')
    if sha256_file(Path(str(checkpoint) + '.index')) != (
            dataset_spec['checkpoint_index_sha256']):
        raise ValueError('Current checkpoint index hash differs from manifest.')
    expected_checkpoint_files = dataset_spec.get('checkpoint_files')
    if (
            expected_checkpoint_files is not None
            and checkpoint_hashes(checkpoint) != expected_checkpoint_files):
        raise ValueError(
            'Current checkpoint files differ from the frozen manifest.'
        )
    if baseline_report['support_unit']['assignments_sha256'] != (
            dataset_spec['assignments_sha256']):
        raise ValueError('Manifest and baseline support assignments differ.')

    hashes_before = checkpoint_hashes(checkpoint)
    conf, experiment, model = restore_model(config_path, checkpoint)
    if experiment.supportUnitMetadata['assignments_sha256'] != (
            dataset_spec['assignments_sha256']):
        model.sess.close()
        raise ValueError('Loaded support assignments differ from manifest.')
    state = model.fetchModelState()
    training_arrays = records_to_arrays(
        model, state, model.data.trainingData
    )
    validation_arrays = {
        name: records_to_arrays(
            model,
            state,
            experiment.supportValidationDataByState[name],
        )
        for name in STATE_NAMES
    }
    recomputed_baseline = recompute_frozen_base_metrics(validation_arrays)
    baseline_differences = verify_baseline_metrics(
        recomputed_baseline, baseline_report['metrics']
    )

    compound_support = model.compound_cp_support_degrees[
        training_arrays['compound_indices']
    ]
    protein_support = model.protein_cp_support_degrees[
        training_arrays['protein_indices']
    ]
    if np.any(compound_support <= 0) or np.any(protein_support <= 0):
        model.sess.close()
        raise ValueError('Second-stage training records are not all warm-warm.')

    preserved_metrics = {
        name: baseline_report['metrics'][name]
        for name in ('warm_cold', 'cold_cold')
    }
    trained = train_head(
        training_arrays['hctx_p_features'].astype(np.float64),
        training_arrays['labels'],
        validation_arrays,
        preserved_metrics,
        settings,
    )
    metrics = evaluate_router(
        validation_arrays,
        trained['head'],
        preserved_metrics=preserved_metrics,
    )
    model.sess.close()
    hashes_after = checkpoint_hashes(checkpoint)

    comparison = compare_to_baseline(
        metrics, baseline_report['metrics']
    )
    preservation_checks = {
        'warm_cold_exact': (
            metrics['warm_cold'] == baseline_report['metrics']['warm_cold']
        ),
        'cold_cold_exact': (
            metrics['cold_cold'] == baseline_report['metrics']['cold_cold']
        ),
        'checkpoint_hashes_unchanged': hashes_before == hashes_after,
    }
    if not all(preservation_checks.values()):
        raise ValueError('Frozen-base preservation check failed.')

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / 'hctx_p_head.npz'
    np.savez(
        head_path,
        context_herb_protein=trained['head'].astype(np.float32),
    )
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'evaluation': 'frozen_base_hctx_router_inner_validation',
        'protocol': manifest['protocol'],
        'dataset': dataset_spec['display_name'],
        'manifest': {
            'path': str(manifest_path),
            'sha256': sha256_file(manifest_path),
        },
        'config': baseline_report['config'],
        'base_checkpoint': {
            'prefix': str(checkpoint),
            'files_before': hashes_before,
            'files_after': hashes_after,
        },
        'baseline_report': str(baseline_report_path),
        'support_unit': experiment.supportUnitMetadata,
        'inner_validation': experiment.supportInnerValidationMetadata,
        'routing': {
            'warm_warm': 'frozen_base_plus_hctx_p',
            'cold_warm': 'hctx_p_only',
            'warm_cold': 'frozen_base',
            'cold_cold': 'frozen_base',
        },
        'base_model_optimizer_steps': 0,
        'head_training': {
            'settings': settings,
            'best_epoch': trained['best_epoch'],
            'best_macro_aupr': trained['best_macro_aupr'],
            'optimizer_steps': trained['optimizer_steps'],
            'history': trained['history'],
            'weight_mean_abs': float(np.mean(np.abs(trained['head']))),
            'weight_l2_norm': float(np.linalg.norm(trained['head'])),
            'artifact': str(head_path),
            'artifact_sha256': sha256_file(head_path),
        },
        'baseline_recomputation_abs_delta': baseline_differences,
        'baseline_metrics': baseline_report['metrics'],
        'metrics': metrics,
        'comparison': comparison,
        'preservation_checks': preservation_checks,
        'outer_test_used': False,
    }
    report_path = output_dir / 'report.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    markdown = markdown_report(report)
    (output_dir / 'report.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()
