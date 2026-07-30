#!/usr/bin/env python3
"""Evaluate a saved checkpoint on one frozen four-state validation unit."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STATE_NAMES = (
    'warm_warm',
    'cold_warm',
    'warm_cold',
    'cold_cold',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pure-inference four-state checkpoint evaluation.'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline-report')
    return parser.parse_args()


def normalize_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        index_path = path / 'hdcti_model.ckpt.index'
    elif str(path).endswith('.index'):
        index_path = path
    else:
        index_path = Path(str(path) + '.index')
    if not index_path.exists():
        raise FileNotFoundError('Checkpoint index not found: %s' % index_path)
    return Path(str(index_path)[:-len('.index')])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_checkpoint(config_path, checkpoint_prefix):
    from util.config import ModelConf, OptionConf
    from util.gpu import configure_cuda_environment
    from util.reproducibility import set_global_seed

    conf = ModelConf(str(config_path))
    evaluation = OptionConf(conf['evaluation.setup'])
    if not evaluation.contains('-four-state-unit'):
        raise ValueError(
            'Checkpoint evaluation requires evaluation.setup=-four-state-unit.'
        )
    configure_cuda_environment(conf)

    from HDR import HDR

    experiment = HDR(conf)
    seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
    set_global_seed(seed, reset_tensorflow_graph=True)

    import tensorflow.compat.v1 as tf
    from HDCTI import HDCTI

    model = HDCTI(
        conf,
        experiment.trainingData,
        experiment.testData,
        '[1]',
    )
    model.validationData = experiment.supportValidationData
    model.validationDataByState = experiment.supportValidationDataByState
    model.validationAggregation = 'macro_support_states'
    model.readConfiguration()
    model.initModel()

    checkpoint_variables = dict(
        tf.train.list_variables(str(checkpoint_prefix))
    )
    graph_variables = {
        variable.name.split(':', 1)[0]: variable
        for variable in tf.global_variables()
    }
    missing = sorted(set(graph_variables) - set(checkpoint_variables))
    mismatched = []
    for name, variable in graph_variables.items():
        if name not in checkpoint_variables:
            continue
        graph_shape = variable.shape.as_list()
        checkpoint_shape = list(checkpoint_variables[name])
        if graph_shape != checkpoint_shape:
            mismatched.append({
                'name': name,
                'graph': graph_shape,
                'checkpoint': checkpoint_shape,
            })
    if missing or mismatched:
        model.sess.close()
        raise ValueError(
            'Checkpoint/config mismatch: missing=%s mismatched=%s.' %
            (missing, mismatched)
        )

    saver = tf.train.Saver(var_list=graph_variables)
    saver.restore(model.sess, str(checkpoint_prefix))
    state = model.fetchModelState()
    metrics = {}
    for state_name in STATE_NAMES:
        records = experiment.supportValidationDataByState[state_name]
        metrics[state_name] = {
            'AUPR': model.evaluateValidation(
                state, 'aupr', validation_records=records
            ),
            'AUC': model.evaluateValidation(
                state, 'auc', validation_records=records
            ),
            'records': len(records),
        }
    model.sess.close()
    metrics['macro'] = {
        metric: sum(metrics[name][metric] for name in STATE_NAMES)
        / len(STATE_NAMES)
        for metric in ('AUPR', 'AUC')
    }
    return conf, experiment, metrics


def compare_to_baseline(metrics, baseline):
    deltas = {
        name: {
            metric: metrics[name][metric] - baseline[name][metric]
            for metric in ('AUPR', 'AUC')
        }
        for name in STATE_NAMES + ('macro',)
    }
    gate_checks = {
        'macro_aupr_delta_at_least_0.005': (
            deltas['macro']['AUPR'] >= 0.005
        ),
        'cold_cold_aupr_not_lower': (
            deltas['cold_cold']['AUPR'] >= 0.0
        ),
        'no_state_aupr_drop_over_0.020': all(
            deltas[name]['AUPR'] >= -0.020 for name in STATE_NAMES
        ),
    }
    return {
        'deltas': deltas,
        'gate_checks': gate_checks,
        'passed': all(gate_checks.values()),
    }


def markdown_report(report):
    lines = [
        '# Four-State Checkpoint Evaluation',
        '',
        '- Config: `%s`' % report['config']['path'],
        '- Checkpoint: `%s`' % report['checkpoint']['prefix'],
        '- Assignment hash: `%s`' %
        report['support_unit']['assignments_sha256'],
        '- Training/optimizer steps: `0`',
        '',
        '| State | AUPR | AUC | Records |',
        '|---|---:|---:|---:|',
    ]
    for name in STATE_NAMES:
        row = report['metrics'][name]
        lines.append(
            '| %s | %.6f | %.6f | %d |' %
            (name, row['AUPR'], row['AUC'], row['records'])
        )
    lines.append(
        '| macro | %.6f | %.6f | - |' % (
            report['metrics']['macro']['AUPR'],
            report['metrics']['macro']['AUC'],
        )
    )
    comparison = report.get('comparison')
    if comparison:
        lines.extend([
            '',
            '## Baseline Delta',
            '',
            '| State | Delta AUPR | Delta AUC |',
            '|---|---:|---:|',
        ])
        for name in STATE_NAMES + ('macro',):
            row = comparison['deltas'][name]
            lines.append(
                '| %s | %+.6f | %+.6f |' %
                (name, row['AUPR'], row['AUC'])
            )
        lines.extend([
            '',
            'Gate: `%s`' % (
                'PASS' if comparison['passed'] else 'FAIL'
            ),
        ])
        for name, passed in comparison['gate_checks'].items():
            lines.append('- `%s`: `%s`' % (name, passed))
    lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_prefix = normalize_checkpoint(args.checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conf, experiment, metrics = evaluate_checkpoint(
        config_path, checkpoint_prefix
    )
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'evaluation': 'four_state_checkpoint_pure_inference',
        'config': {
            'path': str(config_path),
            'sha256': sha256_file(config_path),
            'model_variant': conf['model.variant'],
        },
        'checkpoint': {
            'prefix': str(checkpoint_prefix),
            'index_sha256': sha256_file(
                Path(str(checkpoint_prefix) + '.index')
            ),
        },
        'support_unit': experiment.supportUnitMetadata,
        'inner_validation': experiment.supportInnerValidationMetadata,
        'metrics': metrics,
    }
    if args.baseline_report:
        with open(args.baseline_report, encoding='utf-8') as handle:
            baseline_report = json.load(handle)
        if (
            baseline_report['support_unit']['assignments_sha256']
            != report['support_unit']['assignments_sha256']
        ):
            raise ValueError(
                'Baseline and candidate use different four-state assignments.'
            )
        report['comparison'] = compare_to_baseline(
            metrics, baseline_report['metrics']
        )

    json_path = output_dir / 'report.json'
    markdown_path = output_dir / 'report.md'
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    markdown_path.write_text(markdown_report(report), encoding='utf-8')
    print(markdown_report(report))
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()
