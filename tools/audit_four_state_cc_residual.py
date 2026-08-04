#!/usr/bin/env python3
"""Pure-inference audit of residual routing for cold-cold pairs."""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_four_state_checkpoint import (  # noqa: E402
    STATE_NAMES,
    compare_to_baseline,
    normalize_checkpoint,
    sha256_file,
)
from util.model_components import context_interaction_pair_scores  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit base + alpha * Hctx-Dctx on frozen CC records.'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidate-report', required=True)
    parser.add_argument('--baseline-report', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--alphas',
        default='0,0.25,0.5,1.0',
        help='Predeclared non-negative residual coefficients.',
    )
    return parser.parse_args()


def parse_alphas(value):
    alphas = []
    for item in value.split(','):
        alpha = float(item.strip())
        if alpha < 0:
            raise ValueError('Residual coefficients cannot be negative.')
        if alpha not in alphas:
            alphas.append(alpha)
    if not alphas:
        raise ValueError('At least one residual coefficient is required.')
    return alphas


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def restore_model(config_path, checkpoint_prefix):
    from util.config import ModelConf, OptionConf
    from util.gpu import configure_cuda_environment
    from util.reproducibility import set_global_seed

    conf = ModelConf(str(config_path))
    evaluation = OptionConf(conf['evaluation.setup'])
    if not evaluation.contains('-four-state-unit'):
        raise ValueError(
            'CC residual audit requires evaluation.setup=-four-state-unit.'
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
    tf.train.Saver(var_list=graph_variables).restore(
        model.sess, str(checkpoint_prefix)
    )
    return conf, experiment, model


def cc_indices_and_labels(model, records):
    compound_indices = []
    protein_indices = []
    labels = []
    for compound_id, protein_id, label in records:
        compound_indices.append(model.data.compound[str(compound_id)])
        protein_indices.append(model.data.protein[str(protein_id)])
        labels.append(1 if float(label) > 0 else 0)
    return (
        np.asarray(compound_indices, dtype=np.int64),
        np.asarray(protein_indices, dtype=np.int64),
        np.asarray(labels, dtype=np.int32),
    )


def score_cc_route(model, state, records, base_scale, context_scale):
    compound_indices, protein_indices, labels = cc_indices_and_labels(
        model, records
    )
    pair_count = len(records)
    zero_weight = np.zeros(model.emb_size, dtype=np.float32)
    logits = context_interaction_pair_scores(
        state['compound'],
        state['protein'],
        state['compound_context'],
        state['protein_context'],
        compound_indices,
        protein_indices,
        zero_weight,
        zero_weight,
        state['weights']['context_herb_disease'],
        enabled_terms={
            'compound_disease': False,
            'herb_protein': False,
            'herb_disease': True,
        },
        decoder_type='dot',
        decoder_weights=state['weights'],
        base_score_scale=np.full(
            pair_count, base_scale, dtype=np.float64
        ),
        herb_disease_scale=np.full(
            pair_count, context_scale, dtype=np.float64
        ),
    )
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return {
        'AUPR': float(average_precision_score(labels, scores)),
        'AUC': float(roc_auc_score(labels, scores)),
        'records': pair_count,
    }


def herb_disease_feature_diagnostics(model, state, records):
    compound_indices, protein_indices, labels = cc_indices_and_labels(
        model, records
    )
    features = (
        state['compound_context'][compound_indices]
        * state['protein_context'][protein_indices]
    )
    norms = np.linalg.norm(features, axis=1)
    logits = features.dot(state['weights']['context_herb_disease'])
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    positive = labels == 1
    negative = labels == 0
    return {
        'records': len(records),
        'nonzero_feature_fraction': float(np.mean(norms > 1e-12)),
        'mean_feature_norm': float(np.mean(norms)),
        'positive_mean_feature_norm': float(np.mean(norms[positive])),
        'negative_mean_feature_norm': float(np.mean(norms[negative])),
        'mean_abs_logit': float(np.mean(np.abs(logits))),
        'AUPR': float(average_precision_score(labels, scores)),
    }


def metrics_with_cc(candidate_metrics, cc_metrics):
    metrics = copy.deepcopy(candidate_metrics)
    metrics['cold_cold'] = cc_metrics
    metrics['macro'] = {
        metric: sum(metrics[name][metric] for name in STATE_NAMES)
        / len(STATE_NAMES)
        for metric in ('AUPR', 'AUC')
    }
    return metrics


def audit_routes(
        candidate_metrics,
        baseline_metrics,
        hd_only_metrics,
        residual_metrics):
    rows = []

    def add_row(name, alpha, cc_metrics):
        metrics = metrics_with_cc(candidate_metrics, cc_metrics)
        comparison = compare_to_baseline(metrics, baseline_metrics)
        rows.append({
            'route': name,
            'alpha': alpha,
            'cold_cold': cc_metrics,
            'macro': metrics['macro'],
            'comparison': comparison,
        })

    add_row('hctx_dctx_only', None, hd_only_metrics)
    for alpha, metrics in residual_metrics:
        route = 'base_only' if alpha == 0 else 'base_plus_context'
        add_row(route, alpha, metrics)
    return rows


def markdown_report(report):
    lines = [
        '# Cold-Cold Residual Routing Audit',
        '',
        '- Dataset: `%s`' % report['dataset'],
        '- Checkpoint: `%s`' % report['checkpoint']['prefix'],
        '- Assignment hash: `%s`' %
        report['support_unit']['assignments_sha256'],
        '- Training/optimizer steps: `0`',
        '- Audit status: feasibility only; no coefficient selected',
        '- Training Hctx-Dctx nonzero feature fraction: `%.6f`' %
        report['feature_diagnostics']['training'][
            'nonzero_feature_fraction'
        ],
        '- CC Hctx-Dctx nonzero feature fraction: `%.6f`' %
        report['feature_diagnostics']['cold_cold'][
            'nonzero_feature_fraction'
        ],
        '',
        '| Route | Alpha | CC AUPR | CC Delta | Macro-AUPR | '
        'Macro Delta | Gate |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in report['routes']:
        comparison = row['comparison']
        lines.append(
            '| %s | %s | %.6f | %+.6f | %.6f | %+.6f | %s |' % (
                row['route'],
                '-' if row['alpha'] is None else ('%g' % row['alpha']),
                row['cold_cold']['AUPR'],
                comparison['deltas']['cold_cold']['AUPR'],
                row['macro']['AUPR'],
                comparison['deltas']['macro']['AUPR'],
                'PASS' if comparison['passed'] else 'FAIL',
            )
        )
    lines.extend([
        '',
        'The alpha grid was declared before this audit. These values are '
        'diagnostic and must not be reported as outer-test model selection.',
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_prefix = normalize_checkpoint(args.checkpoint)
    candidate_report_path = Path(
        args.candidate_report
    ).expanduser().resolve()
    baseline_report_path = Path(args.baseline_report).expanduser().resolve()
    candidate_report = load_json(candidate_report_path)
    baseline_report = load_json(baseline_report_path)

    if (
        candidate_report['support_unit']['assignments_sha256']
        != baseline_report['support_unit']['assignments_sha256']
    ):
        raise ValueError('Candidate and baseline assignments differ.')
    if candidate_report['config']['sha256'] != sha256_file(config_path):
        raise ValueError('Candidate report/config hash mismatch.')
    checkpoint_index = Path(str(checkpoint_prefix) + '.index')
    if (
        candidate_report['checkpoint']['index_sha256']
        != sha256_file(checkpoint_index)
    ):
        raise ValueError('Candidate report/checkpoint hash mismatch.')

    conf, experiment, model = restore_model(
        config_path, checkpoint_prefix
    )
    state = model.fetchModelState()
    records = experiment.supportValidationDataByState['cold_cold']
    hd_only = score_cc_route(model, state, records, 0.0, 1.0)
    reported_cc = candidate_report['metrics']['cold_cold']
    recomputed_current_route_abs_delta = {}
    for metric in ('AUPR', 'AUC'):
        difference = abs(hd_only[metric] - reported_cc[metric])
        recomputed_current_route_abs_delta[metric] = difference
        if difference > 1e-4:
            model.sess.close()
            raise ValueError(
                'Recomputed Hctx-Dctx-only %s differs from candidate report: '
                'recomputed=%.12f reported=%.12f abs_delta=%.12g.'
                % (
                    metric,
                    hd_only[metric],
                    reported_cc[metric],
                    difference,
                )
            )

    residual_metrics = [
        (
            alpha,
            score_cc_route(model, state, records, 1.0, alpha),
        )
        for alpha in parse_alphas(args.alphas)
    ]
    feature_diagnostics = {
        'training': herb_disease_feature_diagnostics(
            model, state, model.data.trainingData
        ),
        'cold_cold': herb_disease_feature_diagnostics(
            model, state, records
        ),
    }
    model.sess.close()

    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'audit': 'four_state_cold_cold_residual_pure_inference',
        'dataset': args.dataset,
        'config': {
            'path': str(config_path),
            'sha256': sha256_file(config_path),
            'model_variant': conf['model.variant'],
        },
        'checkpoint': {
            'prefix': str(checkpoint_prefix),
            'index_sha256': sha256_file(checkpoint_index),
        },
        'candidate_report': str(candidate_report_path),
        'baseline_report': str(baseline_report_path),
        'support_unit': experiment.supportUnitMetadata,
        'inner_validation': experiment.supportInnerValidationMetadata,
        'declared_alphas': parse_alphas(args.alphas),
        'recomputed_current_route_abs_delta': (
            recomputed_current_route_abs_delta
        ),
        'feature_diagnostics': feature_diagnostics,
        'routes': audit_routes(
            candidate_report['metrics'],
            baseline_report['metrics'],
            hd_only,
            residual_metrics,
        ),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    markdown = markdown_report(report)
    (output_dir / 'report.md').write_text(markdown, encoding='utf-8')
    print(markdown)
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()
