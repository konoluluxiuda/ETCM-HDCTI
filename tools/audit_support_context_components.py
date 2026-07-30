#!/usr/bin/env python3
"""Decompose a support-unit checkpoint into base and context scores."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Restore one support-unit checkpoint and evaluate base-only, '
            'context-only, and combined inner-validation scores without training.'
        )
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir')
    return parser.parse_args()


def normalize_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        preferred = path / 'hdcti_model.ckpt.index'
        index_path = preferred if preferred.exists() else None
        if index_path is None:
            index_files = sorted(path.glob('*.index'))
            if len(index_files) != 1:
                raise FileNotFoundError(
                    'Expected one checkpoint index in %s, found %d.' %
                    (path, len(index_files))
                )
            index_path = index_files[0]
        path = index_path
    text = str(path)
    for suffix in ('.index', '.meta'):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    if '.data-' in text:
        text = text.split('.data-', 1)[0]
    prefix = Path(text)
    if not Path(str(prefix) + '.index').exists():
        raise FileNotFoundError('Checkpoint index not found: %s.index' % prefix)
    return prefix


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def metric_summary(labels, logits):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int32)
    logits = np.asarray(logits, dtype=np.float64)
    scores = sigmoid(logits)
    positive = logits[labels == 1]
    negative = logits[labels == 0]
    return {
        'AUC': float(roc_auc_score(labels, scores)),
        'AUPR': float(average_precision_score(labels, scores)),
        'positive_logit_mean': float(np.mean(positive)),
        'negative_logit_mean': float(np.mean(negative)),
        'positive_negative_logit_gap': float(
            np.mean(positive) - np.mean(negative)
        ),
        'logit_mean': float(np.mean(logits)),
        'logit_std': float(np.std(logits)),
    }


def markdown_report(report):
    rows = []
    for name in ('base_only', 'context_only', 'base_plus_context'):
        metric = report['metrics'][name]
        rows.append(
            '| %s | %.6f | %.6f | %.6f | %.6f | %.6f |' % (
                name,
                metric['AUC'],
                metric['AUPR'],
                metric['positive_logit_mean'],
                metric['negative_logit_mean'],
                metric['positive_negative_logit_gap'],
            )
        )
    return '\n'.join([
        '# Support Context Component Audit',
        '',
        '- Config: `%s`' % report['config']['path'],
        '- Checkpoint: `%s`' % report['checkpoint']['prefix'],
        '- Unit: `%s`' % report['support_unit']['unit_key'],
        '- Validation records: `%d`' % report['validation']['records'],
        '- Active context term: `%s`' % report['active_context_term'],
        '- Training/optimizer steps: `0`',
        '- Outer-test evaluation: `disabled`',
        '',
        '| Score | AUC | AUPR | Positive mean | Negative mean | Logit gap |',
        '|---|---:|---:|---:|---:|---:|',
        *rows,
        '',
        '## Diagnostics',
        '',
        '- Context/base Pearson correlation: `%.6f`' %
        report['diagnostics']['context_base_pearson'],
        '- Context mean absolute logit: `%.6f`' %
        report['diagnostics']['context_mean_abs_logit'],
        '- Total minus base AUPR: `%+.6f`' %
        report['diagnostics']['total_minus_base_aupr'],
        '',
    ])


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_prefix = normalize_checkpoint(args.checkpoint)

    from util.config import ModelConf, OptionConf
    from util.gpu import configure_cuda_environment
    from util.model_components import (
        pair_decoder_scores,
        resolve_context_terms,
        resolve_pair_decoder,
    )
    from util.reproducibility import set_global_seed

    conf = ModelConf(str(config_path))
    evaluation = OptionConf(conf['evaluation.setup'])
    if not evaluation.contains('-support-unit'):
        raise ValueError('This audit requires evaluation.setup=-support-unit.')
    if not conf.contains('evaluation.outer.test') or (
        str(conf['evaluation.outer.test']).strip().lower()
        not in {'0', 'false', 'no', 'off'}
    ):
        raise ValueError('This audit requires evaluation.outer.test=False.')

    context_terms = resolve_context_terms(conf)
    active_terms = [name for name, enabled in context_terms.items() if enabled]
    if active_terms not in (
        ['compound_disease'],
        ['herb_disease'],
    ):
        raise ValueError(
            'Expected exactly C-Dctx or Hctx-Dctx, found %s.' % active_terms
        )
    active_term = active_terms[0]
    pair_decoder = resolve_pair_decoder(conf)
    if pair_decoder['type'] != 'dot':
        raise ValueError('The frozen component audit currently requires dot decoder.')

    configure_cuda_environment(conf)
    from HDR import HDR

    experiment = HDR(conf)
    if not experiment.supportValidationData:
        raise ValueError('Support-state inner validation is empty.')

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
    model.readConfiguration()
    model.initModel()

    checkpoint_variables = dict(tf.train.list_variables(str(checkpoint_prefix)))
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
    model.sess.close()

    validation = experiment.supportValidationData
    compound_indices = np.asarray([
        model.data.compound[str(compound_id)]
        for compound_id, _, _ in validation
    ], dtype=np.int64)
    protein_indices = np.asarray([
        model.data.protein[str(protein_id)]
        for _, protein_id, _ in validation
    ], dtype=np.int64)
    labels = np.asarray([
        int(float(label) > 0) for _, _, label in validation
    ], dtype=np.int32)

    compounds = state['compound'][compound_indices]
    proteins = state['protein'][protein_indices]
    compound_contexts = state['compound_context'][compound_indices]
    protein_contexts = state['protein_context'][protein_indices]
    base_logits = pair_decoder_scores(
        compounds,
        proteins,
        decoder_type='dot',
        decoder_weights=state['weights'],
    )
    if active_term == 'compound_disease':
        context_weight = state['weights']['context_compound_disease']
        context_logits = np.sum(
            compounds * protein_contexts * context_weight,
            axis=1,
        )
    else:
        context_weight = state['weights']['context_herb_disease']
        context_logits = np.sum(
            compound_contexts * protein_contexts * context_weight,
            axis=1,
        )
    total_logits = base_logits + context_logits

    metrics = {
        'base_only': metric_summary(labels, base_logits),
        'context_only': metric_summary(labels, context_logits),
        'base_plus_context': metric_summary(labels, total_logits),
    }
    pearson = np.corrcoef(base_logits, context_logits)[0, 1]
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'audit_type': 'support_unit_checkpoint_component_pure_inference',
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
        'validation': {
            'records': int(len(labels)),
            'positives': int(np.sum(labels)),
            'negatives': int(len(labels) - np.sum(labels)),
        },
        'active_context_term': active_term,
        'metrics': metrics,
        'diagnostics': {
            'context_base_pearson': float(pearson),
            'context_mean_abs_logit': float(np.mean(np.abs(context_logits))),
            'context_weight_mean_abs': float(np.mean(np.abs(context_weight))),
            'total_minus_base_aupr': float(
                metrics['base_plus_context']['AUPR']
                - metrics['base_only']['AUPR']
            ),
        },
        'training_steps': 0,
        'outer_test_evaluated': False,
    }

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'support_context_components' /
        conf['model.variant']
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'audit.json'
    markdown_path = output_dir / 'summary.md'
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    markdown_path.write_text(markdown_report(report), encoding='utf-8')

    print(markdown_report(report))
    print('JSON: %s' % json_path)
    print('Markdown: %s' % markdown_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
