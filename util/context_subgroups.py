from collections import OrderedDict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def binary_metrics(labels, logits, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int32)
    scores = sigmoid(logits)
    predictions = (scores >= float(threshold)).astype(np.int32)
    has_both_classes = len(np.unique(labels)) == 2
    return {
        'AUC': float(roc_auc_score(labels, scores)) if has_both_classes else None,
        'AUPR': float(average_precision_score(labels, scores)) if has_both_classes else None,
        'Recall': float(recall_score(labels, predictions, zero_division=0)),
        'Precision': float(precision_score(labels, predictions, zero_division=0)),
        'F1-score': float(f1_score(labels, predictions, zero_division=0)),
        'predicted_positives': int(np.sum(predictions)),
    }


def select_f1_threshold(labels, logits):
    labels = np.asarray(labels, dtype=np.int32)
    if len(np.unique(labels)) != 2:
        raise ValueError('Threshold calibration requires both positive and negative labels.')
    scores = sigmoid(logits)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError('Threshold calibration did not produce any candidate thresholds.')
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = float(np.max(f1_values))
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0.0, atol=1e-12))
    best_index = min(
        candidates.tolist(),
        key=lambda index: (abs(float(thresholds[index]) - 0.5), -float(thresholds[index])),
    )
    threshold = float(thresholds[best_index])
    return {
        'objective': 'F1-score',
        'threshold': threshold,
        'validation_metrics': binary_metrics(labels, logits, threshold=threshold),
    }


def herb_degree_bin(value):
    value = int(value)
    if value <= 0:
        return '0'
    if value == 1:
        return '1'
    if value <= 3:
        return '2-3'
    if value <= 10:
        return '4-10'
    return '>10'


def training_cp_degree_bin(value):
    value = int(value)
    if value <= 0:
        return '0'
    if value <= 2:
        return '1-2'
    if value <= 5:
        return '3-5'
    if value <= 10:
        return '6-10'
    return '>10'


def mention_count_bin(value):
    if value is None:
        return 'missing'
    value = int(value)
    if value < 10:
        return '<10'
    if value <= 19:
        return '10-19'
    if value <= 49:
        return '20-49'
    if value <= 99:
        return '50-99'
    return '>=100'


def _metric_delta(current, reference):
    return {
        key: (
            None if current[key] is None or reference[key] is None
            else float(current[key] - reference[key])
        )
        for key in ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')
    }


def _mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else None


def summarize_subset(
        labels,
        baseline_logits,
        herb_base_logits,
        herb_total_logits,
        context_logits,
        baseline_threshold=0.5,
        herb_threshold=0.5):
    labels = np.asarray(labels, dtype=np.int32)
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    herb_base_logits = np.asarray(herb_base_logits, dtype=np.float64)
    herb_total_logits = np.asarray(herb_total_logits, dtype=np.float64)
    context_logits = np.asarray(context_logits, dtype=np.float64)
    lengths = {
        len(labels), len(baseline_logits), len(herb_base_logits),
        len(herb_total_logits), len(context_logits),
    }
    if len(lengths) != 1:
        raise ValueError('All subgroup arrays must have the same length.')

    baseline = binary_metrics(labels, baseline_logits, threshold=baseline_threshold)
    herb_base = binary_metrics(labels, herb_base_logits, threshold=herb_threshold)
    herb_total = binary_metrics(labels, herb_total_logits, threshold=herb_threshold)
    baseline_predictions = (
        sigmoid(baseline_logits) >= float(baseline_threshold)
    ).astype(np.int32)
    herb_predictions = (
        sigmoid(herb_total_logits) >= float(herb_threshold)
    ).astype(np.int32)
    positives = labels == 1
    negatives = ~positives

    return {
        'records': int(len(labels)),
        'positives': int(np.sum(positives)),
        'negatives': int(np.sum(negatives)),
        'thresholds': {
            'baseline': float(baseline_threshold),
            'herb': float(herb_threshold),
        },
        'baseline': baseline,
        'herb_base_only': herb_base,
        'herb_total': herb_total,
        'herb_total_minus_baseline': _metric_delta(herb_total, baseline),
        'herb_total_minus_herb_base': _metric_delta(herb_total, herb_base),
        'context_logit': {
            'mean': _mean_or_none(context_logits),
            'mean_abs': _mean_or_none(np.abs(context_logits)),
            'positive_mean': _mean_or_none(context_logits[positives]),
            'negative_mean': _mean_or_none(context_logits[negatives]),
        },
        'prediction_transitions': {
            'FN_to_TP': int(np.sum(positives & (baseline_predictions == 0) & (herb_predictions == 1))),
            'TP_to_FN': int(np.sum(positives & (baseline_predictions == 1) & (herb_predictions == 0))),
            'FP_to_TN': int(np.sum(negatives & (baseline_predictions == 1) & (herb_predictions == 0))),
            'TN_to_FP': int(np.sum(negatives & (baseline_predictions == 0) & (herb_predictions == 1))),
        },
    }


def build_subgroup_report(
        labels,
        baseline_logits,
        herb_base_logits,
        herb_total_logits,
        context_logits,
        subgroup_values,
        baseline_threshold=0.5,
        herb_threshold=0.5):
    labels = np.asarray(labels, dtype=np.int32)
    arrays = {
        'baseline_logits': np.asarray(baseline_logits, dtype=np.float64),
        'herb_base_logits': np.asarray(herb_base_logits, dtype=np.float64),
        'herb_total_logits': np.asarray(herb_total_logits, dtype=np.float64),
        'context_logits': np.asarray(context_logits, dtype=np.float64),
    }
    expected = len(labels)
    if any(len(values) != expected for values in arrays.values()):
        raise ValueError('Score arrays must match the label count.')

    report = {
        'overall': summarize_subset(
            labels,
            **arrays,
            baseline_threshold=baseline_threshold,
            herb_threshold=herb_threshold,
        ),
        'subgroups': OrderedDict(),
    }
    for subgroup_name, raw_groups in subgroup_values.items():
        raw_groups = np.asarray(raw_groups, dtype=object)
        if len(raw_groups) != expected:
            raise ValueError('Subgroup %s does not match the label count.' % subgroup_name)
        group_report = OrderedDict()
        for group in sorted(set(raw_groups.tolist())):
            mask = raw_groups == group
            group_report[str(group)] = summarize_subset(
                labels[mask],
                arrays['baseline_logits'][mask],
                arrays['herb_base_logits'][mask],
                arrays['herb_total_logits'][mask],
                arrays['context_logits'][mask],
                baseline_threshold=baseline_threshold,
                herb_threshold=herb_threshold,
            )
        report['subgroups'][subgroup_name] = group_report
    return report
