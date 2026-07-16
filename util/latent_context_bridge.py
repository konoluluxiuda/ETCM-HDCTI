from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > 0,
    )


def build_incidence_index(path, entity_map, context_map, entity_column):
    memberships = defaultdict(set)
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError('Invalid relation row %d in %s.' % (line_number, path))
            context_id, entity_id = (
                (parts[0], parts[1]) if entity_column == 1 else (parts[1], parts[0])
            )
            if entity_id not in entity_map or context_id not in context_map:
                continue
            memberships[entity_map[entity_id]].add(context_map[context_id])
    return {key: np.asarray(sorted(values), dtype=np.int64)
            for key, values in memberships.items()}


def pair_alignment_scores(
        herb_edges,
        disease_edges,
        compound_herbs,
        protein_diseases,
        compound_indices,
        protein_indices,
        top_ks=(1, 3, 5)):
    herb_edges = normalize_rows(herb_edges)
    disease_edges = normalize_rows(disease_edges)
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')

    top_ks = tuple(sorted({int(value) for value in top_ks if int(value) > 0}))
    if not top_ks:
        raise ValueError('At least one positive Top-K value is required.')
    scores = {'top%d_mean' % value: np.full(len(compound_indices), np.nan)
              for value in top_ks}
    herb_degrees = np.zeros(len(compound_indices), dtype=np.int64)
    disease_degrees = np.zeros(len(compound_indices), dtype=np.int64)

    for row_index, (compound_index, protein_index) in enumerate(
            zip(compound_indices, protein_indices)):
        herb_indices = compound_herbs.get(int(compound_index))
        disease_indices = protein_diseases.get(int(protein_index))
        if herb_indices is None or disease_indices is None:
            continue
        herb_degrees[row_index] = len(herb_indices)
        disease_degrees[row_index] = len(disease_indices)
        if not len(herb_indices) or not len(disease_indices):
            continue
        similarities = np.matmul(
            herb_edges[herb_indices], disease_edges[disease_indices].T
        ).reshape(-1)
        for top_k in top_ks:
            count = min(top_k, similarities.size)
            if count == similarities.size:
                selected = similarities
            else:
                selected = np.partition(similarities, similarities.size - count)[-count:]
            scores['top%d_mean' % top_k][row_index] = float(np.mean(selected))

    return {
        'scores': scores,
        'herb_degrees': herb_degrees,
        'disease_degrees': disease_degrees,
        'covered': np.isfinite(next(iter(scores.values()))),
    }


def stratified_half_split(labels, seed):
    labels = np.asarray(labels, dtype=np.int32)
    selection = []
    audit = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError('Each class needs at least two records for the nested audit split.')
        indices = indices.copy()
        np.random.RandomState(int(seed) + int(label)).shuffle(indices)
        boundary = len(indices) // 2
        selection.extend(indices[:boundary].tolist())
        audit.extend(indices[boundary:].tolist())
    return (
        np.asarray(sorted(selection), dtype=np.int64),
        np.asarray(sorted(audit), dtype=np.int64),
    )


def _ranking_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int32)
    logits = np.asarray(logits, dtype=np.float64)
    return {
        'AUC': float(roc_auc_score(labels, logits)),
        'AUPR': float(average_precision_score(labels, logits)),
    }


def select_positive_residual(
        labels,
        baseline_logits,
        feature_scores,
        selection_indices,
        audit_indices,
        alphas=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)):
    labels = np.asarray(labels, dtype=np.int32)
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    selection_indices = np.asarray(selection_indices, dtype=np.int64)
    audit_indices = np.asarray(audit_indices, dtype=np.int64)
    candidates = []
    standardized = {}

    for feature_name in sorted(feature_scores):
        values = np.asarray(feature_scores[feature_name], dtype=np.float64)
        mean = float(np.mean(values[selection_indices]))
        std = float(np.std(values[selection_indices]))
        if not np.isfinite(std) or std <= 0:
            continue
        z_values = (values - mean) / std
        standardized[feature_name] = {
            'values': z_values,
            'selection_mean': mean,
            'selection_std': std,
        }
        for alpha in alphas:
            alpha = float(alpha)
            if alpha < 0:
                raise ValueError('Only non-negative residual coefficients are permitted.')
            logits = baseline_logits[selection_indices] + alpha * z_values[selection_indices]
            metrics = _ranking_metrics(labels[selection_indices], logits)
            candidates.append({
                'feature': feature_name,
                'alpha': alpha,
                'selection_AUC': metrics['AUC'],
                'selection_AUPR': metrics['AUPR'],
            })
    if not candidates:
        raise ValueError('No finite alignment feature is available for residual selection.')

    best = sorted(
        candidates,
        key=lambda row: (-row['selection_AUPR'], row['alpha'], row['feature']),
    )[0]
    feature = standardized[best['feature']]['values']
    alpha = best['alpha']
    baseline_selection = _ranking_metrics(
        labels[selection_indices], baseline_logits[selection_indices]
    )
    baseline_audit = _ranking_metrics(labels[audit_indices], baseline_logits[audit_indices])
    fused_audit_logits = baseline_logits[audit_indices] + alpha * feature[audit_indices]
    fused_audit = _ranking_metrics(labels[audit_indices], fused_audit_logits)
    positive_values = feature[audit_indices][labels[audit_indices] == 1]
    negative_values = feature[audit_indices][labels[audit_indices] == 0]
    return {
        'selected': best,
        'standardization': {
            key: value for key, value in standardized[best['feature']].items()
            if key != 'values'
        },
        'selection_baseline': baseline_selection,
        'audit_baseline': baseline_audit,
        'audit_fused': fused_audit,
        'audit_delta': {
            key: float(fused_audit[key] - baseline_audit[key])
            for key in ('AUC', 'AUPR')
        },
        'audit_feature': feature[audit_indices],
        'audit_feature_separation': {
            'positive_mean': float(np.mean(positive_values)),
            'negative_mean': float(np.mean(negative_values)),
            'positive_minus_negative': float(
                np.mean(positive_values) - np.mean(negative_values)
            ),
        },
        'candidates': candidates,
    }


def degree_stratified_permutation(
        labels,
        baseline_logits,
        feature,
        alpha,
        strata,
        draws=200,
        seed=72026):
    labels = np.asarray(labels, dtype=np.int32)
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    feature = np.asarray(feature, dtype=np.float64)
    strata = np.asarray(strata, dtype=object)
    if not (len(labels) == len(baseline_logits) == len(feature) == len(strata)):
        raise ValueError('Permutation arrays must have matching lengths.')
    observed = _ranking_metrics(labels, baseline_logits + float(alpha) * feature)
    baseline = _ranking_metrics(labels, baseline_logits)
    observed_delta = observed['AUPR'] - baseline['AUPR']
    groups = {
        str(group): np.flatnonzero(strata == group)
        for group in sorted(set(strata.tolist()))
    }
    rng = np.random.RandomState(int(seed))
    deltas = []
    for _ in range(int(draws)):
        shuffled = feature.copy()
        for indices in groups.values():
            if len(indices) > 1:
                shuffled[indices] = feature[rng.permutation(indices)]
        value = _ranking_metrics(labels, baseline_logits + float(alpha) * shuffled)
        deltas.append(float(value['AUPR'] - baseline['AUPR']))
    deltas = np.asarray(deltas, dtype=np.float64)
    return {
        'draws': int(draws),
        'seed': int(seed),
        'strata': len(groups),
        'observed_AUPR_delta': float(observed_delta),
        'null_mean_AUPR_delta': float(np.mean(deltas)),
        'null_p95_AUPR_delta': float(np.percentile(deltas, 95)),
        'one_sided_p': float((1 + np.sum(deltas >= observed_delta)) / (len(deltas) + 1)),
    }


def low_rank_bilinear_residual(herb_contexts, disease_contexts, left, right):
    herb_contexts = np.asarray(herb_contexts, dtype=np.float64)
    disease_contexts = np.asarray(disease_contexts, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if herb_contexts.shape != disease_contexts.shape:
        raise ValueError('Herb and disease context arrays must have matching shapes.')
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError('Low-rank projection matrices must have the same 2D shape.')
    if herb_contexts.shape[1] != left.shape[0]:
        raise ValueError('Context and projection dimensions do not match.')
    scale = np.sqrt(float(left.shape[1]))
    return np.sum(
        np.matmul(herb_contexts, left) * np.matmul(disease_contexts, right),
        axis=1,
    ) / scale


def train_low_rank_bilinear_probe(
        herb_contexts,
        disease_contexts,
        labels,
        baseline_logits,
        rank=8,
        steps=500,
        learning_rate=0.01,
        l2=1e-4,
        seed=82026):
    herb_contexts = np.asarray(herb_contexts, dtype=np.float64)
    disease_contexts = np.asarray(disease_contexts, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    if herb_contexts.shape != disease_contexts.shape:
        raise ValueError('Herb and disease probe contexts must have matching shapes.')
    if herb_contexts.ndim != 2:
        raise ValueError('Probe contexts must be 2D arrays.')
    if len(labels) != len(herb_contexts) or len(baseline_logits) != len(labels):
        raise ValueError('Probe labels, logits, and contexts must have matching lengths.')
    rank = int(rank)
    steps = int(steps)
    if rank <= 0 or steps <= 0:
        raise ValueError('Probe rank and optimization steps must be positive.')
    dimension = herb_contexts.shape[1]
    rng = np.random.RandomState(int(seed))
    left = rng.normal(0.0, 1.0 / np.sqrt(dimension), (dimension, rank))
    right = np.zeros((dimension, rank), dtype=np.float64)
    moments = {
        'left_m': np.zeros_like(left),
        'left_v': np.zeros_like(left),
        'right_m': np.zeros_like(right),
        'right_v': np.zeros_like(right),
    }
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    scale = np.sqrt(float(rank))

    def loss_value():
        residual = low_rank_bilinear_residual(
            herb_contexts, disease_contexts, left, right
        )
        logits = baseline_logits + residual
        bce = np.mean(
            np.maximum(logits, 0.0) - labels * logits
            + np.log1p(np.exp(-np.abs(logits)))
        )
        penalty = 0.5 * float(l2) * (
            np.sum(left * left) + np.sum(right * right)
        )
        return float(bce + penalty)

    initial_loss = loss_value()
    for step in range(1, steps + 1):
        herb_projected = np.matmul(herb_contexts, left)
        disease_projected = np.matmul(disease_contexts, right)
        residual = np.sum(herb_projected * disease_projected, axis=1) / scale
        logits = baseline_logits + residual
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        errors = (probabilities - labels) / len(labels)
        left_gradient = np.matmul(
            herb_contexts.T, errors[:, None] * disease_projected
        ) / scale + float(l2) * left
        right_gradient = np.matmul(
            disease_contexts.T, errors[:, None] * herb_projected
        ) / scale + float(l2) * right
        gradient_norm = np.sqrt(
            np.sum(left_gradient * left_gradient)
            + np.sum(right_gradient * right_gradient)
        )
        if gradient_norm > 10.0:
            left_gradient *= 10.0 / gradient_norm
            right_gradient *= 10.0 / gradient_norm

        for name, parameter, gradient in (
                ('left', left, left_gradient), ('right', right, right_gradient)):
            first = moments[name + '_m']
            second = moments[name + '_v']
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * gradient * gradient
            first_hat = first / (1.0 - beta1 ** step)
            second_hat = second / (1.0 - beta2 ** step)
            parameter -= float(learning_rate) * first_hat / (
                np.sqrt(second_hat) + epsilon
            )

    return {
        'left': left.astype(np.float32),
        'right': right.astype(np.float32),
        'initial_loss': initial_loss,
        'final_loss': loss_value(),
        'rank': rank,
        'steps': steps,
        'learning_rate': float(learning_rate),
        'l2': float(l2),
        'seed': int(seed),
    }
