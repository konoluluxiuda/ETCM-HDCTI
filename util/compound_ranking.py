from collections import defaultdict

import numpy as np


def binary_labels_from_records(records):
    return np.asarray([
        1 if float(record[2]) > 0 else 0 for record in records
    ], dtype=np.int32)


def compound_group_ranking(compound_ids, labels, scores):
    compound_ids = np.asarray(compound_ids, dtype=object)
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if not (len(compound_ids) == len(labels) == len(scores)):
        raise ValueError('Compound IDs, labels, and scores must have matching lengths.')
    groups = defaultdict(list)
    for index, compound_id in enumerate(compound_ids):
        groups[str(compound_id)].append(index)

    rows = []
    all_margins = []
    eligible_records = 0
    for compound_id in sorted(groups):
        indices = np.asarray(groups[compound_id], dtype=np.int64)
        group_labels = labels[indices]
        positive_scores = scores[indices][group_labels == 1]
        negative_scores = scores[indices][group_labels == 0]
        if not len(positive_scores) or not len(negative_scores):
            continue
        eligible_records += len(indices)
        margins = (
            positive_scores[:, None] - negative_scores[None, :]
        ).reshape(-1)
        all_margins.append(margins)
        pairwise_accuracy = float(
            np.mean((margins > 0).astype(np.float64) + 0.5 * (margins == 0))
        )
        order = np.argsort(-scores[indices], kind='mergesort')
        ranked_labels = group_labels[order]
        first_positive_rank = int(np.flatnonzero(ranked_labels == 1)[0] + 1)
        rows.append({
            'compound_id': compound_id,
            'candidate_count': int(len(indices)),
            'positive_count': int(len(positive_scores)),
            'negative_count': int(len(negative_scores)),
            'pair_count': int(len(margins)),
            'pairwise_accuracy': pairwise_accuracy,
            'violation_rate': float(1.0 - pairwise_accuracy),
            'bpr_loss': float(np.mean(np.logaddexp(0.0, -margins))),
            'first_positive_rank': first_positive_rank,
            'mrr': float(1.0 / first_positive_rank),
            'top1_hit': int(first_positive_rank == 1),
            'top1_miss': int(first_positive_rank != 1),
            'minimum_margin': float(np.min(margins)),
            'median_margin': float(np.median(margins)),
            'mean_margin': float(np.mean(margins)),
        })
    if not rows:
        raise ValueError('No compound has both positive and negative candidates.')
    margins = np.concatenate(all_margins)
    summary = {
        'all_compounds': int(len(groups)),
        'eligible_compounds': int(len(rows)),
        'eligible_records': int(eligible_records),
        'eligible_record_fraction': float(eligible_records / len(labels)),
        'macro_pairwise_accuracy': float(np.mean([
            row['pairwise_accuracy'] for row in rows
        ])),
        'macro_violation_rate': float(np.mean([
            row['violation_rate'] for row in rows
        ])),
        'micro_pairwise_accuracy': float(
            np.mean((margins > 0).astype(np.float64) + 0.5 * (margins == 0))
        ),
        'micro_violation_rate': float(
            1.0 - np.mean(
                (margins > 0).astype(np.float64) + 0.5 * (margins == 0)
            )
        ),
        'macro_bpr_loss': float(np.mean([row['bpr_loss'] for row in rows])),
        'macro_mrr': float(np.mean([row['mrr'] for row in rows])),
        'top1_hit_rate': float(np.mean([row['top1_hit'] for row in rows])),
        'top1_miss_rate': float(np.mean([row['top1_miss'] for row in rows])),
        'margin': {
            'minimum': float(np.min(margins)),
            'p10': float(np.percentile(margins, 10)),
            'median': float(np.median(margins)),
            'mean': float(np.mean(margins)),
            'p90': float(np.percentile(margins, 90)),
        },
    }
    return summary, rows


def bootstrap_mean_interval(values, draws=1000, seed=142026, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError('Bootstrap values must be a non-empty 1D array.')
    draws = int(draws)
    if draws <= 0:
        raise ValueError('Bootstrap draws must be positive.')
    rng = np.random.RandomState(int(seed))
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.randint(0, len(values), size=len(values))
        means[index] = np.mean(values[sample])
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        'draws': draws,
        'seed': int(seed),
        'confidence': float(confidence),
        'mean': float(np.mean(values)),
        'lower': float(np.quantile(means, alpha)),
        'upper': float(np.quantile(means, 1.0 - alpha)),
    }


def degree_bin(value):
    value = int(value)
    if value <= 0:
        return '0'
    power = int(np.floor(np.log2(value)))
    return '%d-%d' % (2 ** power, 2 ** (power + 1) - 1)


def degree_stratified_ranking(rows, training_positive_degrees):
    groups = defaultdict(list)
    for row in rows:
        degree = int(training_positive_degrees.get(row['compound_id'], 0))
        groups[degree_bin(degree)].append(row)
    output = []
    for name in sorted(groups, key=lambda value: int(value.split('-', 1)[0])):
        values = groups[name]
        output.append({
            'degree_bin': name,
            'compounds': int(len(values)),
            'mean_training_degree': float(np.mean([
                training_positive_degrees.get(row['compound_id'], 0)
                for row in values
            ])),
            'macro_violation_rate': float(np.mean([
                row['violation_rate'] for row in values
            ])),
            'top1_miss_rate': float(np.mean([
                row['top1_miss'] for row in values
            ])),
            'macro_mrr': float(np.mean([row['mrr'] for row in values])),
        })
    return output
