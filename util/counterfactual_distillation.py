from collections import defaultdict

import numpy as np


def grouped_compound_holdout(compound_ids, holdout_ratio=0.2, seed=52026):
    """Return deterministic fit/evaluation masks with disjoint compounds."""
    compound_ids = np.asarray([str(value) for value in compound_ids], dtype=object)
    unique_compounds = np.asarray(sorted(set(compound_ids.tolist())), dtype=object)
    if unique_compounds.size < 2:
        raise ValueError('At least two compounds are required for grouped holdout.')
    holdout_ratio = float(holdout_ratio)
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError('holdout_ratio must be between 0 and 1.')

    rng = np.random.RandomState(int(seed))
    shuffled = unique_compounds.copy()
    rng.shuffle(shuffled)
    holdout_count = min(
        unique_compounds.size - 1,
        max(1, int(round(unique_compounds.size * holdout_ratio))),
    )
    evaluation_compounds = set(shuffled[:holdout_count].tolist())
    evaluation_mask = np.asarray(
        [value in evaluation_compounds for value in compound_ids], dtype=bool
    )
    return ~evaluation_mask, evaluation_mask


def build_cross_pool_counterfactuals(
        source_ids, donor_ids, memberships, draws=5, seed=42026):
    """Match source compounds to equal-degree, herb-disjoint donor compounds."""
    draws = int(draws)
    if draws <= 0:
        raise ValueError('draws must be positive.')
    normalized = {
        str(compound_id): frozenset(str(herb_id) for herb_id in herbs)
        for compound_id, herbs in memberships.items()
        if herbs
    }
    sources = sorted({str(value) for value in source_ids})
    donors = sorted({str(value) for value in donor_ids})
    donor_buckets = defaultdict(list)
    for donor_id in donors:
        herbs = normalized.get(donor_id, frozenset())
        if herbs:
            donor_buckets[len(herbs)].append(donor_id)

    rng = np.random.RandomState(int(seed))
    assignments = {}
    pool_sizes = {}
    for source_id in sources:
        source_herbs = normalized.get(source_id, frozenset())
        if not source_herbs:
            continue
        pool = [
            donor_id for donor_id in donor_buckets[len(source_herbs)]
            if donor_id != source_id
            and not source_herbs.intersection(normalized[donor_id])
        ]
        pool_sizes[source_id] = len(pool)
        if not pool:
            continue
        selected = rng.choice(
            pool, size=draws, replace=len(pool) < draws
        ).tolist()
        assignments[source_id] = [str(value) for value in selected]

    return {
        'assignments': assignments,
        'pool_sizes': pool_sizes,
        'draws': draws,
        'seed': int(seed),
        'requested_sources': len(sources),
        'eligible_sources': len(assignments),
        'donor_compounds': len(donors),
    }


def context_pair_features(compound_contexts, protein_embeddings):
    """Build an ID-free compound-side pair representation."""
    compound_contexts = np.asarray(compound_contexts, dtype=np.float64)
    protein_embeddings = np.asarray(protein_embeddings, dtype=np.float64)
    if compound_contexts.shape != protein_embeddings.shape:
        raise ValueError('Context and protein arrays must have the same shape.')
    if compound_contexts.ndim != 2:
        raise ValueError('Context and protein arrays must be rank-2.')
    return np.concatenate(
        [
            compound_contexts,
            protein_embeddings,
            compound_contexts * protein_embeddings,
            np.abs(compound_contexts - protein_embeddings),
        ],
        axis=1,
    )


def standardize_factual_features(fit_features, *other_features):
    """Scale pair features using factual fit records only."""
    fit_features = np.asarray(fit_features, dtype=np.float64)
    mean = np.mean(fit_features, axis=0)
    scale = np.std(fit_features, axis=0)
    scale[scale < 1e-8] = 1.0
    transformed = [(fit_features - mean) / scale]
    transformed.extend(
        (np.asarray(values, dtype=np.float64) - mean) / scale
        for values in other_features
    )
    return mean, scale, transformed


def explicit_intercept(features, difference=False):
    """Add an intercept column; pair-difference equations receive zero intercept."""
    features = np.asarray(features, dtype=np.float64)
    intercept = np.zeros((features.shape[0], 1), dtype=np.float64)
    if not difference:
        intercept.fill(1.0)
    return np.concatenate([intercept, features], axis=1)

