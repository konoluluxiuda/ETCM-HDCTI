import numpy as np
from scipy import sparse


def l2_normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > 0,
    )


def hyperedge_specificity_weights(incidence):
    incidence = sparse.csr_matrix(incidence, dtype=np.float32)
    degrees = np.asarray(incidence.sum(axis=0)).reshape(-1).astype(np.float32)
    node_count = float(incidence.shape[0])
    weights = np.zeros_like(degrees, dtype=np.float32)
    active = degrees > 0
    weights[active] = np.log1p(node_count / degrees[active])
    return weights, degrees


def aggregate_hyperedge_contexts(incidence, hyperedge_embeddings, weights=None):
    incidence = sparse.csr_matrix(incidence, dtype=np.float32)
    hyperedge_embeddings = np.asarray(hyperedge_embeddings, dtype=np.float32)
    if incidence.shape[1] != hyperedge_embeddings.shape[0]:
        raise ValueError('Incidence columns and hyperedge embeddings do not match.')
    if weights is None:
        weighted_incidence = incidence
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (incidence.shape[1],):
            raise ValueError('Hyperedge weights have the wrong shape.')
        weighted_incidence = incidence.multiply(weights).tocsr()
    denominators = np.asarray(weighted_incidence.sum(axis=1)).reshape(-1)
    contexts = np.asarray(
        weighted_incidence.dot(hyperedge_embeddings), dtype=np.float32
    )
    contexts = np.divide(
        contexts,
        denominators[:, None],
        out=np.zeros_like(contexts),
        where=denominators[:, None] > 0,
    )
    return l2_normalize_rows(contexts)


def context_change_statistics(
        incidence,
        specificity_weights,
        uniform_contexts,
        specificity_contexts):
    incidence = sparse.csr_matrix(incidence, dtype=np.float32)
    specificity_weights = np.asarray(specificity_weights, dtype=np.float32)
    uniform_contexts = l2_normalize_rows(uniform_contexts)
    specificity_contexts = l2_normalize_rows(specificity_contexts)
    covered = np.asarray(incidence.sum(axis=1)).reshape(-1) > 0
    similarities = np.sum(uniform_contexts * specificity_contexts, axis=1)
    distances = 1.0 - np.clip(similarities[covered], -1.0, 1.0)
    degrees = np.asarray(incidence.sum(axis=0)).reshape(-1)
    active = degrees > 0
    active_weights = specificity_weights[active]
    active_degrees = degrees[active]
    if len(active_degrees):
        broad_cutoff = float(np.percentile(active_degrees, 90))
        broad = active_degrees >= broad_cutoff
        uniform_mass = active_degrees
        weighted_mass = active_degrees * active_weights
        broad_uniform_share = float(
            np.sum(uniform_mass[broad]) / np.sum(uniform_mass)
        )
        broad_weighted_share = float(
            np.sum(weighted_mass[broad]) / np.sum(weighted_mass)
        )
        broad_reduction = float(
            (broad_uniform_share - broad_weighted_share) / broad_uniform_share
        ) if broad_uniform_share > 0 else 0.0
        weight_cv = float(np.std(active_weights) / np.mean(active_weights))
    else:
        broad_cutoff = 0.0
        broad_uniform_share = 0.0
        broad_weighted_share = 0.0
        broad_reduction = 0.0
        weight_cv = 0.0
    return {
        'nodes': int(incidence.shape[0]),
        'hyperedges': int(incidence.shape[1]),
        'active_hyperedges': int(np.sum(active)),
        'coverage': float(np.mean(covered)) if len(covered) else 0.0,
        'weight_cv': weight_cv,
        'weight_minimum': float(np.min(active_weights)) if len(active_weights) else 0.0,
        'weight_median': float(np.median(active_weights)) if len(active_weights) else 0.0,
        'weight_maximum': float(np.max(active_weights)) if len(active_weights) else 0.0,
        'mean_cosine_distance': float(np.mean(distances)) if len(distances) else 0.0,
        'median_cosine_distance': float(np.median(distances)) if len(distances) else 0.0,
        'p90_cosine_distance': float(np.percentile(distances, 90)) if len(distances) else 0.0,
        'broad_degree_cutoff': broad_cutoff,
        'broad_uniform_mass_share': broad_uniform_share,
        'broad_weighted_mass_share': broad_weighted_share,
        'broad_mass_relative_reduction': broad_reduction,
    }


def specificity_pair_features(
        compound_embeddings,
        protein_embeddings,
        uniform_compound_contexts,
        specificity_compound_contexts,
        specificity_protein_contexts,
        compound_indices,
        protein_indices,
        herb_protein_weight):
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein indices must have matching shapes.')
    compounds = l2_normalize_rows(compound_embeddings)[compound_indices]
    proteins = l2_normalize_rows(protein_embeddings)[protein_indices]
    uniform_herbs = l2_normalize_rows(uniform_compound_contexts)[compound_indices]
    specific_herbs = l2_normalize_rows(specificity_compound_contexts)[compound_indices]
    specific_diseases = l2_normalize_rows(specificity_protein_contexts)[protein_indices]
    herb_protein_weight = np.asarray(herb_protein_weight, dtype=np.float32)
    if herb_protein_weight.shape != (compounds.shape[1],):
        raise ValueError('Hctx-P weight has the wrong shape.')
    static_herb_term = np.sum(
        uniform_herbs * proteins * herb_protein_weight, axis=1
    )
    specific_herb_term = np.sum(
        specific_herbs * proteins * herb_protein_weight, axis=1
    )
    return {
        'herb_specificity_replacement_delta': (
            specific_herb_term - static_herb_term
        ),
        'compound_specific_disease_cosine': np.sum(
            compounds * specific_diseases, axis=1
        ),
        'specific_context_cosine': np.sum(
            specific_herbs * specific_diseases, axis=1
        ),
    }
