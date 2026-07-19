from collections import defaultdict

import numpy as np


def l2_normalize_rows(values, epsilon=1e-12):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > float(epsilon),
    )


def indexed_hc_memberships(path, herb_map, compound_map):
    herb_members = defaultdict(set)
    compound_herbs = defaultdict(set)
    raw_rows = 0
    mapped_rows = 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            raw_rows += 1
            herb_id, compound_id = str(parts[0]), str(parts[1])
            if herb_id not in herb_map or compound_id not in compound_map:
                continue
            herb_index = int(herb_map[herb_id])
            compound_index = int(compound_map[compound_id])
            herb_members[herb_index].add(compound_index)
            compound_herbs[compound_index].add(herb_index)
            mapped_rows += 1
    return {
        'herb_members': {
            key: np.asarray(sorted(values), dtype=np.int64)
            for key, values in herb_members.items()
        },
        'compound_herbs': {
            key: np.asarray(sorted(values), dtype=np.int64)
            for key, values in compound_herbs.items()
        },
        'raw_rows': int(raw_rows),
        'mapped_rows': int(mapped_rows),
        'unique_mapped_rows': int(sum(len(values) for values in herb_members.values())),
    }


def build_direct_self_excluded_contexts(
        herb_edges,
        hc_edge_layer_inputs,
        herb_members,
        compound_herbs,
        num_compounds):
    """Build frozen direct leave-self-out H-C contexts.

    ``herb_edges`` is the sum of the normalized mean node-to-edge outputs
    across model layers. ``hc_edge_layer_inputs`` retains the node inputs for
    every layer. The direct contribution of compound c to herb h is removed
    before the same per-layer L2 normalization as

        normalize((sum_inputs(h) - input(c)) / (degree(h) - 1)).

    This removes c's direct contribution at readout. It does not rerun message
    passing with c deleted, so indirect earlier-layer effects remain and must be
    treated as an audit boundary rather than a trained inductive encoder.
    """
    herb_edges = np.asarray(herb_edges, dtype=np.float64)
    hc_edge_layer_inputs = np.asarray(hc_edge_layer_inputs, dtype=np.float64)
    if herb_edges.ndim != 2 or hc_edge_layer_inputs.ndim != 3:
        raise ValueError(
            'Herb edges must be a matrix and H-C layer inputs a 3D tensor.'
        )
    if herb_edges.shape[1] != hc_edge_layer_inputs.shape[2]:
        raise ValueError('Herb edge and node input dimensions do not match.')
    if hc_edge_layer_inputs.shape[1] != int(num_compounds):
        raise ValueError('H-C edge inputs do not match num_compounds.')

    dimension = herb_edges.shape[1]
    inclusive_sums = np.zeros((num_compounds, dimension), dtype=np.float64)
    excluded_sums = np.zeros((num_compounds, dimension), dtype=np.float64)
    incident_counts = np.zeros(num_compounds, dtype=np.int64)
    eligible_counts = np.zeros(num_compounds, dtype=np.int64)
    edge_errors = []

    for herb_index, members in herb_members.items():
        members = np.asarray(members, dtype=np.int64)
        if members.size == 0:
            continue
        degree = int(members.size)
        edge = herb_edges[int(herb_index)]
        inclusive_sums[members] += edge
        incident_counts[members] += 1

        layer_member_inputs = hc_edge_layer_inputs[:, members, :]
        layer_totals = np.sum(layer_member_inputs, axis=1)
        inclusive_layer_edges = l2_normalize_rows(
            layer_totals / float(degree)
        )
        reconstructed_edge = np.sum(inclusive_layer_edges, axis=0)
        denominator = max(float(np.linalg.norm(edge)), 1e-12)
        edge_errors.append(float(
            np.linalg.norm(reconstructed_edge - edge) / denominator
        ))
        if degree < 2:
            continue

        leave_one_out_raw = (
            layer_totals[:, None, :] - layer_member_inputs
        ) / float(degree - 1)
        leave_one_out_edges = np.sum(
            l2_normalize_rows(
                leave_one_out_raw.reshape(-1, dimension)
            ).reshape(
                hc_edge_layer_inputs.shape[0], degree, dimension
            ),
            axis=0,
        )
        excluded_sums[members] += leave_one_out_edges
        eligible_counts[members] += 1

    inclusive_contexts = l2_normalize_rows(inclusive_sums)
    excluded_contexts = l2_normalize_rows(excluded_sums)
    eligible = eligible_counts > 0
    herb_degrees = np.asarray(
        [len(compound_herbs.get(index, ())) for index in range(num_compounds)],
        dtype=np.int64,
    )
    return {
        'inclusive_contexts': inclusive_contexts.astype(np.float32),
        'self_excluded_contexts': excluded_contexts.astype(np.float32),
        'eligible': eligible,
        'incident_herb_counts': incident_counts,
        'eligible_herb_counts': eligible_counts,
        'herb_degrees': herb_degrees,
        'edge_reconstruction_relative_error': {
            'mean': float(np.mean(edge_errors)) if edge_errors else None,
            'maximum': float(np.max(edge_errors)) if edge_errors else None,
        },
    }


def cosine_distance_rows(left, right, mask=None):
    left = l2_normalize_rows(left)
    right = l2_normalize_rows(right)
    if left.shape != right.shape:
        raise ValueError('Cosine distance inputs must have identical shapes.')
    distances = 1.0 - np.sum(left * right, axis=1)
    if mask is not None:
        distances = distances[np.asarray(mask, dtype=bool)]
    return distances


def herb_protein_context_logits(
        compound_contexts,
        protein_embeddings,
        compound_indices,
        protein_indices,
        context_weight):
    compound_contexts = np.asarray(compound_contexts, dtype=np.float64)
    protein_embeddings = np.asarray(protein_embeddings, dtype=np.float64)
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    context_weight = np.asarray(context_weight, dtype=np.float64)
    return np.sum(
        compound_contexts[compound_indices]
        * protein_embeddings[protein_indices]
        * context_weight[None, :],
        axis=1,
    )
