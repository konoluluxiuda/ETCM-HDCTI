from collections import defaultdict

import numpy as np


ROLE_FEATURE_NAMES = (
    'supported',
    'log_degree',
    'log_edge_size_min',
    'log_edge_size_mean',
    'log_edge_size_std',
    'log_edge_size_max',
    'edge_rarity_min',
    'edge_rarity_mean',
    'edge_rarity_max',
    'log_two_hop_neighbors',
    'log_neighbor_degree_q25',
    'log_neighbor_degree_q50',
    'log_neighbor_degree_q75',
    'log_neighbor_mentions',
    'neighbor_uniqueness',
)


def read_incidence(path, node_column, edge_column):
    edge_nodes = defaultdict(set)
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            values = line.split()
            if not values:
                continue
            if len(values) <= max(node_column, edge_column):
                raise ValueError(
                    'Invalid incidence row %d in %s.' % (line_number, path)
                )
            node_id = str(values[node_column])
            edge_id = str(values[edge_column])
            edge_nodes[edge_id].add(node_id)
    return dict(edge_nodes)


def build_role_features(edge_nodes, node_universe=()):
    nodes = sorted(
        set(str(value) for value in node_universe).union(
            node_id for members in edge_nodes.values() for node_id in members
        )
    )
    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    node_edges = [[] for _ in nodes]
    normalized_edges = []
    for edge_id in sorted(edge_nodes):
        members = sorted(set(str(value) for value in edge_nodes[edge_id]))
        if not members:
            continue
        indices = [node_index[node_id] for node_id in members]
        normalized_edges.append(indices)
        for index in indices:
            node_edges[index].append(indices)

    node_count = len(nodes)
    degrees = np.asarray([len(edges) for edges in node_edges], dtype=np.float64)
    neighbor_masks = [0] * node_count
    neighbor_mentions = np.zeros(node_count, dtype=np.float64)
    for indices in normalized_edges:
        mask = 0
        for index in indices:
            mask |= 1 << index
        mentions = max(0, len(indices) - 1)
        for index in indices:
            neighbor_masks[index] |= mask
            neighbor_mentions[index] += mentions

    features = np.zeros((node_count, len(ROLE_FEATURE_NAMES)), dtype=np.float64)
    for index, incident_edges in enumerate(node_edges):
        if not incident_edges:
            continue
        edge_sizes = np.asarray([len(values) for values in incident_edges], dtype=np.float64)
        rarity = np.log((node_count + 1.0) / (edge_sizes + 1.0))
        neighbor_mask = neighbor_masks[index] & ~(1 << index)
        neighbor_indices = []
        while neighbor_mask:
            lowest_bit = neighbor_mask & -neighbor_mask
            neighbor_indices.append(lowest_bit.bit_length() - 1)
            neighbor_mask ^= lowest_bit
        neighbor_degrees = degrees[neighbor_indices] if neighbor_indices else np.zeros(1)
        unique_neighbors = len(neighbor_indices)
        mentions = neighbor_mentions[index]
        features[index] = (
            1.0,
            np.log1p(degrees[index]),
            np.log1p(np.min(edge_sizes)),
            np.log1p(np.mean(edge_sizes)),
            np.log1p(np.std(edge_sizes)),
            np.log1p(np.max(edge_sizes)),
            np.min(rarity),
            np.mean(rarity),
            np.max(rarity),
            np.log1p(unique_neighbors),
            np.log1p(np.quantile(neighbor_degrees, 0.25)),
            np.log1p(np.quantile(neighbor_degrees, 0.50)),
            np.log1p(np.quantile(neighbor_degrees, 0.75)),
            np.log1p(mentions),
            unique_neighbors / mentions if mentions else 0.0,
        )
    return {
        'node_ids': nodes,
        'node_index': node_index,
        'features': features,
        'degrees': degrees.astype(np.int64),
        'feature_names': ROLE_FEATURE_NAMES,
    }


def pair_role_features(compound_features, protein_features):
    compound_features = np.asarray(compound_features, dtype=np.float64)
    protein_features = np.asarray(protein_features, dtype=np.float64)
    if compound_features.shape != protein_features.shape:
        raise ValueError('Compound and protein role arrays must have the same shape.')
    return np.concatenate(
        (
            compound_features,
            protein_features,
            compound_features * protein_features,
            np.abs(compound_features - protein_features),
        ),
        axis=1,
    )


def scalar_pair_features(compound_values, protein_values):
    compound_values = np.asarray(compound_values, dtype=np.float64)
    protein_values = np.asarray(protein_values, dtype=np.float64)
    if compound_values.shape != protein_values.shape:
        raise ValueError('Compound and protein values must have the same shape.')
    return np.column_stack((
        compound_values,
        protein_values,
        compound_values * protein_values,
        np.abs(compound_values - protein_values),
    ))


def degree_pair_features(compound_degrees, protein_degrees):
    compound_degrees = np.log1p(np.asarray(compound_degrees, dtype=np.float64))
    protein_degrees = np.log1p(np.asarray(protein_degrees, dtype=np.float64))
    return scalar_pair_features(compound_degrees, protein_degrees)


def empirical_percentile_roles(role):
    features = np.asarray(role['features'], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(ROLE_FEATURE_NAMES):
        raise ValueError('Unexpected role feature shape.')
    transformed = np.zeros_like(features)
    supported = features[:, 0] > 0.5
    transformed[:, 0] = supported.astype(np.float64)
    supported_count = int(np.sum(supported))
    if supported_count == 0:
        result = dict(role)
        result['features'] = transformed
        result['normalization'] = 'within_dataset_empirical_percentile'
        return result

    for column in range(1, features.shape[1]):
        values = features[supported, column]
        order = np.argsort(values, kind='mergesort')
        sorted_values = values[order]
        ranks = np.empty(supported_count, dtype=np.float64)
        start = 0
        while start < supported_count:
            end = start + 1
            while end < supported_count and sorted_values[end] == sorted_values[start]:
                end += 1
            average_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = average_rank
            start = end
        if supported_count > 1:
            ranks /= float(supported_count - 1)
        else:
            ranks.fill(0.5)
        transformed[supported, column] = ranks

    result = dict(role)
    result['features'] = transformed
    result['normalization'] = 'within_dataset_empirical_percentile'
    return result


def quantile_bin_members(node_ids, degrees, bin_count=10):
    node_ids = np.asarray([str(value) for value in node_ids], dtype=object)
    degrees = np.asarray(degrees, dtype=np.float64)
    if node_ids.size != degrees.size:
        raise ValueError('node_ids and degrees must have the same length.')
    if int(bin_count) <= 0:
        raise ValueError('bin_count must be positive.')
    boundaries = np.unique(np.quantile(
        degrees, np.linspace(0.0, 1.0, int(bin_count) + 1)[1:-1]
    ))
    bins = np.searchsorted(boundaries, degrees, side='right')
    members = defaultdict(list)
    node_bins = {}
    for node_id, bin_id in zip(node_ids, bins):
        members[int(bin_id)].append(str(node_id))
        node_bins[str(node_id)] = int(bin_id)
    return node_bins, dict(members)


def sample_degree_matched_negatives(
        positive_pairs, compound_ids, compound_degrees,
        protein_ids, protein_degrees, seed=2026, bin_count=10,
        maximum_attempts=200, excluded_pairs=None):
    positives = {
        (str(left), str(right))
        for left, right in (excluded_pairs or positive_pairs)
    }
    compound_bins, compounds_by_bin = quantile_bin_members(
        compound_ids, compound_degrees, bin_count=bin_count
    )
    protein_bins, proteins_by_bin = quantile_bin_members(
        protein_ids, protein_degrees, bin_count=bin_count
    )
    rng = np.random.RandomState(int(seed))
    matched_positives = []
    negatives = []
    used = set()
    unmatched = 0
    for left, right in positive_pairs:
        left = str(left)
        right = str(right)
        left_pool = compounds_by_bin[compound_bins[left]]
        right_pool = proteins_by_bin[protein_bins[right]]
        selected = None
        for _ in range(int(maximum_attempts)):
            candidate = (
                str(left_pool[rng.randint(len(left_pool))]),
                str(right_pool[rng.randint(len(right_pool))]),
            )
            if candidate not in positives and candidate not in used:
                selected = candidate
                break
        if selected is None:
            unmatched += 1
            continue
        used.add(selected)
        matched_positives.append((left, right))
        negatives.append(selected)
    return matched_positives, negatives, {
        'requested': len(positive_pairs),
        'matched': len(negatives),
        'unmatched': unmatched,
        'coverage': len(negatives) / float(len(positive_pairs)) if positive_pairs else 0.0,
        'bin_count': int(bin_count),
        'seed': int(seed),
    }
