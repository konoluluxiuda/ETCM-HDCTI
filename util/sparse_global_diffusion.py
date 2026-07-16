import numpy as np
from scipy import sparse


def build_incidence_matrix(
        path,
        node_map,
        context_map,
        node_column,
        context_column):
    rows = []
    columns = []
    skipped = 0
    malformed = 0
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) <= max(node_column, context_column):
                malformed += 1
                continue
            node_key = str(parts[node_column])
            context_key = str(parts[context_column])
            if node_key not in node_map or context_key not in context_map:
                skipped += 1
                continue
            rows.append(int(node_map[node_key]))
            columns.append(int(context_map[context_key]))
    matrix = sparse.coo_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(len(node_map), len(context_map)),
        dtype=np.float32,
    ).tocsr()
    if matrix.nnz:
        matrix.data.fill(1.0)
        matrix.sum_duplicates()
        matrix.data.fill(1.0)
    return matrix, {
        'relations': int(matrix.nnz),
        'skipped': int(skipped),
        'malformed': int(malformed),
        'covered_nodes': int(np.sum(np.diff(matrix.indptr) > 0)),
        'covered_contexts': int(np.sum(np.asarray(matrix.sum(axis=0)).reshape(-1) > 0)),
    }


def normalized_two_step_transition(incidence):
    incidence = sparse.csr_matrix(incidence, dtype=np.float32)
    node_degree = np.asarray(incidence.sum(axis=1)).reshape(-1)
    context_degree = np.asarray(incidence.sum(axis=0)).reshape(-1)
    node_scale = np.divide(
        1.0,
        node_degree,
        out=np.zeros_like(node_degree, dtype=np.float32),
        where=node_degree > 0,
    )
    context_scale = np.divide(
        1.0,
        context_degree,
        out=np.zeros_like(context_degree, dtype=np.float32),
        where=context_degree > 0,
    )
    node_to_context = sparse.diags(node_scale).dot(incidence)
    context_to_node = sparse.diags(context_scale).dot(incidence.transpose())
    transition = node_to_context.dot(context_to_node).tocsr()
    transition.sum_duplicates()
    transition.eliminate_zeros()
    return transition


def prune_csr_topk(matrix, top_k, remove_diagonal=False):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError('top_k must be positive.')
    if remove_diagonal:
        matrix = matrix.copy()
        matrix.setdiag(0)
        matrix.eliminate_zeros()
    row_parts = []
    column_parts = []
    value_parts = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        columns = matrix.indices[start:end]
        values = matrix.data[start:end]
        if len(values) > top_k:
            selected = np.argpartition(values, len(values) - top_k)[-top_k:]
            columns = columns[selected]
            values = values[selected]
        if len(values):
            order = np.lexsort((columns, -values))
            row_parts.append(np.full(len(order), row, dtype=np.int64))
            column_parts.append(columns[order].astype(np.int64, copy=False))
            value_parts.append(values[order].astype(np.float32, copy=False))
    if not value_parts:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    return sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    ).tocsr()


def normalize_sparse_rows(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    scales = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float32),
        where=row_sums > 0,
    )
    return sparse.diags(scales).dot(matrix).tocsr()


def sparse_multihop_diffusion(
        incidence,
        top_k=20,
        minimum_hop=2,
        maximum_hop=4,
        restart=0.15,
        candidate_multiplier=4):
    minimum_hop = int(minimum_hop)
    maximum_hop = int(maximum_hop)
    top_k = int(top_k)
    candidate_multiplier = int(candidate_multiplier)
    restart = float(restart)
    if minimum_hop < 2:
        raise ValueError('minimum_hop must be at least 2 to exclude the local one-hop view.')
    if maximum_hop < minimum_hop:
        raise ValueError('maximum_hop must be at least minimum_hop.')
    if not 0.0 < restart < 1.0:
        raise ValueError('restart must be between 0 and 1.')
    if candidate_multiplier <= 0:
        raise ValueError('candidate_multiplier must be positive.')

    transition = normalized_two_step_transition(incidence)
    candidate_k = max(top_k, top_k * candidate_multiplier)
    current = transition
    accumulated = sparse.csr_matrix(transition.shape, dtype=np.float32)
    hop_nnz = {}
    for hop in range(2, maximum_hop + 1):
        current = current.dot(transition).tocsr()
        current = prune_csr_topk(current, candidate_k, remove_diagonal=False)
        current = normalize_sparse_rows(current)
        hop_nnz[str(hop)] = int(current.nnz)
        if hop >= minimum_hop:
            weight = restart * ((1.0 - restart) ** (hop - minimum_hop))
            accumulated = accumulated + np.float32(weight) * current

    diffusion = prune_csr_topk(
        accumulated, top_k, remove_diagonal=True
    )
    diffusion = normalize_sparse_rows(diffusion)
    transition_without_self = transition.copy()
    transition_without_self.setdiag(0)
    transition_without_self.eliminate_zeros()
    statistics = diffusion_statistics(diffusion, transition_without_self)
    statistics.update({
        'top_k': top_k,
        'minimum_hop': minimum_hop,
        'maximum_hop': maximum_hop,
        'restart': restart,
        'candidate_multiplier': candidate_multiplier,
        'transition_nnz': int(transition_without_self.nnz),
        'hop_nnz': hop_nnz,
    })
    return diffusion, transition_without_self, statistics


def diffusion_statistics(diffusion, direct_transition):
    diffusion = sparse.csr_matrix(diffusion)
    direct_transition = sparse.csr_matrix(direct_transition)
    counts = np.diff(diffusion.indptr)
    covered = counts > 0
    coo = diffusion.tocoo()
    if coo.nnz:
        width = diffusion.shape[1]
        diffusion_ids = coo.row.astype(np.int64) * width + coo.col.astype(np.int64)
        direct = direct_transition.tocoo()
        direct_ids = direct.row.astype(np.int64) * width + direct.col.astype(np.int64)
        novel = ~np.isin(diffusion_ids, direct_ids, assume_unique=False)
        novel_fraction = float(np.mean(novel))
    else:
        novel_fraction = 0.0
    return {
        'nodes': int(diffusion.shape[0]),
        'edges': int(diffusion.nnz),
        'coverage': float(np.mean(covered)) if len(covered) else 0.0,
        'mean_neighbors': float(np.mean(counts[covered])) if np.any(covered) else 0.0,
        'novel_edge_fraction': novel_fraction,
    }


def diffuse_embeddings(diffusion, embeddings):
    values = np.asarray(embeddings, dtype=np.float32)
    if diffusion.shape[1] != values.shape[0]:
        raise ValueError('Diffusion matrix and embedding rows do not match.')
    result = np.asarray(diffusion.dot(values), dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return np.divide(
        result,
        norms,
        out=np.zeros_like(result),
        where=norms > 0,
    )


def global_pair_features(
        compound_embeddings,
        protein_embeddings,
        global_compound_embeddings,
        global_protein_embeddings,
        compound_indices,
        protein_indices):
    def normalize(values):
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(
            values,
            norms,
            out=np.zeros_like(values),
            where=norms > 0,
        )

    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein indices must have matching shapes.')
    compounds = normalize(compound_embeddings)[compound_indices]
    proteins = normalize(protein_embeddings)[protein_indices]
    global_compounds = normalize(global_compound_embeddings)[compound_indices]
    global_proteins = normalize(global_protein_embeddings)[protein_indices]
    return {
        'compound_global_cosine': np.sum(global_compounds * proteins, axis=1),
        'protein_global_cosine': np.sum(compounds * global_proteins, axis=1),
        'dual_global_cosine': np.sum(global_compounds * global_proteins, axis=1),
    }
