import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from util.sparse_global_diffusion import (
    build_incidence_matrix,
    diffuse_embeddings,
    global_pair_features,
    normalized_two_step_transition,
    prune_csr_topk,
    sparse_multihop_diffusion,
)


class SparseGlobalDiffusionTest(unittest.TestCase):
    def test_build_incidence_matrix_deduplicates_and_skips_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'relation.txt'
            path.write_text('h0 c0\nh0 c0\nh1 c1\nh2 c0\n', encoding='utf-8')
            matrix, stats = build_incidence_matrix(
                path,
                {'c0': 0, 'c1': 1},
                {'h0': 0, 'h1': 1},
                node_column=1,
                context_column=0,
            )
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix.nnz, 2)
        self.assertEqual(stats['skipped'], 1)
        self.assertEqual(stats['covered_nodes'], 2)

    def test_two_step_transition_is_row_stochastic(self):
        incidence = sparse.csr_matrix(
            np.asarray([
                [1, 0],
                [1, 1],
                [0, 1],
            ], dtype=np.float32)
        )
        transition = normalized_two_step_transition(incidence)
        np.testing.assert_allclose(
            np.asarray(transition.sum(axis=1)).reshape(-1),
            np.ones(3),
            atol=1e-6,
        )

    def test_topk_pruning_is_deterministic_and_removes_diagonal(self):
        matrix = sparse.csr_matrix(
            np.asarray([
                [9, 3, 2, 1],
                [4, 9, 8, 2],
                [7, 6, 9, 5],
                [1, 2, 3, 9],
            ], dtype=np.float32)
        )
        pruned = prune_csr_topk(matrix, 2, remove_diagonal=True)
        self.assertTrue(np.all(np.diff(pruned.indptr) <= 2))
        np.testing.assert_allclose(pruned.diagonal(), np.zeros(4))
        self.assertEqual(pruned[0].indices.tolist(), [1, 2])

    def test_multihop_diffusion_excludes_self_and_limits_neighbors(self):
        incidence = sparse.csr_matrix(
            np.asarray([
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ], dtype=np.float32)
        )
        diffusion, _, stats = sparse_multihop_diffusion(
            incidence,
            top_k=2,
            minimum_hop=2,
            maximum_hop=3,
            restart=0.15,
            candidate_multiplier=2,
        )
        self.assertTrue(np.all(np.diff(diffusion.indptr) <= 2))
        np.testing.assert_allclose(diffusion.diagonal(), np.zeros(4))
        row_sums = np.asarray(diffusion.sum(axis=1)).reshape(-1)
        np.testing.assert_allclose(row_sums[row_sums > 0], 1.0, atol=1e-6)
        self.assertGreater(stats['novel_edge_fraction'], 0.0)

    def test_diffused_pair_features_have_expected_shape(self):
        diffusion = sparse.csr_matrix(
            np.asarray([[0, 1], [1, 0]], dtype=np.float32)
        )
        compound = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        protein = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        global_compound = diffuse_embeddings(diffusion, compound)
        global_protein = diffuse_embeddings(diffusion, protein)
        features = global_pair_features(
            compound,
            protein,
            global_compound,
            global_protein,
            np.asarray([0, 1]),
            np.asarray([0, 1]),
        )
        self.assertEqual(set(features), {
            'compound_global_cosine',
            'protein_global_cosine',
            'dual_global_cosine',
        })
        for values in features.values():
            self.assertEqual(values.shape, (2,))


if __name__ == '__main__':
    unittest.main()
