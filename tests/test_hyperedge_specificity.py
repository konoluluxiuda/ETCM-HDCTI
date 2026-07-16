import unittest

import numpy as np
from scipy import sparse

from util.hyperedge_specificity import (
    aggregate_hyperedge_contexts,
    context_change_statistics,
    hyperedge_specificity_weights,
    specificity_pair_features,
)


class HyperedgeSpecificityTest(unittest.TestCase):
    def setUp(self):
        self.incidence = sparse.csr_matrix(
            np.asarray([
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 0],
            ], dtype=np.float32)
        )
        self.edge_embeddings = np.asarray([
            [1, 0],
            [0, 1],
            [0, -1],
        ], dtype=np.float32)

    def test_specificity_downweights_broad_hyperedges(self):
        weights, degrees = hyperedge_specificity_weights(self.incidence)
        self.assertEqual(degrees.tolist(), [4.0, 1.0, 1.0])
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(weights[1], weights[2])

    def test_aggregation_changes_mixed_node_context(self):
        weights, _ = hyperedge_specificity_weights(self.incidence)
        uniform = aggregate_hyperedge_contexts(
            self.incidence, self.edge_embeddings
        )
        specific = aggregate_hyperedge_contexts(
            self.incidence, self.edge_embeddings, weights
        )
        self.assertGreater(specific[0, 1], uniform[0, 1])
        np.testing.assert_allclose(
            np.linalg.norm(specific, axis=1), np.ones(4), atol=1e-6
        )

    def test_context_statistics_report_broad_mass_reduction(self):
        weights, _ = hyperedge_specificity_weights(self.incidence)
        uniform = aggregate_hyperedge_contexts(
            self.incidence, self.edge_embeddings
        )
        specific = aggregate_hyperedge_contexts(
            self.incidence, self.edge_embeddings, weights
        )
        stats = context_change_statistics(
            self.incidence, weights, uniform, specific
        )
        self.assertEqual(stats['coverage'], 1.0)
        self.assertGreater(stats['weight_cv'], 0.0)
        self.assertGreater(stats['mean_cosine_distance'], 0.0)
        self.assertGreater(stats['broad_mass_relative_reduction'], 0.0)

    def test_pair_features_return_replacement_and_cross_context_terms(self):
        compound = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        protein = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        uniform_context = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        specific_context = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
        features = specificity_pair_features(
            compound,
            protein,
            uniform_context,
            specific_context,
            specific_context,
            np.asarray([0, 1]),
            np.asarray([0, 1]),
            np.ones(2, dtype=np.float32),
        )
        self.assertEqual(set(features), {
            'herb_specificity_replacement_delta',
            'compound_specific_disease_cosine',
            'specific_context_cosine',
        })
        for values in features.values():
            self.assertEqual(values.shape, (2,))


if __name__ == '__main__':
    unittest.main()
