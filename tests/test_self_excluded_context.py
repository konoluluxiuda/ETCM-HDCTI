import unittest

import numpy as np

from util.self_excluded_context import (
    build_direct_self_excluded_contexts,
    herb_protein_context_logits,
    l2_normalize_rows,
)


class SelfExcludedContextTests(unittest.TestCase):
    def test_direct_self_exclusion_matches_layerwise_definition(self):
        layer_inputs = np.asarray([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 3.0],
            ],
        ], dtype=np.float64)
        herb_members = {
            0: np.asarray([0, 1]),
            1: np.asarray([1, 2]),
            2: np.asarray([3]),
        }
        compound_herbs = {
            0: np.asarray([0]),
            1: np.asarray([0, 1]),
            2: np.asarray([1]),
            3: np.asarray([2]),
        }
        herb_edges = []
        for herb_index in range(3):
            members = herb_members[herb_index]
            layer_means = np.mean(layer_inputs[:, members, :], axis=1)
            herb_edges.append(np.sum(l2_normalize_rows(layer_means), axis=0))
        result = build_direct_self_excluded_contexts(
            np.asarray(herb_edges),
            layer_inputs,
            herb_members,
            compound_herbs,
            num_compounds=4,
        )

        expected_c0 = l2_normalize_rows(np.asarray([np.sum(
            l2_normalize_rows(layer_inputs[:, 1, :]), axis=0
        )]))[0]
        expected_c1 = l2_normalize_rows(np.asarray([
            np.sum(l2_normalize_rows(layer_inputs[:, 0, :]), axis=0)
            + np.sum(l2_normalize_rows(layer_inputs[:, 2, :]), axis=0)
        ]))[0]
        np.testing.assert_allclose(
            result['self_excluded_contexts'][0], expected_c0, atol=1e-6
        )
        np.testing.assert_allclose(
            result['self_excluded_contexts'][1], expected_c1, atol=1e-6
        )
        self.assertTrue(result['eligible'][0])
        self.assertTrue(result['eligible'][1])
        self.assertTrue(result['eligible'][2])
        self.assertFalse(result['eligible'][3])
        np.testing.assert_array_equal(
            result['self_excluded_contexts'][3], np.zeros(2)
        )
        self.assertLess(
            result['edge_reconstruction_relative_error']['maximum'], 1e-12
        )

    def test_herb_protein_context_logits(self):
        contexts = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        proteins = np.asarray([[2.0, 1.0], [1.0, 3.0]])
        logits = herb_protein_context_logits(
            contexts,
            proteins,
            np.asarray([0, 1]),
            np.asarray([1, 0]),
            np.asarray([0.5, 2.0]),
        )
        np.testing.assert_allclose(logits, np.asarray([12.5, 11.0]))


if __name__ == '__main__':
    unittest.main()
