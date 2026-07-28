import unittest

import numpy as np

from util.lightgcn import build_normalized_bipartite_adjacency


class LightGCNUtilitiesTest(unittest.TestCase):
    def test_normalized_bipartite_adjacency_is_symmetric_and_deduplicated(self):
        graph = build_normalized_bipartite_adjacency(
            2,
            1,
            [(0, 0), (1, 0), (0, 0)],
        )
        self.assertEqual(graph['edge_count'], 2)
        self.assertEqual(graph['shape'], (3, 3))
        self.assertEqual(graph['active_compounds'], 2)
        self.assertEqual(graph['active_proteins'], 1)
        entries = {
            tuple(index): value
            for index, value in zip(graph['indices'], graph['values'])
        }
        expected_weight = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(entries[(0, 2)], expected_weight)
        self.assertAlmostEqual(entries[(2, 0)], expected_weight)
        self.assertAlmostEqual(entries[(1, 2)], expected_weight)
        self.assertAlmostEqual(entries[(2, 1)], expected_weight)

    def test_empty_or_out_of_range_graph_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no positive'):
            build_normalized_bipartite_adjacency(2, 2, [])
        with self.assertRaisesRegex(ValueError, 'Compound index'):
            build_normalized_bipartite_adjacency(2, 2, [(2, 0)])
        with self.assertRaisesRegex(ValueError, 'Protein index'):
            build_normalized_bipartite_adjacency(2, 2, [(0, 2)])


if __name__ == '__main__':
    unittest.main()
