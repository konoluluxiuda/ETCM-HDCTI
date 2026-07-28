import unittest

import numpy as np

from util.hgt import build_hgt_relations


class HGTUtilitiesTest(unittest.TestCase):
    def _build_graph(self, max_neighbors=2, seed=2026):
        return build_hgt_relations(
            5,
            4,
            4,
            3,
            [
                (0, 0), (1, 0), (2, 0), (3, 0),
                (1, 1), (2, 2), (4, 3),
            ],
            [
                (0, 0), (0, 1), (0, 2), (0, 3),
                (1, 0), (2, 1), (3, 2),
            ],
            [
                (0, 0), (1, 0), (2, 0), (3, 0),
                (0, 1), (1, 2),
            ],
            max_neighbors=max_neighbors,
            seed=seed,
        )

    def test_sampling_is_deterministic_and_capped_per_destination(self):
        first = self._build_graph()
        second = self._build_graph()

        self.assertEqual(len(first['relations']), 6)
        self.assertEqual(
            first['sampling'],
            {
                'mode': 'deterministic_relation_destination_cap',
                'max_neighbors': 2,
                'seed': 2026,
            },
        )
        for relation_name, relation in first['relations'].items():
            with self.subTest(relation=relation_name):
                destination_counts = np.bincount(
                    relation['destination_indices'],
                    minlength=relation['shape'][0],
                )
                self.assertLessEqual(int(destination_counts.max()), 2)
                np.testing.assert_array_equal(
                    relation['source_indices'],
                    second['relations'][relation_name]['source_indices'],
                )
                np.testing.assert_array_equal(
                    relation['destination_indices'],
                    second['relations'][relation_name][
                        'destination_indices'
                    ],
                )
                self.assertEqual(
                    relation['edge_sha256'],
                    second['relations'][relation_name]['edge_sha256'],
                )

        self.assertEqual(
            first['source_edge_counts'],
            {'H_C': 7, 'C_P': 7, 'P_D': 6},
        )
        self.assertLess(
            first['sampled_directed_edges'],
            first['original_directed_edges'],
        )

    def test_zero_cap_keeps_all_edges_and_negative_cap_is_rejected(self):
        graph = self._build_graph(max_neighbors=0)
        self.assertEqual(graph['sampling']['mode'], 'all_edges')
        self.assertEqual(
            graph['sampled_directed_edges'],
            graph['original_directed_edges'],
        )

        with self.assertRaisesRegex(
                ValueError, 'hgt.max.neighbors must be non-negative'):
            self._build_graph(max_neighbors=-1)


if __name__ == '__main__':
    unittest.main()
