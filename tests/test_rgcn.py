import unittest

from util.rgcn import build_rgcn_relations


class RGCNUtilitiesTest(unittest.TestCase):
    def test_relations_are_directed_normalized_and_deduplicated(self):
        graph = build_rgcn_relations(
            2,
            2,
            2,
            2,
            [(0, 0), (1, 0), (0, 0)],
            [(0, 0), (1, 1)],
            [(0, 0), (1, 0)],
        )
        self.assertEqual(len(graph['relations']), 6)
        self.assertEqual(
            graph['source_edge_counts'],
            {'H_C': 2, 'C_P': 2, 'P_D': 2},
        )

        herb_to_compound = graph['relations']['herb_to_compound']
        self.assertEqual(herb_to_compound['shape'], (2, 2))
        entries = {
            tuple(index): value
            for index, value in zip(
                herb_to_compound['indices'],
                herb_to_compound['values'],
            )
        }
        self.assertAlmostEqual(entries[(0, 0)], 0.5)
        self.assertAlmostEqual(entries[(0, 1)], 0.5)

        compound_to_herb = graph['relations']['compound_to_herb']
        reverse_entries = {
            tuple(index): value
            for index, value in zip(
                compound_to_herb['indices'],
                compound_to_herb['values'],
            )
        }
        self.assertAlmostEqual(reverse_entries[(0, 0)], 1.0)
        self.assertAlmostEqual(reverse_entries[(1, 0)], 1.0)

    def test_empty_or_out_of_range_relations_are_rejected(self):
        with self.assertRaisesRegex(
                ValueError, 'compound_to_protein contains no edge'):
            build_rgcn_relations(
                1, 1, 1, 1,
                [(0, 0)],
                [],
                [(0, 0)],
            )
        with self.assertRaisesRegex(ValueError, 'source index'):
            build_rgcn_relations(
                1, 1, 1, 1,
                [(1, 0)],
                [(0, 0)],
                [(0, 0)],
            )


if __name__ == '__main__':
    unittest.main()
