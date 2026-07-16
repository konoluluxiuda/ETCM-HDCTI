import unittest

import numpy as np

from util.compound_ranking import (
    binary_labels_from_records,
    bootstrap_mean_interval,
    compound_group_ranking,
    degree_stratified_ranking,
)


class CompoundRankingTest(unittest.TestCase):
    def test_binary_labels_are_read_from_record_column(self):
        labels = binary_labels_from_records([
            ['c0', 'p0', '1'],
            ['c1', 'p1', 0.0],
            ['c2', 'p2', 2],
        ])
        np.testing.assert_array_equal(labels, np.asarray([1, 0, 1]))

    def test_group_ranking_measures_pairwise_violations_and_top1(self):
        compounds = ['c0', 'c0', 'c0', 'c1', 'c1', 'c2']
        labels = [1, 0, 0, 1, 0, 1]
        scores = [0.9, 0.2, 0.1, 0.3, 0.8, 0.5]
        summary, rows = compound_group_ranking(compounds, labels, scores)
        self.assertEqual(summary['all_compounds'], 3)
        self.assertEqual(summary['eligible_compounds'], 2)
        self.assertEqual(summary['eligible_records'], 5)
        self.assertAlmostEqual(summary['macro_violation_rate'], 0.5)
        self.assertAlmostEqual(summary['top1_miss_rate'], 0.5)
        by_id = {row['compound_id']: row for row in rows}
        self.assertEqual(by_id['c0']['first_positive_rank'], 1)
        self.assertEqual(by_id['c1']['first_positive_rank'], 2)

    def test_ties_count_as_half_accuracy(self):
        summary, _ = compound_group_ranking(
            ['c0', 'c0'], [1, 0], [0.5, 0.5]
        )
        self.assertAlmostEqual(summary['macro_pairwise_accuracy'], 0.5)
        self.assertAlmostEqual(summary['macro_violation_rate'], 0.5)

    def test_bootstrap_interval_is_deterministic(self):
        values = np.asarray([0, 0, 1, 1], dtype=np.float64)
        first = bootstrap_mean_interval(values, draws=100, seed=7)
        second = bootstrap_mean_interval(values, draws=100, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first['lower'], first['mean'])
        self.assertGreaterEqual(first['upper'], first['mean'])

    def test_degree_strata_use_training_positive_degree(self):
        _, rows = compound_group_ranking(
            ['c0', 'c0', 'c1', 'c1'], [1, 0, 1, 0], [0.9, 0.1, 0.2, 0.8]
        )
        strata = degree_stratified_ranking(rows, {'c0': 2, 'c1': 9})
        self.assertEqual([row['degree_bin'] for row in strata], ['2-3', '8-15'])
        self.assertEqual([row['compounds'] for row in strata], [1, 1])


if __name__ == '__main__':
    unittest.main()
