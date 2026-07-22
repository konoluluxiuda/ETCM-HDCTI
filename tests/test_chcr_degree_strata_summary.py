import unittest

from tools.summarize_chcr_degree_strata import aggregate_rows


def row(fold, margin, win, pairs=10, analyzable=True):
    return {
        'dataset': 'Synthetic',
        'slug': 'synthetic',
        'fold': fold,
        'subgroup': 'H-C degree',
        'group': '1',
        'pairs': pairs,
        'compounds': pairs,
        'mean_margin': margin,
        'median_margin': margin,
        'margin_std': 0.1,
        'pair_win_rate': win,
        'compound_win_rate': win,
        'compound_mean_margin': margin,
        'analyzable': analyzable,
        'report': 'report.json',
    }


class DegreeStrataSummaryTest(unittest.TestCase):
    def test_four_of_five_positive_folds_pass_frozen_consistency(self):
        rows = [
            row(1, 0.4, 0.8),
            row(2, 0.3, 0.7),
            row(3, 0.2, 0.7),
            row(4, 0.1, 0.6),
            row(5, -0.1, 0.4),
        ]
        summary = aggregate_rows(rows)[0]
        self.assertEqual(summary['positive_margin_folds'], 4)
        self.assertEqual(summary['positive_fold_fraction'], 0.8)
        self.assertTrue(summary['frozen_direction_consistency'])

    def test_three_of_five_positive_folds_fail_frozen_consistency(self):
        rows = [
            row(1, 0.4, 0.8),
            row(2, 0.3, 0.8),
            row(3, 0.2, 0.8),
            row(4, -0.1, 0.8),
            row(5, -0.2, 0.8),
        ]
        summary = aggregate_rows(rows)[0]
        self.assertEqual(summary['positive_fold_fraction'], 0.6)
        self.assertFalse(summary['frozen_direction_consistency'])

    def test_pair_weighted_metrics_use_positive_pair_counts(self):
        rows = [
            row(1, 1.0, 0.9, pairs=10),
            row(2, -0.2, 0.5, pairs=30),
        ]
        summary = aggregate_rows(
            rows,
            minimum_positive_fold_fraction=0.0,
            reference_pair_win_rate=0.0,
        )[0]
        self.assertAlmostEqual(summary['pair_weighted_margin'], 0.1)
        self.assertAlmostEqual(summary['pair_weighted_pair_win'], 0.6)

    def test_non_analyzable_fold_is_excluded(self):
        rows = [
            row(1, 0.3, 0.7),
            row(2, -1.0, 0.0, analyzable=False),
        ]
        summary = aggregate_rows(rows)[0]
        self.assertEqual(summary['folds'], 2)
        self.assertEqual(summary['analyzable_folds'], 1)
        self.assertEqual(summary['positive_fold_fraction'], 1.0)

    def test_pair_win_is_descriptive_not_part_of_direction_consistency(self):
        rows = [row(fold, 0.1, 0.4) for fold in range(1, 6)]
        summary = aggregate_rows(rows)[0]
        self.assertTrue(summary['frozen_direction_consistency'])
        self.assertFalse(summary['pair_win_reference_met'])


if __name__ == '__main__':
    unittest.main()
