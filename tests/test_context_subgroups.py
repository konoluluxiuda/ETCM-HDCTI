import unittest

import numpy as np

from util.context_subgroups import (
    build_subgroup_report,
    herb_degree_bin,
    mention_count_bin,
    select_f1_threshold,
    summarize_subset,
    training_cp_degree_bin,
)


class ContextSubgroupsTest(unittest.TestCase):
    def test_f1_threshold_is_selected_from_validation_scores(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.9, 0.4, 0.6, 0.1])
        logits = np.log(scores / (1.0 - scores))

        result = select_f1_threshold(labels, logits)

        self.assertAlmostEqual(result['threshold'], 0.4)
        self.assertAlmostEqual(result['validation_metrics']['F1-score'], 0.8)

    def test_f1_threshold_requires_both_classes(self):
        with self.assertRaisesRegex(ValueError, 'both positive and negative'):
            select_f1_threshold([1, 1], [1.0, 2.0])

    def test_fixed_bins_cover_zero_low_and_high_values(self):
        self.assertEqual(herb_degree_bin(0), '0')
        self.assertEqual(herb_degree_bin(1), '1')
        self.assertEqual(herb_degree_bin(3), '2-3')
        self.assertEqual(herb_degree_bin(10), '4-10')
        self.assertEqual(herb_degree_bin(11), '>10')
        self.assertEqual(training_cp_degree_bin(0), '0')
        self.assertEqual(training_cp_degree_bin(2), '1-2')
        self.assertEqual(training_cp_degree_bin(8), '6-10')
        self.assertEqual(mention_count_bin(None), 'missing')
        self.assertEqual(mention_count_bin(10), '10-19')
        self.assertEqual(mention_count_bin(100), '>=100')

    def test_summary_tracks_precision_recall_transitions(self):
        labels = np.asarray([1, 1, 0, 0])
        baseline = np.asarray([1.0, -1.0, 1.0, -1.0])
        herb_total = np.asarray([-1.0, 1.0, -1.0, 1.0])
        result = summarize_subset(
            labels,
            baseline,
            herb_total - 0.2,
            herb_total,
            np.full(4, 0.2),
        )

        self.assertEqual(result['prediction_transitions'], {
            'FN_to_TP': 1,
            'TP_to_FN': 1,
            'FP_to_TN': 1,
            'TN_to_FP': 1,
        })
        self.assertEqual(result['records'], 4)
        self.assertAlmostEqual(result['context_logit']['mean_abs'], 0.2)

    def test_summary_uses_separate_calibrated_thresholds(self):
        labels = np.asarray([1, 0])
        scores = np.asarray([0.4, 0.6])
        logits = np.log(scores / (1.0 - scores))

        result = summarize_subset(
            labels,
            logits,
            logits,
            logits,
            np.zeros(2),
            baseline_threshold=0.5,
            herb_threshold=0.3,
        )

        self.assertEqual(result['thresholds'], {'baseline': 0.5, 'herb': 0.3})
        self.assertEqual(result['prediction_transitions']['FN_to_TP'], 1)
        self.assertEqual(result['baseline']['F1-score'], 0.0)
        self.assertAlmostEqual(result['herb_total']['F1-score'], 2.0 / 3.0)

    def test_subgroup_report_splits_rows_without_changing_overall_count(self):
        labels = np.asarray([1, 0, 1, 0, 1, 0])
        baseline = np.asarray([2, -2, 1, -1, 0.5, -0.5], dtype=float)
        context = np.asarray([0.2, -0.1, 0.3, -0.2, 0.1, -0.3])
        report = build_subgroup_report(
            labels,
            baseline,
            baseline,
            baseline + context,
            context,
            {'hc_degree': ['1', '1', '2-3', '2-3', '>10', '>10']},
        )

        self.assertEqual(report['overall']['records'], 6)
        self.assertEqual(
            sum(row['records'] for row in report['subgroups']['hc_degree'].values()),
            6,
        )

    def test_mismatched_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same length'):
            summarize_subset([1, 0], [1], [1, 0], [1, 0], [0, 0])


if __name__ == '__main__':
    unittest.main()
