import unittest

import numpy as np

from util.counterfactual_context import (
    build_counterfactual_donors,
    build_exact_degree_counterfactuals,
    summarize_donor_assignments,
    summarize_counterfactual_audit,
    wilson_interval,
)
from util.model_components import counterfactual_margin_values


class CounterfactualContextTest(unittest.TestCase):
    def test_separate_donor_universe_keeps_external_candidates(self):
        memberships = {
            'a': {'h1'},
            'b': {'h1'},
            'c': {'h2'},
        }

        result = build_counterfactual_donors(
            ['a'],
            memberships,
            draws=2,
            seed=7,
            strategy='exact_degree_disjoint',
            donor_compound_ids=memberships.keys(),
        )

        self.assertEqual(result['assignments']['a'], ['c', 'c'])

    def test_assignment_filter_preserves_selected_rng_assignments(self):
        memberships = {
            'a': {'h1'},
            'b': {'h2'},
            'c': {'h3'},
            'd': {'h4'},
        }
        full = build_counterfactual_donors(
            memberships,
            memberships,
            draws=3,
            seed=17,
            strategy='exact_degree_disjoint',
        )
        filtered = build_counterfactual_donors(
            memberships,
            memberships,
            draws=3,
            seed=17,
            strategy='exact_degree_disjoint',
            assignment_compound_ids=['c'],
        )

        self.assertEqual(
            filtered['assignments']['c'], full['assignments']['c']
        )
        self.assertEqual(set(filtered['assignments']), {'c'})

    def test_margin_values_apply_only_to_eligible_positives(self):
        values = counterfactual_margin_values(
            factual_context_logits=[0.4, 0.1, 0.0],
            counterfactual_context_logits=[0.3, 0.4, -1.0],
            labels=[1.0, 1.0, 0.0],
            eligible_mask=[1.0, 0.0, 1.0],
            margin=0.2,
        )
        np.testing.assert_allclose(values, [0.1, 0.0, 0.0])

    def test_matching_requires_equal_degree_and_disjoint_herbs(self):
        memberships = {
            'c0': {'h0', 'h1'},
            'c1': {'h2', 'h3'},
            'c2': {'h1', 'h4'},
            'c3': {'h5'},
        }

        result = build_exact_degree_counterfactuals(
            memberships.keys(), memberships, draws=3, seed=7
        )

        self.assertEqual(result['assignments']['c0'], ['c1', 'c1', 'c1'])
        self.assertEqual(len(result['assignments']['c1']), 3)
        self.assertTrue(set(result['assignments']['c1']).issubset({'c0', 'c2'}))
        self.assertNotIn('c3', result['assignments'])
        for source, donors in result['assignments'].items():
            for donor in donors:
                self.assertEqual(len(memberships[source]), len(memberships[donor]))
                self.assertFalse(memberships[source] & memberships[donor])

    def test_matching_is_deterministic(self):
        memberships = {
            'c0': {'h0'}, 'c1': {'h1'}, 'c2': {'h2'}, 'c3': {'h3'},
        }
        first = build_exact_degree_counterfactuals(
            memberships.keys(), memberships, draws=2, seed=11
        )
        second = build_exact_degree_counterfactuals(
            reversed(list(memberships)), memberships, draws=2, seed=11
        )
        self.assertEqual(first['assignments'], second['assignments'])

    def test_three_donor_strategies_apply_distinct_constraints(self):
        memberships = {
            'c0': {'h0', 'h1'},
            'c1': {'h0', 'h2'},
            'c2': {'h3', 'h4'},
            'c3': {'h5'},
        }
        random_donors = build_counterfactual_donors(
            memberships, memberships, draws=20, seed=3, strategy='random'
        )
        degree_donors = build_counterfactual_donors(
            memberships, memberships, draws=20, seed=3, strategy='exact_degree'
        )
        disjoint_donors = build_counterfactual_donors(
            memberships,
            memberships,
            draws=20,
            seed=3,
            strategy='exact_degree_disjoint',
        )

        self.assertIn('c3', random_donors['assignments']['c0'])
        self.assertNotIn('c3', degree_donors['assignments']['c0'])
        self.assertIn('c1', degree_donors['assignments']['c0'])
        self.assertEqual(set(disjoint_donors['assignments']['c0']), {'c2'})

    def test_assignment_summary_reports_overlap_and_degree_matching(self):
        memberships = {
            'c0': {'h0', 'h1'},
            'c1': {'h0', 'h2'},
            'c2': {'h3'},
        }
        summary = summarize_donor_assignments(
            {'c0': ['c1', 'c2']}, memberships
        )

        self.assertEqual(summary['assignments'], 2)
        self.assertEqual(summary['overlap_fraction'], 0.5)
        self.assertEqual(summary['degree_matched_fraction'], 0.5)
        self.assertAlmostEqual(summary['mean_shared_herbs'], 0.5)

    def test_summary_accepts_clear_factual_advantage(self):
        labels = np.asarray([1, 1, 0, 0])
        factual = np.asarray([3.0, 2.0, -2.0, -3.0])
        counterfactual = np.asarray([
            [1.0, 0.5, 1.5, 1.0],
            [0.8, 0.2, 1.2, 0.8],
        ])
        report = summarize_counterfactual_audit(
            labels,
            ['c0', 'c1', 'c2', 'c3'],
            factual,
            counterfactual,
            subgroup_values={'degree': ['1', '1', '1', '1']},
            minimum_group_pairs=2,
            aupr_drop_threshold=0.001,
        )

        self.assertEqual(report['decision'], 'supports_CHCR_training_pilot')
        self.assertEqual(report['positive_pairs']['pair_win_rate'], 1.0)
        self.assertGreater(report['counterfactual_AUPR']['mean_factual_minus_counterfactual'], 0)
        self.assertTrue(all(report['criteria'].values()))

    def test_low_coverage_is_inconclusive(self):
        report = summarize_counterfactual_audit(
            [1, 0],
            ['c0', 'c1'],
            [1.0, -1.0],
            [[0.0, 0.0]],
            requested_records=4,
        )
        self.assertEqual(report['decision'], 'inconclusive_counterfactual_coverage')

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(60, 100)
        self.assertLess(lower, 0.6)
        self.assertGreater(upper, 0.6)


if __name__ == '__main__':
    unittest.main()
