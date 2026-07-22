import unittest

from tools.audit_chcr_donor_controls import comparison_decision


def analysis(
        margin,
        aupr_drop,
        win_rate=0.7,
        coverage=0.95,
        strata=1.0):
    return {
        'coverage': {'fraction': coverage},
        'positive_pairs': {
            'pair_win_rate': win_rate,
            'mean_margin': margin,
        },
        'counterfactual_AUPR': {
            'mean_factual_minus_counterfactual': aupr_drop,
        },
        'degree_strata_positive_fraction': strata,
    }


class ChcrDonorControlDecisionTest(unittest.TestCase):
    def test_supports_degree_and_overlap_specificity(self):
        result = comparison_decision({
            'random': analysis(3.0, 0.05),
            'exact_degree': analysis(1.0, 0.01),
            'exact_degree_disjoint': analysis(2.0, 0.03),
        }, degree_control_overlap_fraction=0.25)

        self.assertEqual(
            result['decision'],
            'supports_context_specificity_beyond_degree_and_overlap',
        )
        self.assertTrue(all(result['criteria'].values()))

    def test_separates_degree_evidence_from_overlap_evidence(self):
        result = comparison_decision({
            'random': analysis(3.0, 0.05),
            'exact_degree': analysis(2.5, 0.04),
            'exact_degree_disjoint': analysis(2.0, 0.03),
        }, degree_control_overlap_fraction=0.25)

        self.assertEqual(
            result['decision'],
            'supports_context_specificity_beyond_degree_disjoint_confirmed',
        )
        self.assertTrue(result['criteria']['disjoint_positive_mean_margin'])
        self.assertFalse(
            result['criteria']['disjoint_margin_exceeds_degree_control']
        )

    def test_rejects_weak_disjoint_signal(self):
        result = comparison_decision({
            'random': analysis(1.0, 0.02),
            'exact_degree': analysis(0.1, 0.0005),
            'exact_degree_disjoint': analysis(
                0.05, 0.0004, win_rate=0.55, strata=0.5
            ),
        }, degree_control_overlap_fraction=0.25)

        self.assertEqual(
            result['decision'],
            'does_not_support_context_specificity_beyond_degree',
        )

    def test_low_common_coverage_is_inconclusive(self):
        result = comparison_decision({
            'random': analysis(3.0, 0.05, coverage=0.5),
            'exact_degree': analysis(1.0, 0.01, coverage=0.5),
            'exact_degree_disjoint': analysis(2.0, 0.03, coverage=0.5),
        }, degree_control_overlap_fraction=0.25)

        self.assertEqual(
            result['decision'], 'inconclusive_degree_control_coverage'
        )

    def test_low_overlap_control_coverage_is_inconclusive_for_overlap(self):
        result = comparison_decision({
            'random': analysis(3.0, 0.05),
            'exact_degree': analysis(1.0, 0.01),
            'exact_degree_disjoint': analysis(2.0, 0.03),
        }, degree_control_overlap_fraction=0.02)

        self.assertEqual(
            result['decision'],
            'supports_context_specificity_beyond_degree_disjoint_confirmed_overlap_inconclusive',
        )
        self.assertFalse(
            result['criteria']['degree_control_overlap_coverage']
        )

    def test_low_disjoint_coverage_preserves_degree_evidence(self):
        result = comparison_decision({
            'random': analysis(3.0, 0.05, coverage=0.7),
            'exact_degree': analysis(1.0, 0.01, coverage=0.7),
            'exact_degree_disjoint': analysis(2.0, 0.03, coverage=0.7),
        }, primary_degree_analysis=analysis(1.0, 0.01, coverage=0.95),
           degree_control_overlap_fraction=0.25)

        self.assertEqual(
            result['decision'],
            'supports_context_specificity_beyond_degree_disjoint_coverage_inconclusive',
        )


if __name__ == '__main__':
    unittest.main()
