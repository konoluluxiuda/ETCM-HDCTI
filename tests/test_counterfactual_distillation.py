import unittest

import numpy as np

from util.counterfactual_distillation import (
    build_cross_pool_counterfactuals,
    context_pair_features,
    explicit_intercept,
    grouped_compound_holdout,
    standardize_factual_features,
)


class CounterfactualDistillationTest(unittest.TestCase):
    def test_grouped_holdout_keeps_compounds_disjoint(self):
        compounds = ['c0', 'c0', 'c1', 'c1', 'c2', 'c3']
        fit_mask, evaluation_mask = grouped_compound_holdout(
            compounds, holdout_ratio=0.5, seed=7
        )
        fit = {compounds[index] for index in np.flatnonzero(fit_mask)}
        evaluation = {
            compounds[index] for index in np.flatnonzero(evaluation_mask)
        }
        self.assertFalse(fit & evaluation)
        self.assertEqual(fit | evaluation, set(compounds))

    def test_cross_pool_matching_is_equal_degree_and_disjoint(self):
        memberships = {
            's0': {'h0', 'h1'},
            's1': {'h2'},
            'd0': {'h2', 'h3'},
            'd1': {'h1', 'h4'},
            'd2': {'h5'},
        }
        result = build_cross_pool_counterfactuals(
            ['s0', 's1'], ['d0', 'd1', 'd2'], memberships, draws=2, seed=3
        )
        self.assertEqual(result['assignments']['s0'], ['d0', 'd0'])
        self.assertEqual(result['assignments']['s1'], ['d2', 'd2'])

    def test_pair_features_and_difference_intercept(self):
        contexts = np.asarray([[1.0, 2.0]])
        proteins = np.asarray([[3.0, 1.0]])
        features = context_pair_features(contexts, proteins)
        np.testing.assert_allclose(
            features, [[1.0, 2.0, 3.0, 1.0, 3.0, 2.0, 2.0, 1.0]]
        )
        self.assertEqual(explicit_intercept(features)[0, 0], 1.0)
        self.assertEqual(explicit_intercept(features, difference=True)[0, 0], 0.0)

    def test_standardization_uses_factual_fit_statistics(self):
        fit = np.asarray([[1.0, 10.0], [3.0, 10.0]])
        other = np.asarray([[5.0, 12.0]])
        mean, scale, transformed = standardize_factual_features(fit, other)
        np.testing.assert_allclose(mean, [2.0, 10.0])
        np.testing.assert_allclose(scale, [1.0, 1.0])
        np.testing.assert_allclose(transformed[1], [[3.0, 2.0]])


if __name__ == '__main__':
    unittest.main()
