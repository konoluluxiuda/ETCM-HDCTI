import unittest

import numpy as np

from util.side_hypergraph_roles import (
    ROLE_FEATURE_NAMES,
    build_role_features,
    degree_pair_features,
    empirical_percentile_roles,
    pair_role_features,
    scalar_pair_features,
    sample_degree_matched_negatives,
)


class SideHypergraphRoleTest(unittest.TestCase):
    def test_role_features_include_isolated_nodes(self):
        result = build_role_features(
            {'h1': {'c1', 'c2'}, 'h2': {'c2', 'c3'}},
            node_universe=('c1', 'c2', 'c3', 'c4'),
        )
        features = result['features']
        self.assertEqual(features.shape, (4, len(ROLE_FEATURE_NAMES)))
        self.assertEqual(result['degrees'].tolist(), [1, 2, 1, 0])
        self.assertEqual(features[result['node_index']['c4'], 0], 0.0)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_pair_feature_dimensions(self):
        left = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        right = np.asarray([[2.0, 1.0], [4.0, 3.0]])
        self.assertEqual(pair_role_features(left, right).shape, (2, 8))
        self.assertEqual(degree_pair_features([1, 2], [3, 4]).shape, (2, 4))
        self.assertEqual(scalar_pair_features([1, 2], [3, 4]).shape, (2, 4))

    def test_percentile_roles_preserve_isolated_and_ties(self):
        role = build_role_features(
            {'h1': {'c1', 'c2'}, 'h2': {'c3'}},
            node_universe=('c1', 'c2', 'c3', 'c4'),
        )
        transformed = empirical_percentile_roles(role)
        features = transformed['features']
        c1 = role['node_index']['c1']
        c2 = role['node_index']['c2']
        c4 = role['node_index']['c4']
        self.assertTrue(np.allclose(features[c1], features[c2]))
        self.assertEqual(features[c4, 0], 0.0)
        self.assertTrue(np.all(features[c4, 1:] == 0.0))
        self.assertTrue(np.all(features[:, 1:] >= 0.0))
        self.assertTrue(np.all(features[:, 1:] <= 1.0))

    def test_degree_matched_negatives_exclude_positives(self):
        positive_pairs = [('c1', 'p1'), ('c2', 'p2')]
        matched_positives, negatives, audit = sample_degree_matched_negatives(
            positive_pairs,
            compound_ids=('c1', 'c2', 'c3', 'c4'),
            compound_degrees=(1, 1, 1, 1),
            protein_ids=('p1', 'p2', 'p3', 'p4'),
            protein_degrees=(1, 1, 1, 1),
            seed=7,
            bin_count=2,
        )
        self.assertEqual(audit['coverage'], 1.0)
        self.assertEqual(matched_positives, positive_pairs)
        self.assertEqual(len(negatives), 2)
        self.assertFalse(set(negatives).intersection(positive_pairs))


if __name__ == '__main__':
    unittest.main()
