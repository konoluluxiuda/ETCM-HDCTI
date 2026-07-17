import unittest

import numpy as np

from util.hyperedge_attention import (
    factorized_incidence_attention,
    hyperedge_specificity_prior,
    ordered_incidence_ids,
    segment_softmax,
)
from util.model_components import resolve_hyperedge_attention


class DummyConf(object):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def contains(self, key):
        return key in self.values


class HyperedgeAttentionTest(unittest.TestCase):
    def test_incidence_ids_are_preordered_in_both_directions(self):
        ordered = ordered_incidence_ids(
            edge_ids=[1, 0, 1, 0],
            node_ids=[0, 2, 1, 0],
        )
        self.assertEqual(
            list(zip(
                ordered['forward_edge_ids'], ordered['forward_node_ids']
            )),
            [(0, 0), (0, 2), (1, 0), (1, 1)],
        )
        self.assertEqual(
            list(zip(
                ordered['reverse_node_ids'], ordered['reverse_edge_ids']
            )),
            [(0, 0), (0, 1), (1, 1), (2, 0)],
        )

    def test_configuration_defaults_off_and_validates_enabled_mode(self):
        defaults = resolve_hyperedge_attention(DummyConf({}))
        self.assertFalse(defaults['enabled'])
        self.assertFalse(defaults['hc_enabled'])
        self.assertFalse(defaults['pd_enabled'])

        enabled = resolve_hyperedge_attention(DummyConf({
            'hyperedge.attention': 'True',
            'hyperedge.attention.hc': 'True',
            'hyperedge.attention.pd': 'False',
            'hyperedge.attention.temperature': '0.5',
            'hyperedge.attention.prior.scale': '0.2',
        }))
        self.assertTrue(enabled['hc_enabled'])
        self.assertFalse(enabled['pd_enabled'])
        self.assertEqual(enabled['temperature'], 0.5)
        self.assertEqual(enabled['prior_scale'], 0.2)

        with self.assertRaisesRegex(ValueError, 'At least one'):
            resolve_hyperedge_attention(DummyConf({
                'hyperedge.attention': 'True',
                'hyperedge.attention.hc': 'False',
                'hyperedge.attention.pd': 'False',
            }))

    def test_specificity_prior_favors_lower_degree_hyperedges(self):
        prior = hyperedge_specificity_prior([1, 2, 8], node_count=16)
        self.assertGreater(prior[0], prior[1])
        self.assertGreater(prior[1], prior[2])
        self.assertAlmostEqual(float(np.mean(prior)), 0.0, places=6)

    def test_segment_softmax_normalizes_each_segment(self):
        weights = segment_softmax([0.0, 1.0, 0.0], [0, 0, 1], 2)
        self.assertAlmostEqual(float(weights[0] + weights[1]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[2]), 1.0, places=6)
        self.assertGreater(weights[1], weights[0])

    def test_factorized_attention_is_uniform_without_logits_or_prior(self):
        node_to_edge, edge_to_node = factorized_incidence_attention(
            node_logits=np.zeros(3),
            edge_logits=np.zeros(2),
            edge_ids=[0, 0, 1],
            node_ids=[0, 1, 1],
            edge_count=2,
            node_count=3,
            specificity_prior=np.zeros(2),
            prior_scale=0.0,
        )
        np.testing.assert_allclose(node_to_edge, [0.5, 0.5, 1.0])
        np.testing.assert_allclose(edge_to_node, [1.0, 0.5, 0.5])

    def test_specificity_prior_changes_only_edge_to_node_routing(self):
        node_to_edge, edge_to_node = factorized_incidence_attention(
            node_logits=np.zeros(2),
            edge_logits=np.zeros(2),
            edge_ids=[0, 1],
            node_ids=[0, 0],
            edge_count=2,
            node_count=2,
            specificity_prior=[1.0, -1.0],
            prior_scale=1.0,
        )
        np.testing.assert_allclose(node_to_edge, [1.0, 1.0])
        self.assertGreater(edge_to_node[0], edge_to_node[1])


if __name__ == '__main__':
    unittest.main()
