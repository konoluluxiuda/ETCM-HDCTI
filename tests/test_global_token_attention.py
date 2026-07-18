import unittest

import numpy as np

from util.global_token_attention import (
    global_token_attention_complexity,
    summarize_global_token_attention,
)
from util.model_components import resolve_global_token_attention


class DummyConf(object):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def contains(self, key):
        return key in self.values


class GlobalTokenAttentionTest(unittest.TestCase):
    def test_resolver_defaults_to_disabled_hyperedge_induced_attention(self):
        settings = resolve_global_token_attention(DummyConf({}))
        self.assertFalse(settings['enabled'])
        self.assertFalse(settings['hc_enabled'])
        self.assertFalse(settings['pd_enabled'])
        self.assertEqual(settings['tokens'], 32)
        self.assertEqual(settings['heads'], 2)

    def test_resolver_accepts_paired_pilot_settings(self):
        settings = resolve_global_token_attention(DummyConf({
            'global.token.attention': 'True',
            'global.token.attention.mode': 'hyperedge_induced',
            'global.token.attention.hc': 'True',
            'global.token.attention.pd': 'False',
            'global.token.attention.tokens': '16',
            'global.token.attention.heads': '4',
            'global.token.attention.temperature': '0.5',
        }))
        self.assertTrue(settings['enabled'])
        self.assertTrue(settings['hc_enabled'])
        self.assertFalse(settings['pd_enabled'])
        self.assertEqual(settings['tokens'], 16)
        self.assertEqual(settings['heads'], 4)
        self.assertEqual(settings['temperature'], 0.5)

    def test_resolver_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            resolve_global_token_attention(DummyConf({
                'global.token.attention': 'True',
                'global.token.attention.tokens': '0',
            }))
        with self.assertRaises(ValueError):
            resolve_global_token_attention(DummyConf({
                'global.token.attention': 'True',
                'global.token.attention.hc': 'False',
                'global.token.attention.pd': 'False',
            }))

    def test_complexity_is_linear_in_nodes_and_hyperedges(self):
        summary = global_token_attention_complexity(
            node_count=1000,
            hyperedge_count=200,
            token_count=32,
            head_count=2,
        )
        self.assertEqual(summary['dense_node_pairs'], 2000000)
        self.assertEqual(summary['token_attention_pairs'], 70400)
        self.assertLess(summary['pair_ratio'], 0.04)

    def test_diagnostics_report_finite_attention_statistics(self):
        edge_assignments = np.asarray([
            [0.7, 0.1],
            [0.2, 0.2],
            [0.1, 0.7],
        ], dtype=np.float32)
        node_attention = np.asarray([[
            [0.8, 0.2],
            [0.4, 0.6],
        ]], dtype=np.float32)
        tokens = np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        summary = summarize_global_token_attention(
            edge_assignments, node_attention, tokens
        )
        self.assertEqual(summary['active_hyperedges'], 3)
        self.assertGreater(summary['edge_assignment_entropy_mean'], 0.0)
        self.assertLessEqual(summary['node_attention_entropy_mean'], 1.0)
        self.assertAlmostEqual(
            summary['token_mean_abs_off_diagonal_cosine'], 0.0
        )


if __name__ == '__main__':
    unittest.main()
