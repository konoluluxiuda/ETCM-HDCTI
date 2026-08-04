import unittest

import numpy as np
import scipy.sparse as sp

from util.hplga import (
    hplga_complexity,
    hypergraph_pagerank,
    pagerank_linear_attention_numpy,
    pagerank_linear_attention_tf,
)
from util.model_components import resolve_hplga


class DummyConf(object):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def contains(self, key):
        return key in self.values


class HPLGATest(unittest.TestCase):
    def test_resolver_defaults_to_disabled(self):
        settings = resolve_hplga(DummyConf({}))
        self.assertFalse(settings['enabled'])
        self.assertFalse(settings['hc_enabled'])
        self.assertFalse(settings['pd_enabled'])
        self.assertEqual(settings['heads'], 2)
        self.assertEqual(settings['kernel'], 'elu_plus_one')

    def test_resolver_accepts_frozen_gate_zero_settings(self):
        settings = resolve_hplga(DummyConf({
            'hplga.enabled': 'True',
            'hplga.hc': 'True',
            'hplga.pd': 'False',
            'hplga.heads': '4',
            'hplga.pagerank.alpha': '0.9',
            'hplga.pagerank.max.iter': '50',
            'hplga.pagerank.tol': '1e-7',
            'hplga.epsilon': '1e-5',
        }))
        self.assertTrue(settings['enabled'])
        self.assertTrue(settings['hc_enabled'])
        self.assertFalse(settings['pd_enabled'])
        self.assertEqual(settings['heads'], 4)
        self.assertEqual(settings['pagerank_alpha'], 0.9)

    def test_resolver_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            resolve_hplga(DummyConf({
                'hplga.enabled': 'True',
                'hplga.hc': 'False',
                'hplga.pd': 'False',
            }))
        with self.assertRaises(ValueError):
            resolve_hplga(DummyConf({
                'hplga.enabled': 'True',
                'hplga.pagerank.alpha': '1.0',
            }))
        with self.assertRaises(ValueError):
            resolve_hplga(DummyConf({
                'hplga.enabled': 'True',
                'hplga.heads': '0',
            }))

    def test_hypergraph_pagerank_handles_zero_degree_nodes(self):
        incidence = sp.csr_matrix(np.asarray([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ], dtype=np.float32))
        prior, diagnostics = hypergraph_pagerank(incidence)
        self.assertEqual(prior.shape, (4,))
        self.assertTrue(np.all(np.isfinite(prior)))
        self.assertTrue(np.all(prior > 0))
        self.assertAlmostEqual(float(np.mean(prior)), 1.0, places=6)
        self.assertEqual(diagnostics['zero_degree_nodes'], 1)
        self.assertAlmostEqual(diagnostics['probability_sum'], 1.0, places=8)

    def test_linear_reordering_matches_explicit_kernel_attention(self):
        random = np.random.RandomState(7)
        queries = random.normal(size=(2, 5, 3))
        keys = random.normal(size=(2, 5, 3))
        values = random.normal(size=(2, 5, 3))
        prior = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5])
        actual = pagerank_linear_attention_numpy(
            queries, keys, values, prior
        )

        query_features = np.where(
            queries > 0, queries + 1.0, np.exp(queries)
        )
        key_features = np.where(
            keys > 0, keys + 1.0, np.exp(keys)
        )
        expected = []
        for head in range(queries.shape[0]):
            kernel = query_features[head].dot(key_features[head].T)
            kernel *= prior[None, :]
            kernel /= np.sum(kernel, axis=1, keepdims=True)
            expected.append(kernel.dot(values[head]))
        np.testing.assert_allclose(
            actual, np.asarray(expected), rtol=1e-10, atol=1e-10
        )

    def test_complexity_has_no_quadratic_attention_tensor(self):
        summary = hplga_complexity(19242, 64, 2)
        self.assertGreater(summary['dense_attention_pairs'], 700000000)
        self.assertEqual(summary['quadratic_attention_elements'], 0)
        self.assertEqual(summary['kernel_state_elements'], 2048)

    def test_zero_initialized_residual_is_exact_identity(self):
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        tf.reset_default_graph()
        embeddings = tf.constant(np.asarray([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32))
        weights = {
            name: tf.Variable(tf.eye(4, dtype=tf.float32))
            for name in ('q', 'k', 'v', 'output')
        }
        weights['gamma'] = tf.Variable(tf.zeros([1], dtype=tf.float32))
        output, _ = pagerank_linear_attention_tf(
            tf,
            embeddings,
            tf.constant([0.75, 1.25], dtype=tf.float32),
            weights,
            head_count=2,
            epsilon=1e-6,
            name='identity_hplga',
        )
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            actual, expected = session.run([output, embeddings])
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
