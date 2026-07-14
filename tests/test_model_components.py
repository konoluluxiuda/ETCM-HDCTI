import unittest
from pathlib import Path

import numpy as np

from util.config import ModelConf
from util.model_components import (
    EarlyStoppingTracker,
    build_regularization_loss,
    context_interaction_pair_scores,
    context_interaction_scores,
    pair_decoder_scores,
    resolve_early_stopping,
    resolve_context_terms,
    resolve_negative_sampling,
    resolve_pair_decoder,
)


class DummyConf(object):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def contains(self, key):
        return key in self.values


class ModelComponentsTest(unittest.TestCase):
    def test_negative_sampling_defaults_to_random(self):
        settings = resolve_negative_sampling(DummyConf({}))
        self.assertEqual(settings, {'strategy': 'random', 'hard_ratio': 0.25})

    def test_negative_sampling_configuration_is_validated(self):
        settings = resolve_negative_sampling(DummyConf({
            'negative.strategy': 'mixed',
            'negative.hard.ratio': '0.4',
        }))
        self.assertEqual(settings, {'strategy': 'mixed', 'hard_ratio': 0.4})
        with self.assertRaisesRegex(ValueError, 'negative.strategy'):
            resolve_negative_sampling(DummyConf({'negative.strategy': 'dynamic'}))
        with self.assertRaisesRegex(ValueError, 'negative.hard.ratio'):
            resolve_negative_sampling(DummyConf({'negative.hard.ratio': '1.1'}))
        with self.assertRaisesRegex(ValueError, 'negative.hard.ratio'):
            resolve_negative_sampling(DummyConf({
                'negative.strategy': 'mixed',
                'negative.hard.ratio': '0',
            }))

    def test_pair_decoder_configuration_defaults_to_dot(self):
        settings = resolve_pair_decoder(DummyConf({}))
        self.assertEqual(settings['type'], 'dot')
        self.assertEqual(settings['hidden_size'], 64)

    def test_pair_decoder_configuration_rejects_unknown_type(self):
        with self.assertRaisesRegex(ValueError, 'pair.decoder'):
            resolve_pair_decoder(DummyConf({'pair.decoder': 'transformer'}))

    def test_bilinear_identity_and_zero_residual_mlp_start_as_dot(self):
        compounds = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
        proteins = np.asarray([[0.5, 3.0], [2.0, -2.0]], dtype=np.float32)
        expected = np.sum(compounds * proteins, axis=1)

        bilinear = pair_decoder_scores(
            compounds,
            proteins,
            decoder_type='bilinear',
            decoder_weights={'decoder_bilinear': np.eye(2, dtype=np.float32)},
        )
        mlp = pair_decoder_scores(
            compounds,
            proteins,
            decoder_type='mlp',
            decoder_weights={
                'decoder_mlp_hidden': np.ones((8, 3), dtype=np.float32),
                'decoder_mlp_hidden_bias': np.zeros(3, dtype=np.float32),
                'decoder_mlp_output': np.zeros((3, 1), dtype=np.float32),
                'decoder_mlp_output_bias': np.zeros(1, dtype=np.float32),
            },
        )

        np.testing.assert_allclose(bilinear, expected)
        np.testing.assert_allclose(mlp, expected)

    def test_pair_scores_match_selected_full_matrix_entries(self):
        rng = np.random.RandomState(7)
        compounds = rng.normal(size=(4, 3)).astype(np.float32)
        proteins = rng.normal(size=(5, 3)).astype(np.float32)
        herb_contexts = rng.normal(size=(4, 3)).astype(np.float32)
        disease_contexts = rng.normal(size=(5, 3)).astype(np.float32)
        weights = [rng.normal(size=3).astype(np.float32) for _ in range(3)]
        terms = {'compound_disease': False, 'herb_protein': True, 'herb_disease': False}
        compound_indices = [0, 3, 1]
        protein_indices = [2, 4, 0]

        full = context_interaction_scores(
            compounds, proteins, herb_contexts, disease_contexts,
            weights[0], weights[1], weights[2], enabled_terms=terms,
        )
        pairs = context_interaction_pair_scores(
            compounds, proteins, herb_contexts, disease_contexts,
            compound_indices, protein_indices,
            weights[0], weights[1], weights[2], enabled_terms=terms,
        )

        np.testing.assert_allclose(
            pairs, full[compound_indices, protein_indices], rtol=1e-6, atol=1e-6
        )

    def test_early_stopping_tracker_uses_patience_and_min_delta(self):
        tracker = EarlyStoppingTracker(patience=2, min_delta=0.01)

        self.assertEqual(tracker.update(0.70, 2), (True, False))
        self.assertEqual(tracker.update(0.705, 4), (False, False))
        self.assertEqual(tracker.update(0.709, 6), (False, True))
        self.assertEqual(tracker.best_epoch, 2)
        self.assertAlmostEqual(tracker.best_value, 0.70)

    def test_early_stopping_configuration_is_validated(self):
        conf = DummyConf({
            'early.stopping': 'True',
            'validation.ratio': '0.1',
            'validation.metric': 'AUPR',
            'validation.interval': '2',
            'validation.patience': '5',
            'validation.min.delta': '0.0001',
        })

        settings = resolve_early_stopping(conf)

        self.assertTrue(settings['enabled'])
        self.assertEqual(settings['metric'], 'aupr')
        self.assertEqual(settings['interval'], 2)
        self.assertEqual(settings['patience'], 5)

    def test_context_interaction_matches_pairwise_formula(self):
        compounds = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        proteins = np.asarray([[0.5, 1.0], [1.5, -1.0]], dtype=np.float32)
        herb_contexts = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        disease_contexts = np.asarray([[1.0, 0.3], [-0.5, 0.7]], dtype=np.float32)
        compound_disease = np.asarray([0.1, 0.2], dtype=np.float32)
        herb_protein = np.asarray([0.3, -0.1], dtype=np.float32)
        herb_disease = np.asarray([-0.2, 0.4], dtype=np.float32)

        actual = context_interaction_scores(
            compounds,
            proteins,
            herb_contexts,
            disease_contexts,
            compound_disease,
            herb_protein,
            herb_disease,
        )
        expected = np.zeros((2, 2), dtype=np.float32)
        for compound_index in range(2):
            for protein_index in range(2):
                expected[compound_index, protein_index] = (
                    np.sum(compounds[compound_index] * proteins[protein_index])
                    + np.sum(compounds[compound_index] * disease_contexts[protein_index] * compound_disease)
                    + np.sum(herb_contexts[compound_index] * proteins[protein_index] * herb_protein)
                    + np.sum(herb_contexts[compound_index] * disease_contexts[protein_index] * herb_disease)
                )

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_zero_context_weights_preserve_dot_product_baseline(self):
        compounds = np.asarray([[1.0, 2.0]], dtype=np.float32)
        proteins = np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        context = np.asarray([[0.5, 0.5]], dtype=np.float32)
        protein_contexts = np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
        zeros = np.zeros(2, dtype=np.float32)

        actual = context_interaction_scores(
            compounds, proteins, context, protein_contexts, zeros, zeros, zeros
        )

        np.testing.assert_allclose(actual, compounds.dot(proteins.transpose()))

    def test_disabled_context_term_is_excluded_from_ranking_scores(self):
        compounds = np.asarray([[1.0, 2.0]], dtype=np.float32)
        proteins = np.asarray([[3.0, 4.0]], dtype=np.float32)
        herb_contexts = np.asarray([[0.5, 0.25]], dtype=np.float32)
        disease_contexts = np.asarray([[0.2, 0.3]], dtype=np.float32)
        weights = np.ones(2, dtype=np.float32)

        actual = context_interaction_scores(
            compounds,
            proteins,
            herb_contexts,
            disease_contexts,
            weights,
            weights,
            weights,
            enabled_terms={
                'compound_disease': True,
                'herb_protein': True,
                'herb_disease': False,
            },
        )
        expected = (
            compounds.dot(proteins.transpose())
            + (compounds * weights).dot(disease_contexts.transpose())
            + (herb_contexts * weights).dot(proteins.transpose())
        )
        np.testing.assert_allclose(actual, expected)

    def test_context_terms_default_to_all_on_under_master_switch(self):
        terms = resolve_context_terms(DummyConf({'context.interaction': 'True'}))
        self.assertEqual(terms, {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        })

    def test_context_terms_can_be_switched_independently(self):
        terms = resolve_context_terms(DummyConf({
            'context.interaction': 'True',
            'context.compound_disease': 'True',
            'context.herb_protein': 'False',
            'context.herb_disease': 'False',
        }))
        self.assertEqual(terms, {
            'compound_disease': True,
            'herb_protein': False,
            'herb_disease': False,
        })

    def test_master_switch_disables_all_context_terms(self):
        terms = resolve_context_terms(DummyConf({
            'context.interaction': 'False',
            'context.compound_disease': 'True',
            'context.herb_protein': 'True',
            'context.herb_disease': 'True',
        }))
        self.assertFalse(any(terms.values()))

    def test_invalid_context_switch_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_context_terms(DummyConf({
                'context.interaction': 'True',
                'context.herb_disease': 'maybe',
            }))

    def test_side_context_experiment_configs_enable_one_term_each(self):
        repository_root = Path(__file__).resolve().parents[1]
        disease_terms = resolve_context_terms(ModelConf(
            str(repository_root / 'configs' / 'HDCTI_disease_only.conf')
        ))
        herb_terms = resolve_context_terms(ModelConf(
            str(repository_root / 'configs' / 'HDCTI_herb_only.conf')
        ))
        self.assertEqual(disease_terms, {
            'compound_disease': True,
            'herb_protein': False,
            'herb_disease': False,
        })
        self.assertEqual(herb_terms, {
            'compound_disease': False,
            'herb_protein': True,
            'herb_disease': False,
        })

    def test_embedding_regularization_is_added_once(self):
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        tf.reset_default_graph()
        weights = {
            'first': tf.constant([1.0, 2.0]),
            'second': tf.constant([3.0]),
        }
        compounds = tf.constant([[1.0, 1.0]])
        proteins = tf.constant([[2.0, 2.0]])
        loss = build_regularization_loss(
            tf, weights, compounds, proteins, 0.1, 0.2, 0.3
        )

        with tf.Session() as session:
            actual = session.run(loss)

        expected = (
            0.1 * (0.5 * (1.0 ** 2 + 2.0 ** 2) + 0.5 * 3.0 ** 2)
            + 0.2 * 0.5 * (1.0 ** 2 + 1.0 ** 2)
            + 0.3 * 0.5 * (2.0 ** 2 + 2.0 ** 2)
        )
        self.assertAlmostEqual(float(actual), expected, places=6)


if __name__ == '__main__':
    unittest.main()
