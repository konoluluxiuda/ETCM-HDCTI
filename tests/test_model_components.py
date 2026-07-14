import unittest

import numpy as np

from util.model_components import build_regularization_loss, context_interaction_scores


class ModelComponentsTest(unittest.TestCase):
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
