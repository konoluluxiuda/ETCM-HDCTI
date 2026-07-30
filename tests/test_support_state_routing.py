import unittest

import numpy as np
import tensorflow.compat.v1 as tf

from HDCTI import HDCTI
from util.config import ModelConf
from util.model_components import resolve_support_state_routing


BASELINE_CONFIG = (
    'configs/HDCTI_tcmsuite_four_state_no_context_unit_pilot.conf'
)
ROUTING_CONFIG = (
    'configs/HDCTI_tcmsuite_four_state_isolated_routing_unit_pilot.conf'
)

ROUTING_KEYS = {
    'context.interaction',
    'context.herb_protein',
    'context.herb_disease',
    'support.state.routing',
    'support.state.routing.mode',
    'support.state.routing.training',
    'support.state.routing.herb_protein.aux.weight',
    'support.state.routing.cold_cold.aux.weight',
    'support.state.routing.detach.cold_cold.features',
}


class SupportStateRoutingTest(unittest.TestCase):
    def test_frozen_pair_changes_only_routing_settings(self):
        baseline = dict(ModelConf(BASELINE_CONFIG).config)
        routing = dict(ModelConf(ROUTING_CONFIG).config)
        baseline.pop('model.variant')
        routing.pop('model.variant')
        for key in ROUTING_KEYS:
            baseline.pop(key, None)
            routing.pop(key, None)
        self.assertEqual(baseline, routing)

        settings = resolve_support_state_routing(
            ModelConf(ROUTING_CONFIG)
        )
        self.assertTrue(settings['enabled'])
        self.assertEqual(settings['mode'], 'hard_four_state')
        self.assertEqual(settings['training_mode'], 'isolated_heads')
        self.assertTrue(settings['detach_cold_cold_features'])
        self.assertEqual(settings['cold_cold_aux_weight'], 1.0)

    def test_cold_cold_auxiliary_head_detaches_encoder_features(self):
        tf.reset_default_graph()
        model = object.__new__(HDCTI)
        model.support_state_routing = {
            'enabled': True,
            'mode': 'hard_four_state',
            'training_mode': 'joint_ww',
            'herb_protein_aux_weight': 1.0,
            'cold_cold_aux_weight': 1.0,
            'detach_cold_cold_features': True,
        }
        model.u_context_embedding = tf.Variable(
            [[1.0, 2.0], [0.5, -1.0]], dtype=tf.float32
        )
        model.v_context_embedding = tf.Variable(
            [[0.2, 0.4], [1.5, 0.5]], dtype=tf.float32
        )
        head = tf.Variable([0.0, 0.0], dtype=tf.float32)
        model.weights = {'context_herb_disease': head}
        model.neg_disease_embedding = tf.constant(
            [1.0, 0.0], dtype=tf.float32
        )

        loss = model.buildSupportStateAuxiliaryLoss()
        gradients = tf.gradients(
            loss,
            [
                model.u_context_embedding,
                model.v_context_embedding,
                head,
            ],
        )
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertIsNotNone(gradients[2])

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            head_gradient = session.run(gradients[2])
        self.assertTrue(np.all(np.isfinite(head_gradient)))
        self.assertGreater(float(np.linalg.norm(head_gradient)), 0.0)

    def test_isolated_herb_protein_head_detaches_encoder_features(self):
        tf.reset_default_graph()
        model = object.__new__(HDCTI)
        model.support_state_routing = {
            'enabled': True,
            'mode': 'hard_four_state',
            'training_mode': 'isolated_heads',
            'herb_protein_aux_weight': 1.0,
            'cold_cold_aux_weight': 1.0,
            'detach_cold_cold_features': True,
        }
        model.u_context_embedding = tf.Variable(
            [[1.0, 2.0], [0.5, -1.0]], dtype=tf.float32
        )
        model.v_embedding = tf.Variable(
            [[0.3, 0.7], [1.0, 0.2]], dtype=tf.float32
        )
        model.v_context_embedding = tf.Variable(
            [[0.2, 0.4], [1.5, 0.5]], dtype=tf.float32
        )
        herb_protein_head = tf.Variable([0.0, 0.0], dtype=tf.float32)
        herb_disease_head = tf.Variable([0.0, 0.0], dtype=tf.float32)
        model.weights = {
            'context_herb_protein': herb_protein_head,
            'context_herb_disease': herb_disease_head,
        }
        model.neg_disease_embedding = tf.constant(
            [1.0, 0.0], dtype=tf.float32
        )

        herb_protein_loss, _ = (
            model.buildSupportStateAuxiliaryLosses()
        )
        gradients = tf.gradients(
            herb_protein_loss,
            [
                model.u_context_embedding,
                model.v_embedding,
                herb_protein_head,
            ],
        )
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertIsNotNone(gradients[2])


if __name__ == '__main__':
    unittest.main()
