import unittest
import numpy as np

from util.model_components import (
    EarlyStoppingTracker,
    build_regularization_loss,
    context_interaction_pair_scores,
    context_masked_pair_scores,
    context_interaction_scores,
    pair_decoder_scores,
    resolve_early_stopping,
    resolve_encoder_profile,
    resolve_context_terms,
    resolve_context_mask_training,
    resolve_counterfactual_context,
    resolve_herb_context_attention,
    resolve_inductive_context,
    resolve_negative_sampling,
    resolve_pair_decoder,
    resolve_support_experts,
    resolve_support_state_routing,
    support_conditioned_cold_gate,
    support_decoupled_base_gate,
    support_state_pair_gates,
    target_conditioned_herb_contexts,
)


class DummyConf(object):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def contains(self, key):
        return key in self.values


class ModelComponentsTest(unittest.TestCase):
    def test_encoder_profiles_separate_hdcti_from_dual_hgnn_baseline(self):
        default_profile = resolve_encoder_profile(DummyConf({}))
        self.assertEqual(default_profile['name'], 'hdcti')
        self.assertTrue(default_profile['use_self_gating'])
        self.assertTrue(default_profile['use_pagerank'])
        self.assertTrue(default_profile['use_dense_full_attention'])
        self.assertTrue(default_profile['use_node_dimension_attention'])
        self.assertFalse(default_profile['external_baseline'])

        dual_hgnn = resolve_encoder_profile(DummyConf({
            'encoder.profile': 'dual_hgnn_cti',
        }))
        self.assertEqual(dual_hgnn['name'], 'dual_hgnn_cti')
        self.assertFalse(dual_hgnn['use_self_gating'])
        self.assertFalse(dual_hgnn['use_pagerank'])
        self.assertFalse(dual_hgnn['use_dense_full_attention'])
        self.assertFalse(dual_hgnn['use_node_dimension_attention'])
        self.assertTrue(dual_hgnn['external_baseline'])

        with self.assertRaisesRegex(ValueError, 'encoder.profile'):
            resolve_encoder_profile(DummyConf({
                'encoder.profile': 'unknown',
            }))

    def test_inductive_context_defaults_off_and_supports_orthogonal_switches(self):
        self.assertEqual(resolve_inductive_context(DummyConf({})), {
            'enabled': False,
            'suppress_base_zero_support': False,
            'self_excluded': False,
        })
        base_only = resolve_inductive_context(DummyConf({
            'inductive.context': 'True',
            'inductive.context.suppress.base.zero.support': 'True',
            'inductive.context.self.excluded': 'False',
        }))
        self.assertEqual(base_only, {
            'enabled': True,
            'suppress_base_zero_support': True,
            'self_excluded': False,
        })
        excluded_only = resolve_inductive_context(DummyConf({
            'inductive.context': 'True',
            'inductive.context.suppress.base.zero.support': 'False',
            'inductive.context.self.excluded': 'True',
        }))
        self.assertEqual(excluded_only, {
            'enabled': True,
            'suppress_base_zero_support': False,
            'self_excluded': True,
        })
        with self.assertRaisesRegex(ValueError, 'requires base suppression'):
            resolve_inductive_context(DummyConf({
                'inductive.context': 'True',
                'inductive.context.suppress.base.zero.support': 'False',
                'inductive.context.self.excluded': 'False',
            }))

    def test_support_decoupled_base_gate_suppresses_only_supported_cold_context(self):
        gates = support_decoupled_base_gate(
            support_degrees=[0, 0, 1, 4],
            context_available=[1, 0, 1, 0],
        )
        np.testing.assert_array_equal(gates, [0.0, 1.0, 1.0, 1.0])

    def test_support_experts_use_a_frozen_hard_zero_support_route(self):
        settings = resolve_support_experts(DummyConf({
            'support.experts': 'True',
        }))
        self.assertEqual(settings, {
            'enabled': True,
            'mode': 'hard_zero_support',
            'pseudo_cold_ratio': 0.1,
            'seed': 72026,
            'detach_cold_features': True,
        })
        gates = support_conditioned_cold_gate(
            support_degrees=[0, 0, 1, 4],
            context_available=[1, 0, 1, 0],
        )
        np.testing.assert_array_equal(gates, [1.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, 'detach.cold.features'):
            resolve_support_experts(DummyConf({
                'support.experts': 'True',
                'support.experts.detach.cold.features': 'False',
            }))

    def test_support_state_routing_defaults_off_and_requires_isolation(self):
        self.assertEqual(
            resolve_support_state_routing(DummyConf({})),
            {
                'enabled': False,
                'mode': 'hard_four_state',
                'training_mode': 'joint_ww',
                'herb_protein_aux_weight': 1.0,
                'cold_cold_aux_weight': 1.0,
                'detach_cold_cold_features': True,
            },
        )
        enabled = resolve_support_state_routing(DummyConf({
            'support.state.routing': 'True',
            'support.state.routing.cold_cold.aux.weight': '0.5',
        }))
        self.assertTrue(enabled['enabled'])
        self.assertEqual(enabled['cold_cold_aux_weight'], 0.5)
        isolated = resolve_support_state_routing(DummyConf({
            'support.state.routing': 'True',
            'support.state.routing.training': 'isolated_heads',
            'support.state.routing.herb_protein.aux.weight': '0.25',
        }))
        self.assertEqual(isolated['training_mode'], 'isolated_heads')
        self.assertEqual(isolated['herb_protein_aux_weight'], 0.25)
        with self.assertRaisesRegex(ValueError, 'detached cold-cold'):
            resolve_support_state_routing(DummyConf({
                'support.state.routing': 'True',
                'support.state.routing.detach.cold_cold.features': 'False',
            }))

    def test_support_state_pair_gates_cover_four_states(self):
        gates = support_state_pair_gates(
            compound_support_degrees=[2, 0, 3, 0],
            protein_support_degrees=[4, 5, 0, 0],
            compound_context_available=[1, 1, 1, 1],
            protein_context_available=[1, 1, 1, 1],
        )
        np.testing.assert_array_equal(gates['state'], [0, 1, 2, 3])
        np.testing.assert_array_equal(gates['base'], [1, 0, 1, 0])
        np.testing.assert_array_equal(
            gates['herb_protein'], [1, 1, 0, 0]
        )
        np.testing.assert_array_equal(
            gates['herb_disease'], [0, 0, 0, 1]
        )

    def test_support_state_pair_gates_require_side_context(self):
        gates = support_state_pair_gates(
            compound_support_degrees=[0, 0],
            protein_support_degrees=[2, 0],
            compound_context_available=[0, 1],
            protein_context_available=[1, 0],
        )
        np.testing.assert_array_equal(
            gates['herb_protein'], [0, 0]
        )
        np.testing.assert_array_equal(
            gates['herb_disease'], [0, 0]
        )

    def test_pair_scores_route_warm_and_cold_experts_without_overlap(self):
        compounds = np.asarray(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        )
        proteins = np.asarray(
            [[0.5, 1.5], [2.0, 1.0]], dtype=np.float32
        )
        herb_contexts = np.asarray(
            [[0.2, 0.8], [0.6, 0.4]], dtype=np.float32
        )
        protein_contexts = np.zeros((2, 2), dtype=np.float32)
        zero = np.zeros(2, dtype=np.float32)
        warm_weight = np.asarray([2.0, 3.0], dtype=np.float32)
        cold_weight = np.asarray([-1.0, 4.0], dtype=np.float32)
        warm_scale = np.asarray([1.0, 0.0], dtype=np.float32)
        cold_scale = 1.0 - warm_scale

        scores = context_interaction_pair_scores(
            compounds,
            proteins,
            herb_contexts,
            protein_contexts,
            compound_indices=[0, 1],
            protein_indices=[0, 1],
            compound_disease_weight=zero,
            herb_protein_weight=warm_weight,
            herb_disease_weight=zero,
            enabled_terms={
                'compound_disease': False,
                'herb_protein': True,
                'herb_disease': False,
            },
            herb_protein_scale=warm_scale,
            base_score_scale=warm_scale,
            cold_herb_protein_weight=cold_weight,
            cold_herb_protein_scale=cold_scale,
        )
        expected_warm = (
            np.sum(compounds[0] * proteins[0])
            + np.sum(herb_contexts[0] * proteins[0] * warm_weight)
        )
        expected_cold = np.sum(
            herb_contexts[1] * proteins[1] * cold_weight
        )
        np.testing.assert_allclose(
            scores, [expected_warm, expected_cold]
        )

    def test_pair_scores_can_suppress_only_the_base_decoder_term(self):
        compounds = np.asarray([[1.0, 2.0]], dtype=np.float32)
        proteins = np.asarray([[3.0, 4.0]], dtype=np.float32)
        herb_contexts = np.asarray([[0.5, 0.25]], dtype=np.float32)
        disease_contexts = np.zeros((1, 2), dtype=np.float32)
        zero = np.zeros(2, dtype=np.float32)
        herb_weight = np.asarray([2.0, 3.0], dtype=np.float32)
        score = context_interaction_pair_scores(
            compounds,
            proteins,
            herb_contexts,
            disease_contexts,
            [0],
            [0],
            zero,
            herb_weight,
            zero,
            enabled_terms={
                'compound_disease': False,
                'herb_protein': True,
                'herb_disease': False,
            },
            base_score_scale=[0.0],
        )
        expected = np.sum(herb_contexts[0] * proteins[0] * herb_weight)
        np.testing.assert_allclose(score, [expected])

    def test_pair_scores_apply_independent_herb_disease_scale(self):
        compounds = np.ones((2, 2), dtype=np.float32)
        proteins = np.ones((2, 2), dtype=np.float32)
        herb_contexts = np.asarray(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        )
        disease_contexts = np.asarray(
            [[0.5, 1.5], [2.0, 1.0]], dtype=np.float32
        )
        zero = np.zeros(2, dtype=np.float32)
        herb_disease = np.asarray([2.0, -1.0], dtype=np.float32)
        scores = context_interaction_pair_scores(
            compounds,
            proteins,
            herb_contexts,
            disease_contexts,
            [0, 1],
            [0, 1],
            zero,
            zero,
            herb_disease,
            enabled_terms={
                'compound_disease': False,
                'herb_protein': False,
                'herb_disease': True,
            },
            base_score_scale=[0.0, 0.0],
            herb_disease_scale=[0.0, 1.0],
        )
        expected_second = np.sum(
            herb_contexts[1] * disease_contexts[1] * herb_disease
        )
        np.testing.assert_allclose(scores, [0.0, expected_second])

    def test_counterfactual_context_defaults_off_and_validates_pilot_settings(self):
        self.assertEqual(resolve_counterfactual_context(DummyConf({})), {
            'enabled': False,
            'weight': 0.05,
            'margin': 0.2,
            'draws': 20,
            'seed': 42026,
            'match': 'exact_hc_degree_disjoint',
        })
        settings = resolve_counterfactual_context(DummyConf({
            'counterfactual.context': 'True',
            'counterfactual.weight': '0.05',
            'counterfactual.margin': '0.2',
            'counterfactual.draws': '20',
            'counterfactual.seed': '42026',
            'counterfactual.match': 'exact_hc_degree_disjoint',
        }))
        self.assertTrue(settings['enabled'])
        with self.assertRaisesRegex(ValueError, 'counterfactual.margin'):
            resolve_counterfactual_context(DummyConf({
                'counterfactual.context': 'True',
                'counterfactual.margin': '0',
            }))

    def test_herb_context_attention_defaults_to_frozen_static_mode(self):
        settings = resolve_herb_context_attention(DummyConf({}))
        self.assertEqual(settings, {'mode': 'static', 'temperature': 1.0})

    def test_herb_context_attention_configuration_is_validated(self):
        settings = resolve_herb_context_attention(DummyConf({
            'context.herb_protein.mode': 'target_attention',
            'context.herb_attention.temperature': '0.5',
        }))
        self.assertEqual(settings, {'mode': 'target_attention', 'temperature': 0.5})
        residual_settings = resolve_herb_context_attention(DummyConf({
            'context.herb_protein.mode': 'target_residual_attention',
        }))
        self.assertEqual(
            residual_settings,
            {'mode': 'target_residual_attention', 'temperature': 1.0},
        )
        with self.assertRaisesRegex(ValueError, 'target_residual_attention'):
            resolve_herb_context_attention(DummyConf({
                'context.herb_protein.mode': 'global_attention',
            }))
        with self.assertRaisesRegex(ValueError, 'temperature must be positive'):
            resolve_herb_context_attention(DummyConf({
                'context.herb_attention.temperature': '0',
            }))

    def test_target_conditioned_context_changes_with_candidate_protein(self):
        herb_edges = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        incidence = np.asarray([[0, 1]], dtype=np.int64)
        mask = np.asarray([[1.0, 1.0]], dtype=np.float32)
        proteins = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)

        contexts, attention = target_conditioned_herb_contexts(
            herb_edges,
            incidence,
            mask,
            proteins,
            compound_indices=[0, 0],
            protein_indices=[0, 1],
            herb_projection=np.eye(2, dtype=np.float32),
            protein_projection=np.eye(2, dtype=np.float32),
        )

        self.assertGreater(attention[0, 0], attention[0, 1])
        self.assertGreater(attention[1, 1], attention[1, 0])
        self.assertGreater(contexts[0, 0], contexts[0, 1])
        self.assertGreater(contexts[1, 1], contexts[1, 0])
        np.testing.assert_allclose(np.sum(attention, axis=1), np.ones(2))
        np.testing.assert_allclose(np.linalg.norm(contexts, axis=1), np.ones(2))

    def test_target_conditioned_context_excludes_padding(self):
        herb_edges = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        incidence = np.asarray([[0, 2]], dtype=np.int64)
        mask = np.asarray([[1.0, 0.0]], dtype=np.float32)
        proteins = np.asarray([[0.0, 1.0]], dtype=np.float32)

        contexts, attention = target_conditioned_herb_contexts(
            herb_edges,
            incidence,
            mask,
            proteins,
            compound_indices=[0],
            protein_indices=[0],
            herb_projection=np.eye(2, dtype=np.float32),
            protein_projection=np.eye(2, dtype=np.float32),
        )

        np.testing.assert_allclose(attention, [[1.0, 0.0]])
        np.testing.assert_allclose(contexts, [[0.6, 0.8]])

    def test_zero_initialized_target_residual_preserves_static_context_score(self):
        compounds = np.asarray([[0.2, 0.4]], dtype=np.float32)
        proteins = np.asarray([[0.5, 0.7]], dtype=np.float32)
        static_contexts = np.asarray([[0.1, 0.9]], dtype=np.float32)
        target_contexts = np.asarray([[0.8, 0.2]], dtype=np.float32)
        protein_contexts = np.zeros((1, 2), dtype=np.float32)
        herb_protein_weight = np.asarray([0.3, -0.2], dtype=np.float32)
        zero_weight = np.zeros(2, dtype=np.float32)
        enabled_terms = {
            'compound_disease': False,
            'herb_protein': True,
            'herb_disease': False,
        }

        static_scores = context_interaction_pair_scores(
            compounds,
            proteins,
            static_contexts,
            protein_contexts,
            [0],
            [0],
            zero_weight,
            herb_protein_weight,
            zero_weight,
            enabled_terms=enabled_terms,
        )
        residual_scores = context_interaction_pair_scores(
            compounds,
            proteins,
            static_contexts,
            protein_contexts,
            [0],
            [0],
            zero_weight,
            herb_protein_weight,
            zero_weight,
            enabled_terms=enabled_terms,
            residual_compound_contexts=target_contexts,
            target_residual_weight=zero_weight,
        )

        np.testing.assert_allclose(residual_scores, static_scores)

    def test_target_residual_adds_only_conditioned_context_delta(self):
        compounds = np.asarray([[0.2, 0.4]], dtype=np.float32)
        proteins = np.asarray([[0.5, 0.7]], dtype=np.float32)
        static_contexts = np.asarray([[0.1, 0.9]], dtype=np.float32)
        target_contexts = np.asarray([[0.8, 0.2]], dtype=np.float32)
        protein_contexts = np.zeros((1, 2), dtype=np.float32)
        herb_protein_weight = np.asarray([0.3, -0.2], dtype=np.float32)
        residual_weight = np.asarray([0.4, 0.6], dtype=np.float32)
        zero_weight = np.zeros(2, dtype=np.float32)
        enabled_terms = {
            'compound_disease': False,
            'herb_protein': True,
            'herb_disease': False,
        }

        actual = context_interaction_pair_scores(
            compounds,
            proteins,
            static_contexts,
            protein_contexts,
            [0],
            [0],
            zero_weight,
            herb_protein_weight,
            zero_weight,
            enabled_terms=enabled_terms,
            residual_compound_contexts=target_contexts,
            target_residual_weight=residual_weight,
        )
        expected = (
            np.sum(compounds[0] * proteins[0])
            + np.sum(static_contexts[0] * proteins[0] * herb_protein_weight)
            + np.sum(
                (target_contexts[0] - static_contexts[0])
                * proteins[0]
                * residual_weight
            )
        )

        np.testing.assert_allclose(actual, [expected])

    def test_context_mask_replaces_only_requested_id_embedding(self):
        compounds = np.asarray([[1.0, 2.0]], dtype=np.float32)
        proteins = np.asarray([[3.0, 4.0]], dtype=np.float32)
        herb_contexts = np.asarray([[0.5, 0.25]], dtype=np.float32)
        disease_contexts = np.asarray([[0.2, 0.4]], dtype=np.float32)
        zero = np.zeros(2, dtype=np.float32)
        herb_weight = np.asarray([2.0, 3.0], dtype=np.float32)
        terms = {
            'compound_disease': False,
            'herb_protein': True,
            'herb_disease': False,
        }
        compound_masked = context_masked_pair_scores(
            compounds, proteins, herb_contexts, disease_contexts,
            [0], [0], zero, herb_weight, zero,
            mask_compound=True, enabled_terms=terms,
        )
        protein_masked = context_masked_pair_scores(
            compounds, proteins, herb_contexts, disease_contexts,
            [0], [0], zero, herb_weight, zero,
            mask_protein=True, enabled_terms=terms,
        )
        expected_compound = np.sum(herb_contexts[0] * proteins[0])
        expected_compound += np.sum(
            herb_contexts[0] * proteins[0] * herb_weight
        )
        expected_protein = np.sum(compounds[0] * disease_contexts[0])
        expected_protein += np.sum(
            herb_contexts[0] * disease_contexts[0] * herb_weight
        )
        np.testing.assert_allclose(compound_masked, [expected_compound])
        np.testing.assert_allclose(protein_masked, [expected_protein])

    def test_negative_sampling_defaults_to_random(self):
        settings = resolve_negative_sampling(DummyConf({}))
        self.assertEqual(settings, {'strategy': 'random', 'hard_ratio': 0.25})

    def test_context_mask_training_configuration_is_validated(self):
        self.assertEqual(
            resolve_context_mask_training(DummyConf({})),
            {'enabled': False, 'side': 'compound', 'weight': 0.1},
        )
        settings = resolve_context_mask_training(DummyConf({
            'context.mask.training': 'True',
            'context.mask.side': 'compound',
            'context.mask.weight': '0.1',
        }))
        self.assertTrue(settings['enabled'])
        with self.assertRaisesRegex(ValueError, 'context.mask.side'):
            resolve_context_mask_training(DummyConf({'context.mask.side': 'herb'}))
        with self.assertRaisesRegex(ValueError, 'context.mask.weight'):
            resolve_context_mask_training(DummyConf({
                'context.mask.training': 'True',
                'context.mask.weight': '0',
            }))

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
